import numpy as np

from flybrain import FlyBrain, Trace

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyfly.sensory_map import create_disjoint_sensory_map


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

NEURONS_PER_TOKEN = 64
AMPLITUDE = 0.8

WARMUP_STEPS = 10
TOKEN_STEPS = 8
GAP_STEPS = 1

TRACE_TAU = 0.25

MAPPING_SEED = 42

TRAIN_SEEDS = list(range(40))
TEST_SEEDS = list(range(40, 60))

POST_STEPS = 4


# ============================================================
# Load brain
# ============================================================

print()
print("========================================")
print("   PyFly STRICT ORDER MEMORY PROBE")
print("========================================")
print()

print("Loading FlyBrain...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")


# ============================================================
# Input mapping
# ============================================================

visual_pool = np.asarray(
    brain.cells(["visual_projection"])
)

TOKENS = [
    "A",
    "B",
    "C",
]


sensory_map = create_disjoint_sensory_map(
    tokens=TOKENS,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


input_neurons = np.concatenate(
    [
        sensory_map.neurons_for("A"),
        sensory_map.neurons_for("B"),
        sensory_map.neurons_for("C"),
    ]
)

input_neurons = np.unique(
    input_neurons
)


# ============================================================
# Indirect readout
# ============================================================

readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()
print(
    f"Input neurons:   {len(input_neurons)}"
)

print(
    f"Readout neurons: {len(readout_population):,}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Run sequence
# ============================================================

def run_sequence(sequence, seed):

    brain.reset(
        seed=seed
    )

    trace.reset()


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    last_state = None

    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        last_state = np.asarray(
            trace.observe(fired)
        ).copy()


    # --------------------------------------------------------
    # Present complete sequence
    # --------------------------------------------------------

    for token in sequence:

        neurons = sensory_map.neurons_for(
            token
        )


        for _ in range(TOKEN_STEPS):

            fired = brain.step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ]
            )

            last_state = np.asarray(
                trace.observe(fired)
            ).copy()


        # Gap
        for _ in range(GAP_STEPS):

            fired = brain.step()

            last_state = np.asarray(
                trace.observe(fired)
            ).copy()


    # --------------------------------------------------------
    # State immediately after common final token C
    # --------------------------------------------------------

    final_sequence_state = (
        last_state.copy()
    )


    # --------------------------------------------------------
    # Let activity evolve with NO input
    # --------------------------------------------------------

    post_states = []

    for _ in range(POST_STEPS):

        fired = brain.step()

        state = np.asarray(
            trace.observe(fired)
        ).copy()

        post_states.append(
            state
        )


    delayed_state = (
        post_states[-1]
    )


    mean_post_state = np.stack(
        post_states
    ).mean(
        axis=0
    )


    # ========================================================
    # IMPORTANT
    #
    # We do NOT include position-1 or position-2 states.
    #
    # Both sequences:
    #
    # A B C
    # B A C
    #
    # finish with exactly the same token C.
    #
    # The decoder only sees the state AFTER C.
    # ========================================================

    feature = np.concatenate(
        [
            final_sequence_state,
            mean_post_state,
            delayed_state,
        ]
    )


    return feature.astype(
        np.float32
    )


# ============================================================
# Build dataset
# ============================================================

def build_dataset(seeds):

    X = []
    y = []


    for seed in seeds:

        # A -> B -> C
        feature_abc = run_sequence(
            ["A", "B", "C"],
            seed,
        )

        X.append(
            feature_abc
        )

        y.append(
            "ABC"
        )


        # B -> A -> C
        feature_bac = run_sequence(
            ["B", "A", "C"],
            seed,
        )

        X.append(
            feature_bac
        )

        y.append(
            "BAC"
        )


    return (
        np.stack(X),
        np.asarray(y),
    )


# ============================================================
# Generate data
# ============================================================

print()
print(
    "Generating training episodes..."
)

X_train, y_train = build_dataset(
    TRAIN_SEEDS
)


print(
    "Generating unseen-seed test episodes..."
)

X_test, y_test = build_dataset(
    TEST_SEEDS
)


print()
print("========================================")
print("              DATASET")
print("========================================")
print()

print(
    f"Train episodes: {len(X_train)}"
)

print(
    f"Test episodes:  {len(X_test)}"
)

print(
    f"Feature size:   {X_train.shape[1]:,}"
)

print(
    f"Finite train:   {np.isfinite(X_train).all()}"
)

print(
    f"Finite test:    {np.isfinite(X_test).all()}"
)


assert np.isfinite(
    X_train
).all()

assert np.isfinite(
    X_test
).all()


# ============================================================
# Classifier
# ============================================================

components = min(
    20,
    len(X_train) - 1,
    X_train.shape[1],
)


model = Pipeline(
    [
        (
            "scale",
            StandardScaler(),
        ),

        (
            "pca",
            PCA(
                n_components=components,
                random_state=42,
            ),
        ),

        (
            "ridge",
            RidgeClassifier(
                alpha=1.0,
            ),
        ),
    ]
)


print()
print(
    "Training strict order decoder..."
)


model.fit(
    X_train,
    y_train,
)


predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions,
)


# ============================================================
# Results
# ============================================================

print()
print("========================================")
print("      STRICT ORDER MEMORY RESULT")
print("========================================")
print()

print(
    "Chance accuracy: 50.00%"
)

print(
    f"Test accuracy:   "
    f"{accuracy * 100:.2f}%"
)


print()
print("Predictions:")
print()


for true, predicted in zip(
    y_test,
    predictions,
):

    mark = (
        "✓"
        if true == predicted
        else "✗"
    )

    print(
        f"{mark} "
        f"{true} -> {predicted}"
    )


print()
print("========================================")
print("           INTERPRETATION")
print("========================================")
print()


if accuracy >= 0.80:

    print(
        "✅ Strong genuine order-memory signal detected."
    )

elif accuracy > 0.60:

    print(
        "⚠️ Some earlier-order memory survives "
        "after the common final token."
    )

else:

    print(
        "❌ Earlier token order is not reliably "
        "recoverable after the common final token."
    )


print()
print(
    f"Trace tau:        {TRACE_TAU}"
)

print(
    f"Token steps:      {TOKEN_STEPS}"
)

print(
    f"Gap steps:        {GAP_STEPS}"
)

print(
    f"Post steps:       {POST_STEPS}"
)

print(
    f"Neurons/token:    {NEURONS_PER_TOKEN}"
)

print()
print(
    "✅ Strict order-memory probe completed"
)