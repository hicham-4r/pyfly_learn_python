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

# Slightly longer trace memory
TRACE_TAU = 0.25

MAPPING_SEED = 42

# Different neural-noise seeds for train/test
TRAIN_SEEDS = list(range(20))
TEST_SEEDS = list(range(20, 30))


# ============================================================
# Load Brain
# ============================================================

print()
print("========================================")
print("       PyFly ORDER MEMORY PROBE")
print("========================================")
print()

print("Loading FlyBrain...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")


# ============================================================
# Input population
# ============================================================

visual_pool = np.asarray(
    brain.cells(["visual_projection"])
)

TOKENS = [
    "A",
    "B",
]


sensory_map = create_disjoint_sensory_map(
    tokens=TOKENS,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


# ============================================================
# Find directly stimulated neurons
# ============================================================

input_neurons = np.concatenate(
    [
        sensory_map.neurons_for("A"),
        sensory_map.neurons_for("B"),
    ]
)

input_neurons = np.unique(
    input_neurons
)


# ============================================================
# Indirect visual readout
#
# Important:
# exclude every directly stimulated neuron
# ============================================================

readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()
print(
    f"Input neurons:   "
    f"{len(input_neurons)}"
)

print(
    f"Readout neurons: "
    f"{len(readout_population):,}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


# ============================================================
# Trace
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Run one sequence
#
# Sequence examples:
#
# ["A", "B"]
#
# ["B", "A"]
# ============================================================

def run_sequence(sequence, seed):

    # Reset network/noise
    brain.reset(
        seed=seed
    )

    # Reset trace state
    trace.reset()


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    last_state = None

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        last_state = np.asarray(
            trace.observe(fired)
        ).copy()


    # --------------------------------------------------------
    # Record state for each token position
    # --------------------------------------------------------

    position_states = []


    for token in sequence:

        token_neurons = (
            sensory_map.neurons_for(
                token
            )
        )

        current_token_states = []


        # ----------------------------------------------------
        # Stimulate token
        # ----------------------------------------------------

        for _ in range(
            TOKEN_STEPS
        ):

            fired = brain.step(
                inject=[
                    (
                        token_neurons,
                        AMPLITUDE,
                    )
                ]
            )


            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


            current_token_states.append(
                state
            )

            last_state = state


        # ----------------------------------------------------
        # Mean neural state during this token
        # ----------------------------------------------------

        token_mean = np.stack(
            current_token_states
        ).mean(
            axis=0
        )


        position_states.append(
            token_mean
        )


        # ----------------------------------------------------
        # Small inter-token gap
        # ----------------------------------------------------

        for _ in range(
            GAP_STEPS
        ):

            fired = brain.step()

            last_state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


    # --------------------------------------------------------
    # Final reservoir state
    #
    # IMPORTANT:
    # Trace does NOT expose trace.value.
    # We explicitly keep the last output returned by
    # trace.observe().
    # --------------------------------------------------------

    final_state = last_state.copy()


    # --------------------------------------------------------
    # Feature:
    #
    # mean state during first token
    # +
    # mean state during second token
    # +
    # final state after sequence
    # --------------------------------------------------------

    feature = np.concatenate(
        [
            position_states[0],
            position_states[1],
            final_state,
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

        # ----------------------------------------------------
        # A -> B
        # ----------------------------------------------------

        feature_ab = run_sequence(
            ["A", "B"],
            seed,
        )


        X.append(
            feature_ab
        )

        y.append(
            "AB"
        )


        # ----------------------------------------------------
        # B -> A
        # ----------------------------------------------------

        feature_ba = run_sequence(
            ["B", "A"],
            seed,
        )


        X.append(
            feature_ba
        )

        y.append(
            "BA"
        )


    return (
        np.stack(X),
        np.asarray(y),
    )


# ============================================================
# Generate datasets
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
print("             DATASET")
print("========================================")
print()

print(
    f"Train episodes: "
    f"{len(X_train)}"
)

print(
    f"Test episodes:  "
    f"{len(X_test)}"
)

print(
    f"Feature size:   "
    f"{X_train.shape[1]:,}"
)

print(
    f"Finite train:   "
    f"{np.isfinite(X_train).all()}"
)

print(
    f"Finite test:    "
    f"{np.isfinite(X_test).all()}"
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


print()
print(
    f"PCA components: "
    f"{components}"
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


# ============================================================
# Train
# ============================================================

print()
print(
    "Training order decoder..."
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# Test
# ============================================================

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
print("          ORDER MEMORY RESULT")
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


# ============================================================
# Interpretation
# ============================================================

print()
print("========================================")
print("            INTERPRETATION")
print("========================================")
print()


if accuracy >= 0.80:

    print(
        "✅ Strong temporal-order memory detected."
    )

elif accuracy > 0.60:

    print(
        "⚠️ Some temporal-order information "
        "was detected."
    )

else:

    print(
        "❌ Current reservoir configuration "
        "does not reliably preserve order."
    )


print()
print(
    f"Trace tau:        "
    f"{TRACE_TAU}"
)

print(
    f"Token steps:      "
    f"{TOKEN_STEPS}"
)

print(
    f"Gap steps:        "
    f"{GAP_STEPS}"
)

print(
    f"Neurons/token:    "
    f"{NEURONS_PER_TOKEN}"
)

print(
    f"Stimulus amplitude: "
    f"{AMPLITUDE}"
)

print()
print(
    "✅ Order-memory probe completed"
)