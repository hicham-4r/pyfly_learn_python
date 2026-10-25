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
POST_STEPS = 4

MAPPING_SEED = 42

# We tune ONLY tau in this experiment.
TAU_VALUES = [
    0.05,
    0.10,
    0.25,
    0.50,
]

# Train / validation / final test seeds
TRAIN_SEEDS = list(range(0, 40))
VALIDATION_SEEDS = list(range(40, 60))
FINAL_TEST_SEEDS = list(range(60, 80))


# ============================================================
# Load brain
# ============================================================

print()
print("========================================")
print("     PyFly STRICT ORDER TAU SWEEP")
print("========================================")
print()

print("Loading FlyBrain on GPU...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")

print(f"Neurons:     {brain.n:,}")
print(f"Connections: {len(brain.weights):,}")


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
# Indirect visual readout
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

print(
    f"Train seeds:      "
    f"{TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}"
)

print(
    f"Validation seeds: "
    f"{VALIDATION_SEEDS[0]}-{VALIDATION_SEEDS[-1]}"
)

print(
    f"Final test seeds: "
    f"{FINAL_TEST_SEEDS[0]}-{FINAL_TEST_SEEDS[-1]}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


# ============================================================
# Run one sequence for a specific tau
# ============================================================

def run_sequence(
    sequence,
    seed,
    tau,
):

    trace = Trace(
        brain,
        idx=readout_population,
        tau=tau,
    )

    brain.reset(
        seed=seed
    )

    trace.reset()

    last_state = None


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        last_state = np.asarray(
            trace.observe(fired)
        ).copy()


    # --------------------------------------------------------
    # Present complete sequence
    # --------------------------------------------------------

    for token in sequence:

        token_neurons = (
            sensory_map.neurons_for(
                token
            )
        )


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

            last_state = np.asarray(
                trace.observe(fired)
            ).copy()


        # Inter-token gap
        for _ in range(
            GAP_STEPS
        ):

            fired = brain.step()

            last_state = np.asarray(
                trace.observe(fired)
            ).copy()


    # --------------------------------------------------------
    # State right after the common final token C
    # --------------------------------------------------------

    final_sequence_state = (
        last_state.copy()
    )


    # --------------------------------------------------------
    # Post-sequence memory window
    # --------------------------------------------------------

    post_states = []


    for _ in range(
        POST_STEPS
    ):

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


    # --------------------------------------------------------
    # Strict feature
    #
    # IMPORTANT:
    # No position-1 state
    # No position-2 state
    #
    # Decoder only gets states AFTER both sequences
    # have ended with the same token C.
    # --------------------------------------------------------

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

def build_dataset(
    seeds,
    tau,
):

    X = []
    y = []


    for seed in seeds:

        # A -> B -> C
        feature_abc = run_sequence(
            ["A", "B", "C"],
            seed,
            tau,
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
            tau,
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
# Create classifier
# ============================================================

def create_model(
    X_train,
):

    components = min(
        20,
        len(X_train) - 1,
        X_train.shape[1],
    )


    return Pipeline(
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
# Sweep tau
# ============================================================

validation_results = []


print()
print("========================================")
print("          VALIDATION SWEEP")
print("========================================")
print()


for tau in TAU_VALUES:

    print()
    print("----------------------------------------")
    print(f"Testing trace tau = {tau}")
    print("----------------------------------------")


    # --------------------------------------------------------
    # Generate train set for this tau
    # --------------------------------------------------------

    print(
        "Generating training episodes..."
    )

    X_train, y_train = build_dataset(
        TRAIN_SEEDS,
        tau,
    )


    # --------------------------------------------------------
    # Generate validation set
    # --------------------------------------------------------

    print(
        "Generating validation episodes..."
    )

    X_val, y_val = build_dataset(
        VALIDATION_SEEDS,
        tau,
    )


    assert np.isfinite(
        X_train
    ).all()

    assert np.isfinite(
        X_val
    ).all()


    print(
        f"Train shape: "
        f"{X_train.shape}"
    )

    print(
        f"Validation shape: "
        f"{X_val.shape}"
    )


    # --------------------------------------------------------
    # Train decoder
    # --------------------------------------------------------

    model = create_model(
        X_train
    )


    model.fit(
        X_train,
        y_train,
    )


    predictions = model.predict(
        X_val
    )


    accuracy = accuracy_score(
        y_val,
        predictions,
    )


    validation_results.append(
        (
            tau,
            accuracy,
        )
    )


    print()
    print(
        f"Tau {tau:0.2f} "
        f"validation accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# Validation summary
# ============================================================

print()
print()
print("========================================")
print("          TAU SWEEP SUMMARY")
print("========================================")
print()


for tau, accuracy in validation_results:

    print(
        f"tau={tau:0.2f} "
        f"validation="
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# Pick best tau using validation only
# ============================================================

best_tau, best_val_accuracy = max(
    validation_results,
    key=lambda item: item[1],
)


print()
print("========================================")
print("             BEST TAU")
print("========================================")
print()

print(
    f"Best tau: "
    f"{best_tau}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# Final training
#
# Combine train + validation seeds
# ============================================================

FINAL_TRAIN_SEEDS = (
    TRAIN_SEEDS
    + VALIDATION_SEEDS
)


print()
print("========================================")
print("          FINAL TRAINING")
print("========================================")
print()

print(
    f"Using tau = {best_tau}"
)

print(
    "Generating final train dataset..."
)


X_final_train, y_final_train = build_dataset(
    FINAL_TRAIN_SEEDS,
    best_tau,
)


print(
    "Generating untouched final test dataset..."
)


X_final_test, y_final_test = build_dataset(
    FINAL_TEST_SEEDS,
    best_tau,
)


assert np.isfinite(
    X_final_train
).all()

assert np.isfinite(
    X_final_test
).all()


print()
print(
    f"Final train episodes: "
    f"{len(X_final_train)}"
)

print(
    f"Final test episodes:  "
    f"{len(X_final_test)}"
)

print(
    f"Feature size:         "
    f"{X_final_train.shape[1]:,}"
)


# ============================================================
# Final model
# ============================================================

final_model = create_model(
    X_final_train
)


final_model.fit(
    X_final_train,
    y_final_train,
)


final_predictions = final_model.predict(
    X_final_test
)


final_accuracy = accuracy_score(
    y_final_test,
    final_predictions,
)


# ============================================================
# Final result
# ============================================================

print()
print()
print("========================================")
print("       STRICT ORDER FINAL RESULT")
print("========================================")
print()

print(
    "Chance accuracy:          "
    "50.00%"
)

print(
    f"Selected tau:             "
    f"{best_tau}"
)

print(
    f"Validation accuracy:      "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"UNTOUCHED TEST accuracy:  "
    f"{final_accuracy * 100:.2f}%"
)


print()
print("Final test predictions:")
print()


for true, predicted in zip(
    y_final_test,
    final_predictions,
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


if final_accuracy >= 0.80:

    print(
        "✅ Strong temporal-order memory "
        "generalizes to unseen neural seeds."
    )

elif final_accuracy > 0.60:

    print(
        "⚠️ Some temporal-order memory "
        "generalizes to unseen neural seeds."
    )

else:

    print(
        "❌ Temporal-order memory remains "
        "weak under this configuration."
    )


print()
print("Configuration:")

print(
    f"  tau              = "
    f"{best_tau}"
)

print(
    f"  token steps      = "
    f"{TOKEN_STEPS}"
)

print(
    f"  gap steps        = "
    f"{GAP_STEPS}"
)

print(
    f"  post steps       = "
    f"{POST_STEPS}"
)

print(
    f"  neurons/token    = "
    f"{NEURONS_PER_TOKEN}"
)

print(
    f"  amplitude        = "
    f"{AMPLITUDE}"
)

print()
print(
    "✅ Tau sweep completed"
)