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
POST_STEPS = 4

TRACE_TAU = 0.25
MAPPING_SEED = 42


# ============================================================
# Timing sweep
# ============================================================

TOKEN_STEP_VALUES = [
    2,
    4,
    8,
    16,
]

GAP_STEP_VALUES = [
    0,
    1,
    2,
    4,
]


# ============================================================
# IMPORTANT:
#
# Seeds 60-79 were already inspected in the previous experiment,
# so they are now part of validation, NOT final test.
#
# New untouched final test:
# 80-99
# ============================================================

TRAIN_SEEDS = list(
    range(0, 40)
)

VALIDATION_SEEDS = list(
    range(40, 80)
)

FINAL_TEST_SEEDS = list(
    range(80, 100)
)


# ============================================================
# Load brain
# ============================================================

print()
print("============================================")
print("   PyFly STRICT ORDER TIMING SWEEP")
print("============================================")
print()

print("Loading FlyBrain on GPU...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")

print(
    f"Neurons:     {brain.n:,}"
)

print(
    f"Connections: {len(brain.weights):,}"
)


# ============================================================
# Input mapping
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
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
    f"Trace tau:       "
    f"{TRACE_TAU}"
)

print(
    f"Amplitude:       "
    f"{AMPLITUDE}"
)

print(
    f"Neurons/token:   "
    f"{NEURONS_PER_TOKEN}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


# ============================================================
# Run one sequence
# ============================================================

def run_sequence(
    sequence,
    seed,
    token_steps,
    gap_steps,
):

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )


    brain.reset(
        seed=seed
    )

    trace.reset()


    last_state = None


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        last_state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()


    # --------------------------------------------------------
    # Present sequence
    #
    # ABC
    #
    # or
    #
    # BAC
    # --------------------------------------------------------

    for token in sequence:

        token_neurons = (
            sensory_map.neurons_for(
                token
            )
        )


        # Token stimulation
        for _ in range(
            token_steps
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
                trace.observe(
                    fired
                )
            ).copy()


        # Inter-token gap
        for _ in range(
            gap_steps
        ):

            fired = brain.step()

            last_state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


    # --------------------------------------------------------
    # State after common final token C
    # --------------------------------------------------------

    final_sequence_state = (
        last_state.copy()
    )


    # --------------------------------------------------------
    # Post sequence propagation
    # --------------------------------------------------------

    post_states = []


    for _ in range(
        POST_STEPS
    ):

        fired = brain.step()

        state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()


        post_states.append(
            state
        )


    mean_post_state = np.stack(
        post_states
    ).mean(
        axis=0
    )


    delayed_state = (
        post_states[-1]
    )


    # --------------------------------------------------------
    # STRICT FEATURE
    #
    # We only expose states AFTER the full sequence.
    #
    # No explicit position 1 state.
    # No explicit position 2 state.
    #
    # Both sequences end with C.
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
    token_steps,
    gap_steps,
):

    X = []
    y = []


    for seed in seeds:

        # ----------------------------------------------------
        # ABC
        # ----------------------------------------------------

        feature_abc = run_sequence(
            ["A", "B", "C"],
            seed,
            token_steps,
            gap_steps,
        )


        X.append(
            feature_abc
        )

        y.append(
            "ABC"
        )


        # ----------------------------------------------------
        # BAC
        # ----------------------------------------------------

        feature_bac = run_sequence(
            ["B", "A", "C"],
            seed,
            token_steps,
            gap_steps,
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
# Validation timing sweep
# ============================================================

results = []


print()
print("============================================")
print("          TIMING VALIDATION SWEEP")
print("============================================")
print()


total_configurations = (
    len(TOKEN_STEP_VALUES)
    *
    len(GAP_STEP_VALUES)
)

configuration_number = 0


for token_steps in TOKEN_STEP_VALUES:

    for gap_steps in GAP_STEP_VALUES:

        configuration_number += 1


        print()
        print(
            "--------------------------------------------"
        )

        print(
            f"Configuration "
            f"{configuration_number}/"
            f"{total_configurations}"
        )

        print(
            f"TOKEN_STEPS = {token_steps}"
        )

        print(
            f"GAP_STEPS   = {gap_steps}"
        )

        print(
            "--------------------------------------------"
        )


        # ----------------------------------------------------
        # Training features
        # ----------------------------------------------------

        print(
            "Generating training episodes..."
        )


        X_train, y_train = build_dataset(
            TRAIN_SEEDS,
            token_steps,
            gap_steps,
        )


        # ----------------------------------------------------
        # Validation features
        # ----------------------------------------------------

        print(
            "Generating validation episodes..."
        )


        X_val, y_val = build_dataset(
            VALIDATION_SEEDS,
            token_steps,
            gap_steps,
        )


        assert np.isfinite(
            X_train
        ).all()


        assert np.isfinite(
            X_val
        ).all()


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

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


        results.append(
            {
                "token_steps": token_steps,
                "gap_steps": gap_steps,
                "accuracy": accuracy,
            }
        )


        print()
        print(
            f"Validation accuracy: "
            f"{accuracy * 100:.2f}%"
        )


# ============================================================
# Sweep summary
# ============================================================

print()
print()
print("============================================")
print("          TIMING SWEEP SUMMARY")
print("============================================")
print()


sorted_results = sorted(
    results,
    key=lambda item: (
        item["accuracy"]
    ),
    reverse=True,
)


for result in sorted_results:

    print(
        f"token_steps="
        f"{result['token_steps']:2} "
        f"gap_steps="
        f"{result['gap_steps']:2} "
        f"validation="
        f"{result['accuracy'] * 100:6.2f}%"
    )


# ============================================================
# Select best configuration
# ============================================================

best = sorted_results[0]


BEST_TOKEN_STEPS = (
    best["token_steps"]
)

BEST_GAP_STEPS = (
    best["gap_steps"]
)

BEST_VALIDATION = (
    best["accuracy"]
)


print()
print("============================================")
print("       BEST TIMING CONFIGURATION")
print("============================================")
print()


print(
    f"Token steps: "
    f"{BEST_TOKEN_STEPS}"
)

print(
    f"Gap steps:   "
    f"{BEST_GAP_STEPS}"
)

print(
    f"Validation:  "
    f"{BEST_VALIDATION * 100:.2f}%"
)


# ============================================================
# Final train
#
# We can now combine:
#
# seeds 0-39
# +
# seeds 40-79
#
# because timing selection is complete.
# ============================================================

FINAL_TRAIN_SEEDS = (
    TRAIN_SEEDS
    +
    VALIDATION_SEEDS
)


print()
print("============================================")
print("             FINAL TRAINING")
print("============================================")
print()


print(
    f"Best token steps: "
    f"{BEST_TOKEN_STEPS}"
)

print(
    f"Best gap steps:   "
    f"{BEST_GAP_STEPS}"
)


print()
print(
    "Generating final training episodes..."
)


X_final_train, y_final_train = build_dataset(
    FINAL_TRAIN_SEEDS,
    BEST_TOKEN_STEPS,
    BEST_GAP_STEPS,
)


print(
    "Generating UNTOUCHED final test episodes..."
)


X_final_test, y_final_test = build_dataset(
    FINAL_TEST_SEEDS,
    BEST_TOKEN_STEPS,
    BEST_GAP_STEPS,
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


final_predictions = (
    final_model.predict(
        X_final_test
    )
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
print("============================================")
print("        STRICT TIMING FINAL RESULT")
print("============================================")
print()


print(
    "Chance accuracy:          "
    "50.00%"
)

print(
    f"Selected token steps:     "
    f"{BEST_TOKEN_STEPS}"
)

print(
    f"Selected gap steps:       "
    f"{BEST_GAP_STEPS}"
)

print(
    f"Validation accuracy:      "
    f"{BEST_VALIDATION * 100:.2f}%"
)

print(
    f"UNTOUCHED TEST accuracy:  "
    f"{final_accuracy * 100:.2f}%"
)


# ============================================================
# Predictions
# ============================================================

print()
print(
    "Final test predictions:"
)

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
print("============================================")
print("             INTERPRETATION")
print("============================================")
print()


if final_accuracy >= 0.80:

    print(
        "✅ Strong order-memory performance "
        "generalizes to unseen seeds."
    )

elif final_accuracy >= 0.70:

    print(
        "✅ Useful temporal-order memory detected."
    )

elif final_accuracy > 0.60:

    print(
        "⚠️ Some temporal-order memory survives, "
        "but it is still noisy."
    )

else:

    print(
        "❌ Temporal-order memory remains weak."
    )


print()
print(
    "Final configuration:"
)

print(
    f"  tau              = "
    f"{TRACE_TAU}"
)

print(
    f"  token steps      = "
    f"{BEST_TOKEN_STEPS}"
)

print(
    f"  gap steps        = "
    f"{BEST_GAP_STEPS}"
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
    "✅ Timing sweep completed"
)