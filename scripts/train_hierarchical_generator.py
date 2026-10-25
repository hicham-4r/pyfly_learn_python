from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Files
# ============================================================

TRAIN_FILE = (
    "artifacts/features/"
    "generator_train.npz"
)

VALIDATION_FILE = (
    "artifacts/features/"
    "generator_validation.npz"
)

TEST_FILE = (
    "artifacts/features/"
    "generator_test.npz"
)


# Existing best global decoder from Step 68.
GLOBAL_DECODER_FILE = (
    "artifacts/models/"
    "generator_best_decoder.joblib"
)


OUTPUT_MODEL = Path(
    "artifacts/models/"
    "generator_hierarchical.joblib"
)

OUTPUT_METRICS = Path(
    "artifacts/metrics/"
    "generator_hierarchical_metrics.json"
)


# ============================================================
# Search space
# ============================================================

PCA_COMPONENTS = [
    5,
    10,
    20,
    40,
    60,
]

C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# ============================================================
# Load data
# ============================================================

def load_split(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    return {
        "X": data["X"].astype(
            np.float32
        ),

        "targets": data[
            "targets"
        ].astype(str),

        "sample_ids": data[
            "sample_ids"
        ].astype(str),

        "example_ids": data[
            "example_ids"
        ].astype(str),

        "intents": data[
            "intents"
        ].astype(str),

        "positions": data[
            "target_positions"
        ].astype(
            np.int32
        ),

        "feedback": data[
            "feedback_tokens"
        ].astype(str),

        "seeds": data[
            "seeds"
        ].astype(
            np.int32
        ),
    }


# ============================================================
# Convert detailed intent -> broad program family
# ============================================================

def intent_to_family(
    intent: str,
) -> str:

    if intent.startswith(
        "copy_"
    ):
        return "COPY"

    if intent.startswith(
        "sub_"
    ):
        return "SUBTRACT"

    if intent.startswith(
        "div_"
    ):
        return "DIVIDE"

    raise ValueError(
        f"Unknown intent: {intent}"
    )


def families_from_intents(
    intents,
):

    return np.asarray([
        intent_to_family(
            str(intent)
        )
        for intent in intents
    ])


# ============================================================
# Model factory
# ============================================================

def make_model(
    components,
    c_value,
):

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
                    svd_solver="randomized",
                    random_state=42,
                ),
            ),

            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# Search a balanced logistic model
# ============================================================

def tune_model(
    X_train,
    y_train,
    X_val,
    y_val,
    label,
):

    best = None

    max_components = min(
        X_train.shape[0],
        X_train.shape[1],
    )

    print()
    print(
        f"Searching: {label}"
    )


    for components in PCA_COMPONENTS:

        if components >= max_components:
            continue

        for c_value in C_VALUES:

            model = make_model(
                components,
                c_value,
            )

            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(
                X_val
            )

            accuracy = accuracy_score(
                y_val,
                prediction,
            )

            macro_f1 = f1_score(
                y_val,
                prediction,
                average="macro",
                zero_division=0,
            )

            print(
                f"  PCA={components:2} "
                f"C={c_value:7.3f} "
                f"acc={accuracy * 100:6.2f}% "
                f"F1={macro_f1 * 100:6.2f}%"
            )

            if (
                best is None
                or
                macro_f1
                >
                best["macro_f1"]
                or (
                    macro_f1
                    ==
                    best["macro_f1"]
                    and
                    accuracy
                    >
                    best["accuracy"]
                )
            ):

                best = {
                    "components": (
                        components
                    ),

                    "c_value": (
                        c_value
                    ),

                    "accuracy": (
                        accuracy
                    ),

                    "macro_f1": (
                        macro_f1
                    ),
                }


    if best is None:

        raise RuntimeError(
            f"No valid model for {label}"
        )


    print(
        f"  BEST: PCA="
        f"{best['components']} "
        f"C={best['c_value']} "
        f"accuracy="
        f"{best['accuracy'] * 100:.2f}% "
        f"F1="
        f"{best['macro_f1'] * 100:.2f}%"
    )


    X_final = np.concatenate(
        [
            X_train,
            X_val,
        ],
        axis=0,
    )

    y_final = np.concatenate(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )


    final_model = make_model(
        best["components"],
        best["c_value"],
    )

    final_model.fit(
        X_final,
        y_final,
    )


    return (
        final_model,
        best,
    )


# ============================================================
# Header
# ============================================================

print()
print(
    "===================================================="
)

print(
    "       PyFly HIERARCHICAL GENERATOR TRAINING"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load splits
# ============================================================

train = load_split(
    TRAIN_FILE
)

validation = load_split(
    VALIDATION_FILE
)

test = load_split(
    TEST_FILE
)


print(
    f"Train:      "
    f"{train['X'].shape}"
)

print(
    f"Validation: "
    f"{validation['X'].shape}"
)

print(
    f"Test:       "
    f"{test['X'].shape}"
)


# ============================================================
# Leakage
# ============================================================

train_ids = set(
    train[
        "example_ids"
    ].tolist()
)

validation_ids = set(
    validation[
        "example_ids"
    ].tolist()
)

test_ids = set(
    test[
        "example_ids"
    ].tolist()
)


assert train_ids.isdisjoint(
    validation_ids
)

assert train_ids.isdisjoint(
    test_ids
)

assert validation_ids.isdisjoint(
    test_ids
)


print()
print(
    "Logical-prompt leakage: PASS"
)


# ============================================================
# Determine block size
#
# Step 62 created:
#
#   prompt_mean
#   prompt_max
#   prompt_final
#   feedback_mean
#   feedback_max
#   feedback_final
#   delayed_state
#
# = 7 equal neural blocks
# ============================================================

feature_size = train[
    "X"
].shape[1]


assert (
    feature_size % 7
    == 0
)


BLOCK_SIZE = (
    feature_size
    //
    7
)


PROMPT_FEATURE_SIZE = (
    3
    *
    BLOCK_SIZE
)


print()
print(
    f"Feature size:        "
    f"{feature_size:,}"
)

print(
    f"Neural block size:   "
    f"{BLOCK_SIZE:,}"
)

print(
    f"Prompt-only feature: "
    f"{PROMPT_FEATURE_SIZE:,}"
)


# ============================================================
# Family labels
# ============================================================

train_families = (
    families_from_intents(
        train["intents"]
    )
)

validation_families = (
    families_from_intents(
        validation["intents"]
    )
)

test_families = (
    families_from_intents(
        test["intents"]
    )
)


# ============================================================
# FAMILY DECODER
#
# Use only position-0 rows.
#
# This gives one prompt feature per logical episode rather
# than duplicating the same prompt for every output token.
# ============================================================

train_family_mask = (
    train[
        "positions"
    ]
    ==
    0
)

validation_family_mask = (
    validation[
        "positions"
    ]
    ==
    0
)

test_family_mask = (
    test[
        "positions"
    ]
    ==
    0
)


X_family_train = (
    train[
        "X"
    ][
        train_family_mask,
        :PROMPT_FEATURE_SIZE,
    ]
)

y_family_train = (
    train_families[
        train_family_mask
    ]
)


X_family_val = (
    validation[
        "X"
    ][
        validation_family_mask,
        :PROMPT_FEATURE_SIZE,
    ]
)

y_family_val = (
    validation_families[
        validation_family_mask
    ]
)


X_family_test = (
    test[
        "X"
    ][
        test_family_mask,
        :PROMPT_FEATURE_SIZE,
    ]
)

y_family_test = (
    test_families[
        test_family_mask
    ]
)


print()
print(
    "===================================================="
)

print(
    "              PROGRAM FAMILY DECODER"
)

print(
    "===================================================="
)

print()


print(
    f"Family train: "
    f"{X_family_train.shape}"
)

print(
    f"Family val:   "
    f"{X_family_val.shape}"
)

print(
    f"Family test:  "
    f"{X_family_test.shape}"
)


family_model, family_config = (
    tune_model(
        X_family_train,
        y_family_train,

        X_family_val,
        y_family_val,

        "PROGRAM FAMILY",
    )
)


family_test_prediction = (
    family_model.predict(
        X_family_test
    )
)


family_test_accuracy = (
    accuracy_score(
        y_family_test,
        family_test_prediction,
    )
)


family_test_macro_f1 = (
    f1_score(
        y_family_test,
        family_test_prediction,
        average="macro",
        zero_division=0,
    )
)


print()
print(
    "Family test result:"
)

print(
    f"Accuracy: "
    f"{family_test_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{family_test_macro_f1 * 100:.2f}%"
)


print()
print(
    classification_report(
        y_family_test,
        family_test_prediction,
        zero_division=0,
    )
)


print(
    "Family confusion matrix:"
)

print(
    confusion_matrix(
        y_family_test,
        family_test_prediction,
        labels=[
            "COPY",
            "SUBTRACT",
            "DIVIDE",
        ],
    )
)


# ============================================================
# Family prediction for EVERY next-token row
#
# Only the first 3 neural blocks are used.
# ============================================================

train_family_predictions = (
    family_model.predict(
        train[
            "X"
        ][
            :,
            :PROMPT_FEATURE_SIZE,
        ]
    )
)

validation_family_predictions = (
    family_model.predict(
        validation[
            "X"
        ][
            :,
            :PROMPT_FEATURE_SIZE,
        ]
    )
)

test_family_predictions = (
    family_model.predict(
        test[
            "X"
        ][
            :,
            :PROMPT_FEATURE_SIZE,
        ]
    )
)


# ============================================================
# Train family-specific, position-specific token heads
# ============================================================

FAMILIES = [
    "COPY",
    "SUBTRACT",
    "DIVIDE",
]


heads = {}

head_metrics = {}


print()
print(
    "===================================================="
)

print(
    "          FAMILY-SPECIFIC TOKEN HEADS"
)

print(
    "===================================================="
)


for family in FAMILIES:

    heads[
        family
    ] = {}

    head_metrics[
        family
    ] = {}


    family_positions = sorted(
        np.unique(
            train[
                "positions"
            ][
                train_families
                ==
                family
            ]
        )
    )


    for position in family_positions:

        train_mask = (
            (
                train_families
                ==
                family
            )
            &
            (
                train[
                    "positions"
                ]
                ==
                position
            )
        )


        val_mask = (
            (
                validation_families
                ==
                family
            )
            &
            (
                validation[
                    "positions"
                ]
                ==
                position
            )
        )


        test_mask = (
            (
                test_families
                ==
                family
            )
            &
            (
                test[
                    "positions"
                ]
                ==
                position
            )
        )


        X_head_train = (
            train[
                "X"
            ][
                train_mask
            ]
        )

        y_head_train = (
            train[
                "targets"
            ][
                train_mask
            ]
        )


        X_head_val = (
            validation[
                "X"
            ][
                val_mask
            ]
        )

        y_head_val = (
            validation[
                "targets"
            ][
                val_mask
            ]
        )


        X_head_test = (
            test[
                "X"
            ][
                test_mask
            ]
        )

        y_head_test = (
            test[
                "targets"
            ][
                test_mask
            ]
        )


        classes = sorted(
            np.unique(
                y_head_train
            )
        )


        print()
        print(
            "----------------------------------------------------"
        )

        print(
            f"{family} / POSITION {position}"
        )

        print(
            f"Classes: {classes}"
        )


        # ====================================================
        # Constant head
        # ====================================================

        if len(
            classes
        ) == 1:

            token = classes[
                0
            ]


            prediction = np.full(
                len(
                    y_head_test
                ),

                token,
            )


            test_accuracy = (
                accuracy_score(
                    y_head_test,
                    prediction,
                )
            )


            print(
                f"Constant head: "
                f"{token}"
            )

            print(
                f"Test accuracy: "
                f"{test_accuracy * 100:.2f}%"
            )


            heads[
                family
            ][
                int(
                    position
                )
            ] = {
                "type": (
                    "constant"
                ),

                "token": (
                    token
                ),
            }


            head_metrics[
                family
            ][
                str(
                    position
                )
            ] = {
                "type": (
                    "constant"
                ),

                "token": (
                    token
                ),

                "test_accuracy": float(
                    test_accuracy
                ),
            }


            continue


        # ====================================================
        # Learned head
        # ====================================================

        model, config = tune_model(
            X_head_train,
            y_head_train,

            X_head_val,
            y_head_val,

            (
                f"{family} "
                f"POSITION {position}"
            ),
        )


        prediction = model.predict(
            X_head_test
        )


        test_accuracy = (
            accuracy_score(
                y_head_test,
                prediction,
            )
        )


        test_macro_f1 = (
            f1_score(
                y_head_test,
                prediction,
                average="macro",
                zero_division=0,
            )
        )


        print(
            f"TEST accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        print(
            f"TEST macro F1: "
            f"{test_macro_f1 * 100:.2f}%"
        )


        heads[
            family
        ][
            int(
                position
            )
        ] = {
            "type": (
                "model"
            ),

            "model": (
                model
            ),

            "classes": (
                classes
            ),
        }


        head_metrics[
            family
        ][
            str(
                position
            )
        ] = {
            "type": (
                "model"
            ),

            "classes": (
                classes
            ),

            "pca_components": int(
                config[
                    "components"
                ]
            ),

            "c_value": float(
                config[
                    "c_value"
                ]
            ),

            "validation_accuracy": float(
                config[
                    "accuracy"
                ]
            ),

            "validation_macro_f1": float(
                config[
                    "macro_f1"
                ]
            ),

            "test_accuracy": float(
                test_accuracy
            ),

            "test_macro_f1": float(
                test_macro_f1
            ),
        }


# ============================================================
# Existing global decoder = fallback
# ============================================================

global_decoder = joblib.load(
    GLOBAL_DECODER_FILE
)


# ============================================================
# Hierarchical prediction helper
# ============================================================

def predict_from_head(
    feature,
    family,
    position,
):

    family_heads = heads.get(
        str(
            family
        )
    )


    if (
        family_heads is None
        or
        int(
            position
        )
        not in family_heads
    ):

        # Family/position combination unsupported.
        # Fall back to current best global decoder.
        return str(
            global_decoder.predict(
                feature.reshape(
                    1,
                    -1,
                )
            )[
                0
            ]
        ), True


    head = family_heads[
        int(
            position
        )
    ]


    if head[
        "type"
    ] == "constant":

        return str(
            head[
                "token"
            ]
        ), False


    prediction = (
        head[
            "model"
        ].predict(
            feature.reshape(
                1,
                -1,
            )
        )[
            0
        ]
    )


    return str(
        prediction
    ), False


# ============================================================
# Evaluate:
#
# 1. Oracle hierarchy
#    Uses TRUE family.
#
# 2. Predicted hierarchy
#    Uses family decoder.
# ============================================================

oracle_predictions = []

predicted_family_predictions = []

fallback_count = 0


for index in range(
    len(
        test[
            "targets"
        ]
    )
):

    feature = test[
        "X"
    ][
        index
    ]

    position = int(
        test[
            "positions"
        ][
            index
        ]
    )


    # --------------------------------------------------------
    # Oracle family
    # --------------------------------------------------------

    oracle_family = (
        test_families[
            index
        ]
    )


    oracle_token, _ = (
        predict_from_head(
            feature,
            oracle_family,
            position,
        )
    )


    oracle_predictions.append(
        oracle_token
    )


    # --------------------------------------------------------
    # Predicted family
    # --------------------------------------------------------

    predicted_family = (
        test_family_predictions[
            index
        ]
    )


    predicted_token, used_fallback = (
        predict_from_head(
            feature,
            predicted_family,
            position,
        )
    )


    if used_fallback:

        fallback_count += 1


    predicted_family_predictions.append(
        predicted_token
    )


oracle_predictions = np.asarray(
    oracle_predictions
)


predicted_family_predictions = (
    np.asarray(
        predicted_family_predictions
    )
)


y_test = test[
    "targets"
]


# ============================================================
# Metrics helper
# ============================================================

def evaluate_predictions(
    name,
    predictions,
):

    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )


    semantic_mask = np.isin(
        test[
            "positions"
        ],
        [
            0,
            2,
            3,
            4,
        ],
    )


    semantic_accuracy = (
        accuracy_score(
            y_test[
                semantic_mask
            ],
            predictions[
                semantic_mask
            ],
        )
    )


    print()
    print(
        name
    )

    print(
        f"  Top-1:              "
        f"{accuracy * 100:6.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{macro_f1 * 100:6.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{semantic_accuracy * 100:6.2f}%"
    )


    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "semantic_accuracy": float(
            semantic_accuracy
        ),
    }


# ============================================================
# Existing global decoder predictions
# ============================================================

global_predictions = (
    global_decoder.predict(
        test[
            "X"
        ]
    )
)


# ============================================================
# Final comparison
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "       HIERARCHICAL DECODER COMPARISON"
)

print(
    "===================================================="
)


global_result = (
    evaluate_predictions(
        "GLOBAL Balanced Logistic:",
        global_predictions,
    )
)


oracle_result = (
    evaluate_predictions(
        "HIERARCHICAL with TRUE family:",
        oracle_predictions,
    )
)


predicted_result = (
    evaluate_predictions(
        "HIERARCHICAL with PREDICTED family:",
        predicted_family_predictions,
    )
)


print()
print(
    f"Family classifier accuracy: "
    f"{family_test_accuracy * 100:.2f}%"
)

print(
    f"Family classifier macro F1: "
    f"{family_test_macro_f1 * 100:.2f}%"
)

print(
    f"Fallback predictions: "
    f"{fallback_count}"
)


print()
print(
    "Changes versus global decoder:"
)

print(
    f"Oracle hierarchy top-1: "
    f"{(oracle_result['accuracy'] - global_result['accuracy']) * 100:+.2f} pp"
)

print(
    f"Predicted hierarchy top-1: "
    f"{(predicted_result['accuracy'] - global_result['accuracy']) * 100:+.2f} pp"
)

print(
    f"Predicted hierarchy macro-F1: "
    f"{(predicted_result['macro_f1'] - global_result['macro_f1']) * 100:+.2f} pp"
)


# ============================================================
# Accuracy by position
# ============================================================

print()
print(
    "===================================================="
)

print(
    "        PREDICTED HIERARCHY BY POSITION"
)

print(
    "===================================================="
)

print()


position_metrics = {}


for position in sorted(
    np.unique(
        test[
            "positions"
        ]
    )
):

    mask = (
        test[
            "positions"
        ]
        ==
        position
    )


    accuracy = accuracy_score(
        y_test[
            mask
        ],

        predicted_family_predictions[
            mask
        ],
    )


    position_metrics[
        int(
            position
        )
    ] = float(
        accuracy
    )


    print(
        f"Position {position}: "
        f"{accuracy * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Especially inspect position 0
# ============================================================

position_zero_mask = (
    test[
        "positions"
    ]
    ==
    0
)


print()
print(
    "===================================================="
)

print(
    "              POSITION-0 DETAIL"
)

print(
    "===================================================="
)

print()


for (
    example_id,
    seed,
    true_family,
    predicted_family,
    target,
    prediction,
) in zip(
    test[
        "example_ids"
    ][
        position_zero_mask
    ],

    test[
        "seeds"
    ][
        position_zero_mask
    ],

    test_families[
        position_zero_mask
    ],

    test_family_predictions[
        position_zero_mask
    ],

    y_test[
        position_zero_mask
    ],

    predicted_family_predictions[
        position_zero_mask
    ],
):

    mark = (
        "✓"
        if target
        ==
        prediction
        else "✗"
    )


    print(
        f"{mark} "
        f"{example_id:16} "
        f"seed={seed} "
        f"family="
        f"{true_family:8} "
        f"family_pred="
        f"{predicted_family:8} "
        f"target="
        f"{target:8} "
        f"pred="
        f"{prediction}"
    )


# ============================================================
# Save model bundle
# ============================================================

bundle = {
    "prompt_feature_size": (
        PROMPT_FEATURE_SIZE
    ),

    "block_size": (
        BLOCK_SIZE
    ),

    "family_model": (
        family_model
    ),

    "heads": (
        heads
    ),

    "global_fallback": (
        global_decoder
    ),

    "families": (
        FAMILIES
    ),
}


OUTPUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    bundle,
    OUTPUT_MODEL,
)


# ============================================================
# Save metrics
# ============================================================

OUTPUT_METRICS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = {
    "family_classifier": {
        "test_accuracy": float(
            family_test_accuracy
        ),

        "test_macro_f1": float(
            family_test_macro_f1
        ),

        "pca_components": int(
            family_config[
                "components"
            ]
        ),

        "c_value": float(
            family_config[
                "c_value"
            ]
        ),
    },

    "global_decoder": (
        global_result
    ),

    "oracle_hierarchy": (
        oracle_result
    ),

    "predicted_hierarchy": (
        predicted_result
    ),

    "predicted_hierarchy_by_position": (
        position_metrics
    ),

    "fallback_count": int(
        fallback_count
    ),

    "heads": (
        head_metrics
    ),
}


with OUTPUT_METRICS.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


print()
print(
    f"Saved hierarchical model: "
    f"{OUTPUT_MODEL}"
)

print(
    f"Saved metrics: "
    f"{OUTPUT_METRICS}"
)

print()
print(
    "✅ Hierarchical generator training completed"
)