from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
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
    "generator_positional_train.npz"
)

VALIDATION_FILE = (
    "artifacts/features/"
    "generator_positional_validation.npz"
)

TEST_FILE = (
    "artifacts/features/"
    "generator_positional_test.npz"
)


MODEL_OUTPUT = Path(
    "artifacts/models/"
    "generator_positional_next_token.joblib"
)

METRICS_OUTPUT = Path(
    "artifacts/metrics/"
    "generator_positional_next_token_metrics.json"
)


OLD_METRICS_FILE = Path(
    "artifacts/metrics/"
    "generator_next_token_metrics.json"
)


# ============================================================
# Hyperparameters
# ============================================================

PCA_COMPONENTS = [
    10,
    20,
    40,
    60,
    100,
]

RIDGE_ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# ============================================================
# Load split
# ============================================================

def load_split(
    path,
):

    data = np.load(
        path,
        allow_pickle=True,
    )


    return {
        "X": data[
            "X"
        ].astype(
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

        "feedback_tokens": data[
            "feedback_tokens"
        ].astype(str),

        "feedback_positions": data[
            "feedback_positions"
        ].astype(
            np.int32
        ),

        "seeds": data[
            "seeds"
        ].astype(
            np.int32
        ),
    }


# ============================================================
# Header
# ============================================================

print()
print(
    "================================================"
)

print(
    " PyFly POSITIONAL NEXT-TOKEN DECODER TRAINING"
)

print(
    "================================================"
)

print()


# ============================================================
# Load data
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


X_train = train[
    "X"
]

y_train = train[
    "targets"
]


X_val = validation[
    "X"
]

y_val = validation[
    "targets"
]


X_test = test[
    "X"
]

y_test = test[
    "targets"
]


print(
    f"Train:      "
    f"{X_train.shape}"
)

print(
    f"Validation: "
    f"{X_val.shape}"
)

print(
    f"Test:       "
    f"{X_test.shape}"
)


# ============================================================
# Leakage check
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
    "Logical-prompt leakage check: PASS"
)


# ============================================================
# Position consistency
# ============================================================

assert np.array_equal(
    train[
        "positions"
    ],
    train[
        "feedback_positions"
    ],
)

assert np.array_equal(
    validation[
        "positions"
    ],
    validation[
        "feedback_positions"
    ],
)

assert np.array_equal(
    test[
        "positions"
    ],
    test[
        "feedback_positions"
    ],
)


print(
    "Feedback-position consistency: PASS"
)


# ============================================================
# Classes
# ============================================================

classes = sorted(
    np.unique(
        y_train
    )
)


chance = (
    1.0
    /
    len(classes)
)


print()
print(
    f"Next-token classes: "
    f"{len(classes)}"
)

print(
    f"Uniform chance: "
    f"{chance * 100:.2f}%"
)


# ============================================================
# Training target statistics
# ============================================================

print()
print(
    "Training target distribution:"
)

print()


for token in classes:

    count = int(
        np.sum(
            y_train
            ==
            token
        )
    )


    print(
        f"{token:12} "
        f"{count:4}"
    )


# ============================================================
# Majority baseline
# ============================================================

unique_tokens, counts = np.unique(
    y_train,
    return_counts=True,
)


majority_token = unique_tokens[
    np.argmax(
        counts
    )
]


majority_predictions = np.full(
    len(
        y_test
    ),
    majority_token,
)


majority_accuracy = accuracy_score(
    y_test,
    majority_predictions,
)


print()
print(
    f"Majority token: "
    f"{majority_token}"
)

print(
    f"Majority baseline: "
    f"{majority_accuracy * 100:.2f}%"
)


# ============================================================
# Model factory
# ============================================================

def create_model(
    components,
    alpha,
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
                    n_components=(
                        components
                    ),

                    svd_solver=(
                        "randomized"
                    ),

                    random_state=42,
                ),
            ),

            (
                "ridge",
                RidgeClassifier(
                    alpha=alpha,
                ),
            ),
        ]
    )


# ============================================================
# Validation search
# ============================================================

print()
print(
    "================================================"
)

print(
    "              VALIDATION SEARCH"
)

print(
    "================================================"
)

print()


best_accuracy = -1.0

best_macro_f1 = -1.0

best_components = None

best_alpha = None


search_results = []


for components in (
    PCA_COMPONENTS
):

    if components >= len(
        X_train
    ):

        continue


    for alpha in (
        RIDGE_ALPHAS
    ):

        model = create_model(
            components=components,
            alpha=alpha,
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


        macro_f1 = f1_score(
            y_val,
            predictions,

            average="macro",

            zero_division=0,
        )


        search_results.append(
            {
                "components": int(
                    components
                ),

                "alpha": float(
                    alpha
                ),

                "accuracy": float(
                    accuracy
                ),

                "macro_f1": float(
                    macro_f1
                ),
            }
        )


        print(
            f"PCA={components:3} "
            f"alpha={alpha:7.3f} "
            f"accuracy="
            f"{accuracy * 100:6.2f}% "
            f"macro_F1="
            f"{macro_f1 * 100:6.2f}%"
        )


        # ----------------------------------------------------
        # Accuracy first.
        # Macro F1 breaks ties.
        # ----------------------------------------------------

        if (
            accuracy
            >
            best_accuracy
        ) or (
            accuracy
            ==
            best_accuracy
            and
            macro_f1
            >
            best_macro_f1
        ):

            best_accuracy = (
                accuracy
            )

            best_macro_f1 = (
                macro_f1
            )

            best_components = (
                components
            )

            best_alpha = (
                alpha
            )


# ============================================================
# Best configuration
# ============================================================

print()
print(
    "================================================"
)

print(
    "            BEST CONFIGURATION"
)

print(
    "================================================"
)

print()


print(
    f"PCA components: "
    f"{best_components}"
)

print(
    f"Ridge alpha:    "
    f"{best_alpha}"
)

print(
    f"Validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Validation macro F1: "
    f"{best_macro_f1 * 100:.2f}%"
)


# ============================================================
# Refit train + validation
# ============================================================

print()
print(
    "Refitting on train + validation..."
)


X_final_train = np.concatenate(
    [
        X_train,
        X_val,
    ],

    axis=0,
)


y_final_train = np.concatenate(
    [
        y_train,
        y_val,
    ],

    axis=0,
)


final_model = create_model(
    components=(
        best_components
    ),

    alpha=(
        best_alpha
    ),
)


final_model.fit(
    X_final_train,
    y_final_train,
)


# ============================================================
# Test predictions
# ============================================================

test_predictions = (
    final_model.predict(
        X_test
    )
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)


test_macro_f1 = f1_score(
    y_test,
    test_predictions,

    average="macro",

    zero_division=0,
)


# ============================================================
# Decision scores
# ============================================================

scores = (
    final_model.decision_function(
        X_test
    )
)


scores = np.asarray(
    scores
)


if scores.ndim == 1:

    scores = np.column_stack(
        [
            -scores,
            scores,
        ]
    )


model_classes = np.asarray(
    final_model.named_steps[
        "ridge"
    ].classes_
)


# ============================================================
# Top-K
# ============================================================

def top_k_accuracy(
    scores,
    true_labels,
    class_names,
    k,
):

    k = min(
        k,
        len(
            class_names
        ),
    )


    top_indices = np.argsort(
        scores,
        axis=1,
    )[
        :,
        -k:
    ]


    correct = 0


    for row_index, true_token in enumerate(
        true_labels
    ):

        candidates = {
            str(
                class_names[
                    index
                ]
            )

            for index in (
                top_indices[
                    row_index
                ]
            )
        }


        if true_token in candidates:

            correct += 1


    return (
        correct
        /
        len(
            true_labels
        )
    )


top2_accuracy = (
    top_k_accuracy(
        scores,
        y_test,
        model_classes,
        2,
    )
)


top3_accuracy = (
    top_k_accuracy(
        scores,
        y_test,
        model_classes,
        3,
    )
)


# ============================================================
# Accuracy by position
# ============================================================

print()
print(
    "================================================"
)

print(
    "          ACCURACY BY OUTPUT POSITION"
)

print(
    "================================================"
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

        test_predictions[
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
# Accuracy by token
# ============================================================

print()
print(
    "================================================"
)

print(
    "              ACCURACY BY TOKEN"
)

print(
    "================================================"
)

print()


token_metrics = {}


for token in sorted(
    np.unique(
        y_test
    )
):

    mask = (
        y_test
        ==
        token
    )


    accuracy = accuracy_score(
        y_test[
            mask
        ],

        test_predictions[
            mask
        ],
    )


    token_metrics[
        token
    ] = float(
        accuracy
    )


    print(
        f"{token:12} "
        f"{accuracy * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Semantic-position metric
#
# These are the positions that caused the old generator
# problems:
#
# 0 = first variable
# 2 = operand
# 3 = operator / termination decision
# 4 = second operand
# ============================================================

semantic_positions = [
    0,
    2,
    3,
    4,
]


semantic_mask = np.isin(
    test[
        "positions"
    ],
    semantic_positions,
)


semantic_accuracy = (
    accuracy_score(
        y_test[
            semantic_mask
        ],

        test_predictions[
            semantic_mask
        ],
    )
)


print()
print(
    f"Semantic-position accuracy "
    f"(0,2,3,4): "
    f"{semantic_accuracy * 100:.2f}%"
)


# ============================================================
# Classification report
# ============================================================

print()
print(
    "Classification report:"
)

print()


print(
    classification_report(
        y_test,
        test_predictions,

        zero_division=0,
    )
)


# ============================================================
# Confusion matrix
# ============================================================

confusion = confusion_matrix(
    y_test,
    test_predictions,

    labels=model_classes,
)


print()
print(
    "Class order:"
)

print(
    model_classes.tolist()
)

print()
print(
    "Confusion matrix:"
)

print(
    confusion
)


# ============================================================
# Old model metrics
# ============================================================

old_metrics = None


if OLD_METRICS_FILE.exists():

    with OLD_METRICS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        old_metrics = json.load(
            file
        )


# ============================================================
# Direct comparison
# ============================================================

print()
print()
print(
    "================================================"
)

print(
    "       POSITIONAL FEEDBACK COMPARISON"
)

print(
    "================================================"
)

print()


print(
    f"Uniform chance:              "
    f"{chance * 100:6.2f}%"
)

print(
    f"Majority baseline:           "
    f"{majority_accuracy * 100:6.2f}%"
)


if old_metrics is not None:

    old_top1 = old_metrics[
        "test_top1_accuracy"
    ]

    old_top2 = old_metrics[
        "test_top2_accuracy"
    ]

    old_top3 = old_metrics[
        "test_top3_accuracy"
    ]

    old_f1 = old_metrics[
        "test_macro_f1"
    ]


    print()
    print(
        "OLD feedback-token-only model:"
    )

    print(
        f"  Top-1:                    "
        f"{old_top1 * 100:6.2f}%"
    )

    print(
        f"  Top-2:                    "
        f"{old_top2 * 100:6.2f}%"
    )

    print(
        f"  Top-3:                    "
        f"{old_top3 * 100:6.2f}%"
    )

    print(
        f"  Macro F1:                 "
        f"{old_f1 * 100:6.2f}%"
    )


print()
print(
    "NEW positional-feedback model:"
)

print(
    f"  Top-1:                    "
    f"{test_accuracy * 100:6.2f}%"
)

print(
    f"  Top-2:                    "
    f"{top2_accuracy * 100:6.2f}%"
)

print(
    f"  Top-3:                    "
    f"{top3_accuracy * 100:6.2f}%"
)

print(
    f"  Macro F1:                 "
    f"{test_macro_f1 * 100:6.2f}%"
)

print(
    f"  Semantic positions:       "
    f"{semantic_accuracy * 100:6.2f}%"
)


# ============================================================
# Improvement
# ============================================================

if old_metrics is not None:

    improvement = (
        test_accuracy
        -
        old_top1
    )


    print()
    print(
        f"Top-1 change: "
        f"{improvement * 100:+.2f} "
        f"percentage points"
    )


    if improvement > 0:

        print(
            "✅ Positional feedback improved "
            "next-token decoding."
        )

    elif improvement == 0:

        print(
            "⚠️ Positional feedback tied "
            "the previous decoder."
        )

    else:

        print(
            "⚠️ Positional feedback reduced "
            "overall next-token accuracy."
        )


# ============================================================
# Save model
# ============================================================

MODEL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    final_model,
    MODEL_OUTPUT,
)


# ============================================================
# Save metrics
# ============================================================

METRICS_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = {
    "uniform_chance": float(
        chance
    ),

    "majority_baseline": float(
        majority_accuracy
    ),

    "validation_accuracy": float(
        best_accuracy
    ),

    "validation_macro_f1": float(
        best_macro_f1
    ),

    "test_top1_accuracy": float(
        test_accuracy
    ),

    "test_top2_accuracy": float(
        top2_accuracy
    ),

    "test_top3_accuracy": float(
        top3_accuracy
    ),

    "test_macro_f1": float(
        test_macro_f1
    ),

    "semantic_position_accuracy": float(
        semantic_accuracy
    ),

    "pca_components": int(
        best_components
    ),

    "ridge_alpha": float(
        best_alpha
    ),

    "classes": (
        model_classes.tolist()
    ),

    "accuracy_by_position": (
        position_metrics
    ),

    "accuracy_by_token": (
        token_metrics
    ),

    "search_results": (
        search_results
    ),
}


if old_metrics is not None:

    metrics[
        "old_model"
    ] = {
        "top1": float(
            old_top1
        ),

        "top2": float(
            old_top2
        ),

        "top3": float(
            old_top3
        ),

        "macro_f1": float(
            old_f1
        ),

        "top1_change": float(
            test_accuracy
            -
            old_top1
        ),
    }


with METRICS_OUTPUT.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# Final
# ============================================================

print()
print(
    f"Saved model: "
    f"{MODEL_OUTPUT}"
)

print(
    f"Saved metrics: "
    f"{METRICS_OUTPUT}"
)

print()
print(
    "✅ Positional-feedback decoder training completed"
)