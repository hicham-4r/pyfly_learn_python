from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# Files
#
# IMPORTANT:
# We intentionally use the ORIGINAL Step 62 generator
# features, because they slightly outperformed the
# positional-feedback version.
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


OLD_METRICS_FILE = Path(
    "artifacts/metrics/"
    "generator_next_token_metrics.json"
)


OUTPUT_MODEL = Path(
    "artifacts/models/"
    "generator_best_decoder.joblib"
)


OUTPUT_METRICS = Path(
    "artifacts/metrics/"
    "generator_decoder_comparison.json"
)


# ============================================================
# Search space
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


LOGISTIC_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


SVM_C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]


# ============================================================
# Load one split
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
# Header
# ============================================================

print()
print(
    "===================================================="
)

print(
    "       PyFly GENERATOR DECODER COMPARISON"
)

print(
    "===================================================="
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

val_ids = set(
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
    val_ids
)

assert train_ids.isdisjoint(
    test_ids
)

assert val_ids.isdisjoint(
    test_ids
)


print()
print(
    "Logical-prompt leakage check: PASS"
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
    f"Classes: "
    f"{len(classes)}"
)

print(
    f"Uniform chance: "
    f"{chance * 100:.2f}%"
)


# ============================================================
# Training distribution
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
    f"Majority accuracy: "
    f"{majority_accuracy * 100:.2f}%"
)


# ============================================================
# Pipeline builder
# ============================================================

def make_pipeline(
    components,
    classifier,
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
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# Candidate configurations
# ============================================================

def build_candidates():

    candidates = []


    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    for components in PCA_COMPONENTS:

        for alpha in RIDGE_ALPHAS:

            candidates.append(
                {
                    "family": (
                        "Ridge"
                    ),

                    "name": (
                        f"Ridge "
                        f"PCA={components} "
                        f"alpha={alpha}"
                    ),

                    "components": (
                        components
                    ),

                    "parameter": (
                        alpha
                    ),

                    "model": (
                        make_pipeline(
                            components,

                            RidgeClassifier(
                                alpha=alpha,
                            ),
                        )
                    ),
                }
            )


    # --------------------------------------------------------
    # Balanced Ridge
    # --------------------------------------------------------

    for components in PCA_COMPONENTS:

        for alpha in RIDGE_ALPHAS:

            candidates.append(
                {
                    "family": (
                        "Balanced Ridge"
                    ),

                    "name": (
                        f"Balanced Ridge "
                        f"PCA={components} "
                        f"alpha={alpha}"
                    ),

                    "components": (
                        components
                    ),

                    "parameter": (
                        alpha
                    ),

                    "model": (
                        make_pipeline(
                            components,

                            RidgeClassifier(
                                alpha=alpha,

                                class_weight=(
                                    "balanced"
                                ),
                            ),
                        )
                    ),
                }
            )


    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    for components in PCA_COMPONENTS:

        for c_value in LOGISTIC_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Logistic"
                    ),

                    "name": (
                        f"Logistic "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),

                    "components": (
                        components
                    ),

                    "parameter": (
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            components,

                            LogisticRegression(
                                C=c_value,

                                solver="lbfgs",

                                max_iter=5000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )


    # --------------------------------------------------------
    # Balanced Logistic Regression
    # --------------------------------------------------------

    for components in PCA_COMPONENTS:

        for c_value in LOGISTIC_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Balanced Logistic"
                    ),

                    "name": (
                        f"Balanced Logistic "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),

                    "components": (
                        components
                    ),

                    "parameter": (
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            components,

                            LogisticRegression(
                                C=c_value,

                                solver="lbfgs",

                                class_weight=(
                                    "balanced"
                                ),

                                max_iter=5000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )


    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    for components in PCA_COMPONENTS:

        for c_value in SVM_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Linear SVM"
                    ),

                    "name": (
                        f"Linear SVM "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),

                    "components": (
                        components
                    ),

                    "parameter": (
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            components,

                            LinearSVC(
                                C=c_value,

                                max_iter=10000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )


    # --------------------------------------------------------
    # Balanced Linear SVM
    # --------------------------------------------------------

    for components in PCA_COMPONENTS:

        for c_value in SVM_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Balanced Linear SVM"
                    ),

                    "name": (
                        f"Balanced Linear SVM "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),

                    "components": (
                        components
                    ),

                    "parameter": (
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            components,

                            LinearSVC(
                                C=c_value,

                                class_weight=(
                                    "balanced"
                                ),

                                max_iter=10000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )


    return candidates


# ============================================================
# Validation search
#
# PRIMARY selection metric:
# macro F1
#
# Tie-break:
# accuracy
# ============================================================

print()
print(
    "===================================================="
)

print(
    "               VALIDATION SEARCH"
)

print(
    "===================================================="
)

print()


candidates = build_candidates()


best = None

all_results = []


# Track best configuration inside each family.
family_best = {}


for index, candidate in enumerate(
    candidates,
    start=1,
):

    components = candidate[
        "components"
    ]


    if components >= min(
        X_train.shape[0],
        X_train.shape[1],
    ):

        continue


    model = candidate[
        "model"
    ]


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


    result = {
        "family": candidate[
            "family"
        ],

        "name": candidate[
            "name"
        ],

        "components": int(
            components
        ),

        "parameter": float(
            candidate[
                "parameter"
            ]
        ),

        "validation_accuracy": float(
            accuracy
        ),

        "validation_macro_f1": float(
            macro_f1
        ),
    }


    all_results.append(
        result
    )


    print(
        f"[{index:03}/"
        f"{len(candidates):03}] "
        f"{candidate['family']:20} "
        f"PCA={components:3} "
        f"param="
        f"{candidate['parameter']:8.4f} "
        f"acc="
        f"{accuracy * 100:6.2f}% "
        f"F1="
        f"{macro_f1 * 100:6.2f}%"
    )


    # --------------------------------------------------------
    # Family best
    # --------------------------------------------------------

    family = candidate[
        "family"
    ]


    current_family_best = (
        family_best.get(
            family
        )
    )


    if (
        current_family_best
        is None
        or
        macro_f1
        >
        current_family_best[
            "validation_macro_f1"
        ]
        or (
            macro_f1
            ==
            current_family_best[
                "validation_macro_f1"
            ]
            and
            accuracy
            >
            current_family_best[
                "validation_accuracy"
            ]
        )
    ):

        family_best[
            family
        ] = {
            **result,

            "model_spec": candidate,
        }


    # --------------------------------------------------------
    # Global best
    # --------------------------------------------------------

    if (
        best is None
        or
        macro_f1
        >
        best[
            "validation_macro_f1"
        ]
        or (
            macro_f1
            ==
            best[
                "validation_macro_f1"
            ]
            and
            accuracy
            >
            best[
                "validation_accuracy"
            ]
        )
    ):

        best = {
            **result,

            "model_spec": candidate,
        }


# ============================================================
# Family summary
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "          BEST VALIDATION MODEL BY FAMILY"
)

print(
    "===================================================="
)

print()


for family, result in family_best.items():

    print(
        f"{family:22} "
        f"acc="
        f"{result['validation_accuracy'] * 100:6.2f}% "
        f"F1="
        f"{result['validation_macro_f1'] * 100:6.2f}% "
        f"PCA="
        f"{result['components']:3} "
        f"param="
        f"{result['parameter']}"
    )


# ============================================================
# Winning validation configuration
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "            SELECTED DECODER"
)

print(
    "===================================================="
)

print()


print(
    f"Family: "
    f"{best['family']}"
)

print(
    f"Configuration: "
    f"{best['name']}"
)

print(
    f"Validation accuracy: "
    f"{best['validation_accuracy'] * 100:.2f}%"
)

print(
    f"Validation macro F1: "
    f"{best['validation_macro_f1'] * 100:.2f}%"
)


# ============================================================
# Rebuild winning model
# ============================================================

winning_spec = best[
    "model_spec"
]


winning_model = winning_spec[
    "model"
]


# ============================================================
# Refit on train + validation
# ============================================================

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


print()
print(
    "Refitting selected decoder on "
    "train + validation..."
)


winning_model.fit(
    X_final_train,
    y_final_train,
)


# ============================================================
# Test
# ============================================================

test_predictions = (
    winning_model.predict(
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
# Decision function
#
# All chosen classifiers expose decision_function().
# ============================================================

scores = (
    winning_model.decision_function(
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


classifier = (
    winning_model.named_steps[
        "classifier"
    ]
)


model_classes = np.asarray(
    classifier.classes_
).astype(str)


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


    for row_index, target in enumerate(
        true_labels
    ):

        candidates = {
            str(
                class_names[
                    class_index
                ]
            )

            for class_index
            in top_indices[
                row_index
            ]
        }


        if target in candidates:

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
    "===================================================="
)

print(
    "            ACCURACY BY POSITION"
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


    position_accuracy = accuracy_score(
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
        position_accuracy
    )


    print(
        f"Position {position}: "
        f"{position_accuracy * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Semantic positions
# ============================================================

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


semantic_mask = np.isin(
    test[
        "positions"
    ],
    SEMANTIC_POSITIONS,
)


semantic_accuracy = accuracy_score(
    y_test[
        semantic_mask
    ],
    test_predictions[
        semantic_mask
    ],
)


print()
print(
    f"Semantic-position accuracy: "
    f"{semantic_accuracy * 100:.2f}%"
)


# ============================================================
# Accuracy by token
# ============================================================

print()
print(
    "===================================================="
)

print(
    "              ACCURACY BY TOKEN"
)

print(
    "===================================================="
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


    token_accuracy = accuracy_score(
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
        token_accuracy
    )


    print(
        f"{token:12} "
        f"{token_accuracy * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
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

cm = confusion_matrix(
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
    cm
)


# ============================================================
# Old Ridge metrics
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
# Final comparison
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "       PyFly DECODER FINAL COMPARISON"
)

print(
    "===================================================="
)

print()


print(
    f"Uniform chance:         "
    f"{chance * 100:6.2f}%"
)

print(
    f"Majority baseline:      "
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

    old_macro_f1 = old_metrics[
        "test_macro_f1"
    ]


    print()
    print(
        "OLD Ridge decoder:"
    )

    print(
        f"  Top-1:              "
        f"{old_top1 * 100:6.2f}%"
    )

    print(
        f"  Top-2:              "
        f"{old_top2 * 100:6.2f}%"
    )

    print(
        f"  Top-3:              "
        f"{old_top3 * 100:6.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{old_macro_f1 * 100:6.2f}%"
    )


print()
print(
    f"NEW {best['family']}:"
)

print(
    f"  Top-1:              "
    f"{test_accuracy * 100:6.2f}%"
)

print(
    f"  Top-2:              "
    f"{top2_accuracy * 100:6.2f}%"
)

print(
    f"  Top-3:              "
    f"{top3_accuracy * 100:6.2f}%"
)

print(
    f"  Macro F1:           "
    f"{test_macro_f1 * 100:6.2f}%"
)

print(
    f"  Semantic positions: "
    f"{semantic_accuracy * 100:6.2f}%"
)


if old_metrics is not None:

    top1_change = (
        test_accuracy
        -
        old_top1
    )

    f1_change = (
        test_macro_f1
        -
        old_macro_f1
    )


    print()
    print(
        f"Top-1 change: "
        f"{top1_change * 100:+.2f} pp"
    )

    print(
        f"Macro-F1 change: "
        f"{f1_change * 100:+.2f} pp"
    )


# ============================================================
# Sample predictions
# ============================================================

print()
print(
    "===================================================="
)

print(
    "              SAMPLE PREDICTIONS"
)

print(
    "===================================================="
)

print()


for index in range(
    min(
        50,
        len(
            y_test
        ),
    )
):

    true_token = (
        y_test[
            index
        ]
    )


    predicted_token = (
        test_predictions[
            index
        ]
    )


    mark = (
        "✓"
        if true_token
        ==
        predicted_token
        else "✗"
    )


    ranked_indices = np.argsort(
        scores[
            index
        ]
    )[
        ::-1
    ]


    top3_tokens = [
        str(
            model_classes[
                class_index
            ]
        )

        for class_index
        in ranked_indices[
            :3
        ]
    ]


    print(
        f"{mark} "
        f"{test['example_ids'][index]:16} "
        f"seed="
        f"{test['seeds'][index]} "
        f"pos="
        f"{test['positions'][index]} "
        f"feedback="
        f"{test['feedback'][index]:10} "
        f"target="
        f"{true_token:10} "
        f"pred="
        f"{predicted_token:10} "
        f"top3="
        f"{top3_tokens}"
    )


# ============================================================
# Save winning model
# ============================================================

OUTPUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    winning_model,
    OUTPUT_MODEL,
)


# ============================================================
# Save metrics
# ============================================================

OUTPUT_METRICS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


serializable_family_results = {}


for family, result in (
    family_best.items()
):

    serializable_family_results[
        family
    ] = {
        "validation_accuracy": float(
            result[
                "validation_accuracy"
            ]
        ),

        "validation_macro_f1": float(
            result[
                "validation_macro_f1"
            ]
        ),

        "pca_components": int(
            result[
                "components"
            ]
        ),

        "parameter": float(
            result[
                "parameter"
            ]
        ),
    }


metrics = {
    "selected_family": (
        best[
            "family"
        ]
    ),

    "selected_configuration": (
        best[
            "name"
        ]
    ),

    "validation_accuracy": float(
        best[
            "validation_accuracy"
        ]
    ),

    "validation_macro_f1": float(
        best[
            "validation_macro_f1"
        ]
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

    "majority_baseline": float(
        majority_accuracy
    ),

    "uniform_chance": float(
        chance
    ),

    "accuracy_by_position": (
        position_metrics
    ),

    "accuracy_by_token": (
        token_metrics
    ),

    "family_validation_results": (
        serializable_family_results
    ),

    "all_validation_results": (
        all_results
    ),
}


if old_metrics is not None:

    metrics[
        "old_ridge"
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
            old_macro_f1
        ),

        "top1_change": float(
            test_accuracy
            -
            old_top1
        ),

        "macro_f1_change": float(
            test_macro_f1
            -
            old_macro_f1
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
    f"Saved winning model: "
    f"{OUTPUT_MODEL}"
)

print(
    f"Saved metrics: "
    f"{OUTPUT_METRICS}"
)

print()
print(
    "✅ PyFly decoder comparison completed"
)