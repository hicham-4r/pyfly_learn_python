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
# ============================================================

TRAIN_FILE = (
    "artifacts/features/"
    "family_rich_train.npz"
)

VALIDATION_FILE = (
    "artifacts/features/"
    "family_rich_validation.npz"
)

TEST_FILE = (
    "artifacts/features/"
    "family_rich_test.npz"
)


OUTPUT_MODEL = Path(
    "artifacts/models/"
    "family_rich_best.joblib"
)

OUTPUT_METRICS = Path(
    "artifacts/metrics/"
    "family_rich_metrics.json"
)


# ============================================================
# Previous family classifier baseline
# ============================================================

OLD_FAMILY_ACCURACY = 0.3667
OLD_FAMILY_MACRO_F1 = 0.3282


# ============================================================
# Search spaces
# ============================================================

PCA_COMPONENTS = [
    5,
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


LOGISTIC_C = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


SVM_C = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]


# ============================================================
# Load feature split
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

        "families": data[
            "families"
        ].astype(str),

        "intents": data[
            "intents"
        ].astype(str),

        "example_ids": data[
            "example_ids"
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
    "      PyFly RICH FAMILY CLASSIFIER TRAINING"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load datasets
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
    "families"
]


X_val = validation[
    "X"
]

y_val = validation[
    "families"
]


X_test = test[
    "X"
]

y_test = test[
    "families"
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
# Logical prompt leakage check
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
    len(
        classes
    )
)


print()
print(
    f"Classes: "
    f"{classes}"
)

print(
    f"Chance accuracy: "
    f"{chance * 100:.2f}%"
)


# ============================================================
# Training distribution
# ============================================================

print()
print(
    "Training family distribution:"
)


for family in classes:

    count = int(
        np.sum(
            y_train
            ==
            family
        )
    )

    print(
        f"  {family:10} "
        f"{count}"
    )


# ============================================================
# Pipeline helper
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
                    n_components=components,
                    svd_solver="randomized",
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
# Candidate list
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
                    "family": "Ridge",

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

        for c_value in LOGISTIC_C:

            candidates.append(
                {
                    "family": (
                        "Logistic"
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

        for c_value in LOGISTIC_C:

            candidates.append(
                {
                    "family": (
                        "Balanced Logistic"
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

        for c_value in SVM_C:

            candidates.append(
                {
                    "family": (
                        "Linear SVM"
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

        for c_value in SVM_C:

            candidates.append(
                {
                    "family": (
                        "Balanced Linear SVM"
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

family_best = {}

all_results = []


max_components = min(
    X_train.shape[
        0
    ],

    X_train.shape[
        1
    ],
)


for index, candidate in enumerate(
    candidates,
    start=1,
):

    components = candidate[
        "components"
    ]


    if components >= max_components:

        continue


    model = candidate[
        "model"
    ]


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


    result = {
        "family": candidate[
            "family"
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
    # Best inside classifier family
    # --------------------------------------------------------

    classifier_family = (
        candidate[
            "family"
        ]
    )


    existing = family_best.get(
        classifier_family
    )


    if (
        existing is None
        or
        macro_f1
        >
        existing[
            "validation_macro_f1"
        ]
        or (
            macro_f1
            ==
            existing[
                "validation_macro_f1"
            ]
            and
            accuracy
            >
            existing[
                "validation_accuracy"
            ]
        )
    ):

        family_best[
            classifier_family
        ] = {
            **result,

            "candidate": (
                candidate
            ),
        }


    # --------------------------------------------------------
    # Global best:
    #
    # Macro F1 first.
    # Accuracy breaks ties.
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

            "candidate": (
                candidate
            ),
        }


# ============================================================
# Best by model family
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "         BEST VALIDATION MODEL BY FAMILY"
)

print(
    "===================================================="
)

print()


for classifier_family, result in (
    family_best.items()
):

    print(
        f"{classifier_family:22} "
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
# Winning configuration
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "           SELECTED FAMILY CLASSIFIER"
)

print(
    "===================================================="
)

print()


print(
    f"Classifier: "
    f"{best['family']}"
)

print(
    f"PCA components: "
    f"{best['components']}"
)

print(
    f"Parameter: "
    f"{best['parameter']}"
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
# Refit selected model on train + validation
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


winning_model = (
    best[
        "candidate"
    ][
        "model"
    ]
)


print()
print(
    "Refitting selected model "
    "on train + validation..."
)


winning_model.fit(
    X_final_train,
    y_final_train,
)


# ============================================================
# Test
# ============================================================

test_prediction = (
    winning_model.predict(
        X_test
    )
)


test_accuracy = accuracy_score(
    y_test,
    test_prediction,
)


test_macro_f1 = f1_score(
    y_test,
    test_prediction,

    average="macro",

    zero_division=0,
)


# ============================================================
# Final family classification result
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "        RICH FAMILY CLASSIFIER TEST RESULT"
)

print(
    "===================================================="
)

print()


print(
    f"Chance:             "
    f"{chance * 100:.2f}%"
)

print(
    f"Old family model:   "
    f"{OLD_FAMILY_ACCURACY * 100:.2f}%"
)

print(
    f"Rich family model:  "
    f"{test_accuracy * 100:.2f}%"
)


print()
print(
    f"Old macro F1:       "
    f"{OLD_FAMILY_MACRO_F1 * 100:.2f}%"
)

print(
    f"Rich macro F1:      "
    f"{test_macro_f1 * 100:.2f}%"
)


print()
print(
    f"Accuracy change: "
    f"{(test_accuracy - OLD_FAMILY_ACCURACY) * 100:+.2f} pp"
)

print(
    f"Macro-F1 change: "
    f"{(test_macro_f1 - OLD_FAMILY_MACRO_F1) * 100:+.2f} pp"
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
        test_prediction,

        labels=[
            "COPY",
            "SUBTRACT",
            "DIVIDE",
        ],

        zero_division=0,
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    test_prediction,

    labels=[
        "COPY",
        "SUBTRACT",
        "DIVIDE",
    ],
)


print()
print(
    "Class order:"
)

print(
    [
        "COPY",
        "SUBTRACT",
        "DIVIDE",
    ]
)

print()
print(
    "Confusion matrix:"
)

print(
    cm
)


# ============================================================
# Per-example predictions
# ============================================================

print()
print(
    "===================================================="
)

print(
    "              TEST PREDICTIONS"
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
) in zip(
    test[
        "example_ids"
    ],

    test[
        "seeds"
    ],

    y_test,

    test_prediction,
):

    mark = (
        "✓"
        if true_family
        ==
        predicted_family
        else "✗"
    )


    print(
        f"{mark} "
        f"{example_id:16} "
        f"seed="
        f"{seed} "
        f"true="
        f"{true_family:8} "
        f"pred="
        f"{predicted_family}"
    )


# ============================================================
# Accuracy by logical prompt
# ============================================================

print()
print(
    "===================================================="
)

print(
    "          ACCURACY BY LOGICAL PROMPT"
)

print(
    "===================================================="
)

print()


prompt_metrics = {}


for example_id in sorted(
    np.unique(
        test[
            "example_ids"
        ]
    )
):

    mask = (
        test[
            "example_ids"
        ]
        ==
        example_id
    )


    prompt_accuracy = (
        accuracy_score(
            y_test[
                mask
            ],

            test_prediction[
                mask
            ],
        )
    )


    prompt_metrics[
        example_id
    ] = float(
        prompt_accuracy
    )


    print(
        f"{example_id:16} "
        f"{prompt_accuracy * 100:6.2f}%"
    )


# ============================================================
# Save model
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


family_results = {}


for classifier_family, result in (
    family_best.items()
):

    family_results[
        classifier_family
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
    "selected_classifier": (
        best[
            "family"
        ]
    ),

    "pca_components": int(
        best[
            "components"
        ]
    ),

    "parameter": float(
        best[
            "parameter"
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

    "test_accuracy": float(
        test_accuracy
    ),

    "test_macro_f1": float(
        test_macro_f1
    ),

    "old_family_accuracy": (
        OLD_FAMILY_ACCURACY
    ),

    "old_family_macro_f1": (
        OLD_FAMILY_MACRO_F1
    ),

    "accuracy_change": float(
        test_accuracy
        -
        OLD_FAMILY_ACCURACY
    ),

    "macro_f1_change": float(
        test_macro_f1
        -
        OLD_FAMILY_MACRO_F1
    ),

    "accuracy_by_prompt": (
        prompt_metrics
    ),

    "family_validation_results": (
        family_results
    ),

    "all_validation_results": (
        all_results
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


# ============================================================
# Interpretation
# ============================================================

print()
print(
    "===================================================="
)

print(
    "              INTERPRETATION"
)

print(
    "===================================================="
)

print()


if test_accuracy >= 0.85:

    print(
        "✅ Excellent family separation."
    )

    print(
        "The hierarchical generator is now "
        "worth testing in closed loop."
    )


elif test_accuracy >= 0.75:

    print(
        "✅ Strong family separation."
    )

    print(
        "This is likely sufficient to improve "
        "hierarchical generation."
    )


elif test_accuracy >= 0.60:

    print(
        "⚠️ Family separation improved, "
        "but errors remain substantial."
    )


elif test_accuracy > OLD_FAMILY_ACCURACY:

    print(
        "⚠️ Improvement exists, but the family "
        "gate is still weak."
    )


else:

    print(
        "❌ Rich prompt features did not improve "
        "family classification."
    )


print()
print(
    f"Saved model: "
    f"{OUTPUT_MODEL}"
)

print(
    f"Saved metrics: "
    f"{OUTPUT_METRICS}"
)

print()
print(
    "✅ Rich family classifier training completed"
)