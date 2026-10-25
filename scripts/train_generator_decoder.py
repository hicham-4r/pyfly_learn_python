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


MODEL_OUTPUT = Path(
    "artifacts/models/"
    "generator_next_token.joblib"
)

METRICS_OUTPUT = Path(
    "artifacts/metrics/"
    "generator_next_token_metrics.json"
)


# ============================================================
# Hyperparameter search
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
        ].astype(np.int32),

        "feedback": data[
            "feedback_tokens"
        ].astype(str),

        "seeds": data[
            "seeds"
        ].astype(np.int32),
    }


# ============================================================
# Load datasets
# ============================================================

print()
print(
    "============================================"
)

print(
    "    PyFly NEXT-TOKEN DECODER TRAINING"
)

print(
    "============================================"
)

print()


train = load_split(
    TRAIN_FILE
)

validation = load_split(
    VALIDATION_FILE
)

test = load_split(
    TEST_FILE
)


X_train = train["X"]
y_train = train["targets"]

X_val = validation["X"]
y_val = validation["targets"]

X_test = test["X"]
y_test = test["targets"]


print(
    f"Train:      {X_train.shape}"
)

print(
    f"Validation: {X_val.shape}"
)

print(
    f"Test:       {X_test.shape}"
)


# ============================================================
# Leakage check
# ============================================================

train_logical_ids = set(
    train["example_ids"].tolist()
)

val_logical_ids = set(
    validation["example_ids"].tolist()
)

test_logical_ids = set(
    test["example_ids"].tolist()
)


assert train_logical_ids.isdisjoint(
    val_logical_ids
)

assert train_logical_ids.isdisjoint(
    test_logical_ids
)

assert val_logical_ids.isdisjoint(
    test_logical_ids
)


print()
print(
    "Logical-prompt leakage check: PASS"
)


# ============================================================
# Target vocabulary
# ============================================================

classes = sorted(
    np.unique(
        y_train
    )
)


print()
print(
    f"Next-token classes: "
    f"{len(classes)}"
)

print()


for token in classes:

    count = np.sum(
        y_train == token
    )

    print(
        f"{token:12} "
        f"{count:4} training examples"
    )


chance = (
    1.0
    /
    len(classes)
)


# ============================================================
# Baseline: majority token
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
    len(y_test),
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
                    n_components=components,
                    svd_solver="randomized",
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
# Hyperparameter search
# ============================================================

print()
print(
    "============================================"
)

print(
    "          VALIDATION SEARCH"
)

print(
    "============================================"
)

print()


best_model = None

best_accuracy = -1.0
best_macro_f1 = -1.0

best_components = None
best_alpha = None


for components in PCA_COMPONENTS:

    # PCA cannot exceed available sample count.
    if components >= len(
        X_train
    ):
        continue


    for alpha in RIDGE_ALPHAS:

        model = create_model(
            components,
            alpha,
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


        print(
            f"PCA={components:3} "
            f"alpha={alpha:7.3f} "
            f"accuracy="
            f"{accuracy * 100:6.2f}% "
            f"macro_F1="
            f"{macro_f1 * 100:6.2f}%"
        )


        # Accuracy first.
        # Macro F1 breaks ties.
        if (
            accuracy > best_accuracy
            or (
                accuracy == best_accuracy
                and macro_f1 > best_macro_f1
            )
        ):

            best_accuracy = accuracy

            best_macro_f1 = macro_f1

            best_components = components

            best_alpha = alpha

            best_model = model


# ============================================================
# Best validation configuration
# ============================================================

print()
print(
    "============================================"
)

print(
    "          BEST CONFIGURATION"
)

print(
    "============================================"
)

print()


print(
    f"PCA components:     "
    f"{best_components}"
)

print(
    f"Ridge alpha:        "
    f"{best_alpha}"
)

print(
    f"Validation accuracy:"
    f" {best_accuracy * 100:.2f}%"
)

print(
    f"Validation macro F1:"
    f" {best_macro_f1 * 100:.2f}%"
)


# ============================================================
# Refit on TRAIN + VALIDATION
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
    best_components,
    best_alpha,
)


final_model.fit(
    X_final_train,
    y_final_train,
)


# ============================================================
# Test prediction
# ============================================================

test_predictions = final_model.predict(
    X_test
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
# Top-K prediction
# ============================================================

decision_scores = (
    final_model.decision_function(
        X_test
    )
)


model_classes = (
    final_model.named_steps[
        "ridge"
    ].classes_
)


# Ensure matrix shape.
if decision_scores.ndim == 1:

    decision_scores = np.column_stack(
        [
            -decision_scores,
            decision_scores,
        ]
    )


def top_k_accuracy(
    scores,
    true_labels,
    class_names,
    k,
):

    k = min(
        k,
        len(class_names),
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

        predicted_tokens = {
            class_names[index]
            for index in top_indices[
                row_index
            ]
        }


        if true_token in predicted_tokens:

            correct += 1


    return (
        correct
        /
        len(true_labels)
    )


top2_accuracy = top_k_accuracy(
    decision_scores,
    y_test,
    model_classes,
    2,
)


top3_accuracy = top_k_accuracy(
    decision_scores,
    y_test,
    model_classes,
    3,
)


# ============================================================
# Accuracy by target position
# ============================================================

positions = test[
    "positions"
]


position_metrics = {}


print()
print(
    "============================================"
)

print(
    "       ACCURACY BY OUTPUT POSITION"
)

print(
    "============================================"
)

print()


for position in sorted(
    np.unique(
        positions
    )
):

    mask = (
        positions
        == position
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
        int(position)
    ] = float(
        position_accuracy
    )


    print(
        f"Position {position}: "
        f"{position_accuracy * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Accuracy by target token
# ============================================================

print()
print(
    "============================================"
)

print(
    "         ACCURACY BY TOKEN"
)

print(
    "============================================"
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
        == token
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
# Final result
# ============================================================

print()
print()
print(
    "============================================"
)

print(
    "       PyFly NEXT-TOKEN TEST RESULT"
)

print(
    "============================================"
)

print()


print(
    f"Uniform chance:       "
    f"{chance * 100:.2f}%"
)

print(
    f"Majority baseline:    "
    f"{majority_accuracy * 100:.2f}%"
)

print(
    f"Validation accuracy:  "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"TEST top-1 accuracy:  "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"TEST top-2 accuracy:  "
    f"{top2_accuracy * 100:.2f}%"
)

print(
    f"TEST top-3 accuracy:  "
    f"{top3_accuracy * 100:.2f}%"
)

print(
    f"TEST macro F1:        "
    f"{test_macro_f1 * 100:.2f}%"
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
# Show some individual predictions
# ============================================================

print()
print(
    "============================================"
)

print(
    "         SAMPLE PREDICTIONS"
)

print(
    "============================================"
)

print()


for index in range(
    min(
        40,
        len(y_test),
    )
):

    true = y_test[
        index
    ]

    predicted = test_predictions[
        index
    ]


    mark = (
        "✓"
        if true == predicted
        else "✗"
    )


    print(
        f"{mark} "
        f"{test['example_ids'][index]:16} "
        f"seed={test['seeds'][index]} "
        f"pos={test['positions'][index]} "
        f"feedback="
        f"{test['feedback'][index]:10} "
        f"target="
        f"{true:10} "
        f"pred="
        f"{predicted}"
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

    "train_samples": int(
        len(X_train)
    ),

    "validation_samples": int(
        len(X_val)
    ),

    "test_samples": int(
        len(X_test)
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
    "✅ PyFly next-token decoder training completed"
)