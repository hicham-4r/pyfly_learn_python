import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Files
# ============================================================

TRAIN_FILE = "artifacts/features/template_train.npz"
VALIDATION_FILE = "artifacts/features/template_validation.npz"
TEST_FILE = "artifacts/features/template_test.npz"

MODEL_OUTPUT = Path(
    "artifacts/models/template_classifier.joblib"
)

METRICS_OUTPUT = Path(
    "artifacts/metrics/template_final_metrics.json"
)


# ============================================================
# Hyperparameter grid
# ============================================================

PCA_COMPONENTS = [
    5,
    10,
    20,
    40,
    60,
]

ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]


# ============================================================
# Load split helper
# ============================================================

def load_split(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    return (
        data["X"],
        data["intents"].astype(str),
        data["example_ids"].astype(str),
        data["seeds"],
    )


# ============================================================
# Load features
# ============================================================

print()
print("========================================")
print("      PyFly FINAL TEMPLATE MODEL")
print("========================================")
print()


X_train, y_train, train_ids, train_seeds = load_split(
    TRAIN_FILE
)

X_val, y_val, val_ids, val_seeds = load_split(
    VALIDATION_FILE
)

X_test, y_test, test_ids, test_seeds = load_split(
    TEST_FILE
)


print(
    f"Train:      {X_train.shape}"
)

print(
    f"Validation: {X_val.shape}"
)

print(
    f"Test:       {X_test.shape}"
)


classes = sorted(
    np.unique(y_train)
)


print()
print(
    f"Classes: {len(classes)}"
)

for class_name in classes:
    print(
        f"  {class_name}"
    )


# ============================================================
# Verify prompt leakage does not exist
# ============================================================

train_prompt_ids = set(
    train_ids.tolist()
)

val_prompt_ids = set(
    val_ids.tolist()
)

test_prompt_ids = set(
    test_ids.tolist()
)


assert train_prompt_ids.isdisjoint(
    val_prompt_ids
)

assert train_prompt_ids.isdisjoint(
    test_prompt_ids
)

assert val_prompt_ids.isdisjoint(
    test_prompt_ids
)


print()
print(
    "Logical-prompt leakage check: PASS"
)


# ============================================================
# Hyperparameter search
# ============================================================

print()
print("========================================")
print("      VALIDATION SEARCH")
print("========================================")
print()


best_model = None
best_score = -1
best_components = None
best_alpha = None


results = []


for n_components in PCA_COMPONENTS:

    for alpha in ALPHAS:

        model = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),

                (
                    "pca",
                    PCA(
                        n_components=n_components,
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


        model.fit(
            X_train,
            y_train,
        )


        val_predictions = model.predict(
            X_val
        )


        val_accuracy = accuracy_score(
            y_val,
            val_predictions,
        )


        results.append(
            (
                n_components,
                alpha,
                val_accuracy,
            )
        )


        print(
            f"PCA={n_components:2} "
            f"alpha={alpha:6.3f} "
            f"validation="
            f"{val_accuracy * 100:6.2f}%"
        )


        if val_accuracy > best_score:

            best_score = val_accuracy

            best_components = (
                n_components
            )

            best_alpha = alpha

            best_model = model


# ============================================================
# Best validation model
# ============================================================

print()
print("========================================")
print("       BEST CONFIGURATION")
print("========================================")
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
    f"Validation:     "
    f"{best_score * 100:.2f}%"
)


# ============================================================
# Refit on TRAIN + VALIDATION
#
# Test data remains untouched.
# ============================================================

print()
print(
    "Refitting best configuration "
    "on train + validation..."
)


X_train_final = np.concatenate(
    [
        X_train,
        X_val,
    ],
    axis=0,
)

y_train_final = np.concatenate(
    [
        y_train,
        y_val,
    ],
    axis=0,
)


final_model = Pipeline(
    [
        (
            "scale",
            StandardScaler(),
        ),

        (
            "pca",
            PCA(
                n_components=best_components,
                random_state=42,
            ),
        ),

        (
            "ridge",
            RidgeClassifier(
                alpha=best_alpha,
            ),
        ),
    ]
)


final_model.fit(
    X_train_final,
    y_train_final,
)


# ============================================================
# FINAL TEST
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
)


chance_accuracy = (
    1.0 / len(classes)
)


# ============================================================
# Results
# ============================================================

print()
print()
print("========================================")
print("        PyFly FINAL TEST RESULT")
print("========================================")
print()

print(
    f"Chance accuracy: "
    f"{chance_accuracy * 100:.2f}%"
)

print(
    f"Validation accuracy: "
    f"{best_score * 100:.2f}%"
)

print(
    f"TEST accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"TEST macro F1: "
    f"{test_macro_f1 * 100:.2f}%"
)


# ============================================================
# Individual predictions
# ============================================================

print()
print("Test predictions:")
print()


for example_id, seed, true, predicted in zip(
    test_ids,
    test_seeds,
    y_test,
    test_predictions,
):

    mark = (
        "✓"
        if true == predicted
        else "✗"
    )


    print(
        f"{mark} "
        f"{example_id:22} "
        f"seed={seed} "
        f"{true:15} -> "
        f"{predicted}"
    )


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=classes,
)


print()
print("Classes:")

print(classes)

print()
print("Confusion matrix:")

print(cm)


print()
print("Classification report:")
print()

print(
    classification_report(
        y_test,
        test_predictions,
        labels=classes,
        zero_division=0,
    )
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
    "chance_accuracy": float(
        chance_accuracy
    ),

    "validation_accuracy": float(
        best_score
    ),

    "test_accuracy": float(
        test_accuracy
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

    "classes": classes,

    "train_episodes": int(
        len(X_train)
    ),

    "validation_episodes": int(
        len(X_val)
    ),

    "test_episodes": int(
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
    "✅ Final template experiment completed"
)