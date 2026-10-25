import json

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyfly.dataset import load_dataset


# ============================================================
# Files
# ============================================================

DATASET_FILE = "data/raw/sequence_challenge.jsonl"

TRAIN_FILE = "artifacts/features/sequence_random_train.npz"
VAL_FILE = "artifacts/features/sequence_random_validation.npz"
TEST_FILE = "artifacts/features/sequence_random_test.npz"

OUTPUT_FILE = (
    "artifacts/metrics/"
    "sequence_random_readout_results.json"
)


# ============================================================
# Hyperparameters
# ============================================================

RIDGE_ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

PCA_COMPONENTS = [
    5,
    10,
    20,
    40,
    60,
]


# ============================================================
# Load logical dataset
# ============================================================

examples = load_dataset(
    DATASET_FILE
)

print()
print("============================================")
print(" PyFly RANDOM WHOLE-BRAIN READOUT TEST")
print("============================================")
print()

print(
    f"Logical prompts: {len(examples)}"
)


# ============================================================
# Load feature splits
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


X_train, y_train, train_ids, train_seeds = load_split(
    TRAIN_FILE
)

X_val, y_val, val_ids, val_seeds = load_split(
    VAL_FILE
)

X_test, y_test, test_ids, test_seeds = load_split(
    TEST_FILE
)


classes = sorted(
    np.unique(y_train)
)


print()
print(
    f"Train:      {X_train.shape}"
)

print(
    f"Validation: {X_val.shape}"
)

print(
    f"Test:       {X_test.shape}"
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
# Leakage checks
# ============================================================

assert set(
    train_ids
).isdisjoint(
    set(val_ids)
)

assert set(
    train_ids
).isdisjoint(
    set(test_ids)
)

assert set(
    val_ids
).isdisjoint(
    set(test_ids)
)


print()
print(
    "Logical-prompt leakage check: PASS"
)


# ============================================================
# Hyperparameter search
# ============================================================

print()
print("============================================")
print("       VALIDATION SEARCH")
print("============================================")
print()


best_accuracy = -1
best_components = None
best_alpha = None


for components in PCA_COMPONENTS:

    for alpha in RIDGE_ALPHAS:

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


        print(
            f"PCA={components:2} "
            f"alpha={alpha:7.3f} "
            f"validation="
            f"{accuracy * 100:6.2f}%"
        )


        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_components = (
                components
            )

            best_alpha = alpha


# ============================================================
# Best configuration
# ============================================================

print()
print("============================================")
print("        BEST CONFIGURATION")
print("============================================")
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
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# Refit on train + validation
# ============================================================

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
                svd_solver="randomized",
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
    X_final,
    y_final,
)


# ============================================================
# Test
# ============================================================

predictions = final_model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    predictions,
)


macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)


chance = (
    1 / len(classes)
)


# ============================================================
# Results
# ============================================================

print()
print()
print("============================================")
print(" RANDOM WHOLE-BRAIN SEQUENCE RESULT")
print("============================================")
print()

print(
    f"Random chance:       "
    f"{chance * 100:.2f}%"
)

print(
    f"Validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"TEST accuracy:       "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"TEST macro F1:       "
    f"{macro_f1 * 100:.2f}%"
)


# ============================================================
# Predictions
# ============================================================

print()
print("Test predictions:")
print()


for example_id, seed, true, predicted in zip(
    test_ids,
    test_seeds,
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
        f"{example_id:16} "
        f"seed={seed} "
        f"{true:10} -> "
        f"{predicted}"
    )


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=classes,
)


print()
print("Classes:")
print(classes)

print()
print("Confusion matrix:")
print(cm)


# ============================================================
# Save
# ============================================================

results = {
    "chance_accuracy": float(
        chance
    ),

    "validation_accuracy": float(
        best_accuracy
    ),

    "test_accuracy": float(
        test_accuracy
    ),

    "macro_f1": float(
        macro_f1
    ),

    "pca_components": int(
        best_components
    ),

    "ridge_alpha": float(
        best_alpha
    ),

    "readout_type": (
        "random_whole_brain"
    ),

    "readout_neurons": 2000,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=2,
    )


print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()

print(
    "✅ Random whole-brain readout test completed"
)