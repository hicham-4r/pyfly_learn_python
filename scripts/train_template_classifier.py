import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_FILE = (
    "artifacts/features/"
    "template_prompt_features.npz"
)


print("Loading complete-prompt features...")

data = np.load(
    FEATURE_FILE,
    allow_pickle=True,
)


X = data["X"]

intents = data[
    "intents"
].astype(str)

example_ids = data[
    "example_ids"
].astype(str)

seeds = data[
    "seeds"
]


classes = sorted(
    np.unique(intents)
)


print()
print("========================================")
print("          PyFly TEMPLATE TASK")
print("========================================")
print()

print(f"Episodes: {len(X)}")
print(f"Features: {X.shape[1]:,}")
print(f"Classes:  {len(classes)}")

print()

for class_name in classes:
    print(f"  {class_name}")


# ============================================================
# Model
# ============================================================

def create_model():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),

            (
                "pca",
                PCA(
                    n_components=20,
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
# Two-fold paraphrase evaluation
#
# Fold A:
# train *_01
# test  *_02
#
# Fold B:
# train *_02
# test  *_01
# ============================================================

folds = [
    ("01 -> 02", "_01", "_02"),
    ("02 -> 01", "_02", "_01"),
]


all_true = []
all_pred = []

fold_accuracies = []


for fold_name, train_suffix, test_suffix in folds:

    print()
    print(
        "========================================"
    )

    print(
        f"     FOLD {fold_name}"
    )

    print(
        "========================================"
    )


    train_mask = np.asarray([
        example_id.endswith(
            train_suffix
        )
        for example_id in example_ids
    ])


    test_mask = np.asarray([
        example_id.endswith(
            test_suffix
        )
        for example_id in example_ids
    ])


    X_train = X[
        train_mask
    ]

    y_train = intents[
        train_mask
    ]


    X_test = X[
        test_mask
    ]

    y_test = intents[
        test_mask
    ]


    test_ids = example_ids[
        test_mask
    ]


    print(
        f"Training episodes: "
        f"{len(X_train)}"
    )

    print(
        f"Testing episodes:  "
        f"{len(X_test)}"
    )


    model = create_model()


    model.fit(
        X_train,
        y_train,
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    fold_accuracies.append(
        accuracy
    )


    all_true.extend(
        y_test.tolist()
    )

    all_pred.extend(
        predictions.tolist()
    )


    print()
    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    print()
    print("Predictions:")


    for example_id, true, predicted in zip(
        test_ids,
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
            f"{example_id:20} "
            f"{true:15} -> "
            f"{predicted}"
        )


# ============================================================
# Overall results
# ============================================================

all_true = np.asarray(
    all_true
)

all_pred = np.asarray(
    all_pred
)


overall_accuracy = accuracy_score(
    all_true,
    all_pred,
)


mean_accuracy = np.mean(
    fold_accuracies
)

std_accuracy = np.std(
    fold_accuracies
)


chance = (
    1 / len(classes)
)


print()
print()
print("========================================")
print("       PyFly TEMPLATE CLASSIFIER")
print("========================================")
print()

print(
    f"Classes:          "
    f"{len(classes)}"
)

print(
    f"Chance accuracy:  "
    f"{chance * 100:.2f}%"
)

print(
    f"Overall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Mean accuracy:    "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Std across folds: "
    f"{std_accuracy * 100:.2f}%"
)


print()
print("Fold accuracies:")

for fold, accuracy in zip(
    folds,
    fold_accuracies,
):

    print(
        f"{fold[0]}: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,
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

OUTPUT = (
    "artifacts/metrics/"
    "template_classifier_results.npz"
)


np.savez(
    OUTPUT,

    true_labels=all_true,

    predicted_labels=all_pred,

    fold_accuracies=np.asarray(
        fold_accuracies
    ),

    overall_accuracy=overall_accuracy,

    classes=np.asarray(classes),
)


print()
print(
    f"Saved results: {OUTPUT}"
)

print()
print(
    "✅ Template classification completed"
)