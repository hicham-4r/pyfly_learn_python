import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_FILE = "artifacts/features/token_visual_features.npz"


print("Loading FlyBrain features...")

data = np.load(
    FEATURE_FILE,
    allow_pickle=True,
)

X = data["X"]
labels = data["labels"]
seeds = data["seeds"]


print()
print("=== Dataset ===")
print("Episodes:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", len(np.unique(labels)))
print("Seeds:", sorted(np.unique(seeds)))


# -------------------------------------------------
# Leave-one-neural-seed-out evaluation
# -------------------------------------------------

all_true = []
all_pred = []

seed_accuracies = []


for test_seed in sorted(np.unique(seeds)):

    print()
    print(
        f"========== TEST SEED {test_seed} =========="
    )

    train_mask = seeds != test_seed
    test_mask = seeds == test_seed

    X_train = X[train_mask]
    y_train = labels[train_mask]

    X_test = X[test_mask]
    y_test = labels[test_mask]


    print(
        f"Training episodes: {len(X_train)}"
    )

    print(
        f"Testing episodes:  {len(X_test)}"
    )


    # ---------------------------------------------
    # PCA + Ridge readout
    # ---------------------------------------------

    model = Pipeline(
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


    seed_accuracies.append(
        accuracy
    )

    all_true.extend(
        y_test.tolist()
    )

    all_pred.extend(
        predictions.tolist()
    )


    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


    print()
    print("Predictions:")

    for true, predicted in zip(
        y_test,
        predictions,
    ):

        result = (
            "✓"
            if true == predicted
            else "✗"
        )

        print(
            f"{result} "
            f"{true:12} -> "
            f"{predicted}"
        )


# -------------------------------------------------
# Overall results
# -------------------------------------------------

all_true = np.asarray(all_true)
all_pred = np.asarray(all_pred)

overall_accuracy = accuracy_score(
    all_true,
    all_pred,
)

mean_accuracy = np.mean(
    seed_accuracies
)

std_accuracy = np.std(
    seed_accuracies
)


classes = sorted(
    np.unique(labels)
)

chance = 1 / len(classes)


print()
print()
print("================================")
print("     PyFly TOKEN DECODER")
print("================================")

print(
    f"Classes:          {len(classes)}"
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
    f"Std across seeds: "
    f"{std_accuracy * 100:.2f}%"
)


print()
print("Per-seed accuracy:")

for seed, accuracy in zip(
    sorted(np.unique(seeds)),
    seed_accuracies,
):

    print(
        f"Seed {seed}: "
        f"{accuracy * 100:.2f}%"
    )


# -------------------------------------------------
# Confusion matrix
# -------------------------------------------------

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


# -------------------------------------------------
# Save predictions
# -------------------------------------------------

np.savez(
    "artifacts/metrics/token_decoder_results.npz",

    true_labels=all_true,
    predicted_labels=all_pred,

    seed_accuracies=np.asarray(
        seed_accuracies
    ),

    overall_accuracy=overall_accuracy,

    classes=np.asarray(classes),
)


print()
print(
    "Saved results:"
    " artifacts/metrics/"
    "token_decoder_results.npz"
)

print()
print("✅ Token decoder experiment completed")