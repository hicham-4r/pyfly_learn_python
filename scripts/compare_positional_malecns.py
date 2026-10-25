import json

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Files
# ============================================================

TRAIN_FILE = (
    "artifacts/features/"
    "sequence_positional_train.npz"
)

VALIDATION_FILE = (
    "artifacts/features/"
    "sequence_positional_validation.npz"
)

TEST_FILE = (
    "artifacts/features/"
    "sequence_positional_test.npz"
)

OUTPUT_FILE = (
    "artifacts/metrics/"
    "sequence_positional_malecns.json"
)


# Previous Sequence Challenge comparison.
#
# This should contain:
# bag-of-words
# position-aware linear
# old MaleCNS
#
PREVIOUS_RESULTS = (
    "artifacts/metrics/"
    "sequence_model_comparison.json"
)


# ============================================================
# Hyperparameter search
# ============================================================

PCA_COMPONENTS = [
    5,
    10,
    20,
    40,
    60,
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
print("============================================")
print("   PyFly POSITIONAL MaleCNS COMPARISON")
print("============================================")
print()


X_train, y_train, train_ids, train_seeds = (
    load_split(
        TRAIN_FILE
    )
)

X_val, y_val, val_ids, val_seeds = (
    load_split(
        VALIDATION_FILE
    )
)

X_test, y_test, test_ids, test_seeds = (
    load_split(
        TEST_FILE
    )
)


classes = sorted(
    np.unique(
        y_train
    )
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

print()

print(
    f"Classes: {len(classes)}"
)

for class_name in classes:

    print(
        f"  {class_name}"
    )


# ============================================================
# Leakage check
# ============================================================

train_prompt_ids = set(
    train_ids.tolist()
)

validation_prompt_ids = set(
    val_ids.tolist()
)

test_prompt_ids = set(
    test_ids.tolist()
)


assert train_prompt_ids.isdisjoint(
    validation_prompt_ids
)

assert train_prompt_ids.isdisjoint(
    test_prompt_ids
)

assert validation_prompt_ids.isdisjoint(
    test_prompt_ids
)


print()
print(
    "Logical-prompt leakage check: PASS"
)


# ============================================================
# Validation search
# ============================================================

print()
print("============================================")
print("          VALIDATION SEARCH")
print("============================================")
print()


best_accuracy = -1.0

best_components = None
best_alpha = None

search_results = []


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


        search_results.append(
            {
                "components": components,
                "alpha": alpha,
                "accuracy": accuracy,
            }
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
# Best model configuration
# ============================================================

print()
print("============================================")
print("        BEST POSITIONAL CONFIGURATION")
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
    X_final_train,
    y_final_train,
)


# ============================================================
# Test
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


chance = (
    1.0
    / len(classes)
)


# ============================================================
# Main result
# ============================================================

print()
print()
print("============================================")
print(" POSITIONAL MaleCNS SEQUENCE RESULT")
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
    f"{test_macro_f1 * 100:.2f}%"
)


# ============================================================
# Predictions
# ============================================================

print()
print("Test predictions:")
print()


for (
    example_id,
    seed,
    true,
    predicted,
) in zip(
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
    test_predictions,
    labels=classes,
)


print()
print("Classes:")

print(
    classes
)

print()
print(
    "Confusion matrix:"
)

print(
    cm
)


print()
print(
    "Classification report:"
)

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
# Load previous comparison results
# ============================================================

previous = None


try:

    with open(
        PREVIOUS_RESULTS,
        "r",
        encoding="utf-8",
    ) as file:

        previous = json.load(
            file
        )

except FileNotFoundError:

    print()
    print(
        "Previous comparison JSON not found."
    )

    print(
        "The new positional MaleCNS result "
        "will still be saved."
    )


# ============================================================
# Final comparison table
# ============================================================

print()
print()
print("============================================")
print("       PyFly FINAL SEQUENCE COMPARISON")
print("============================================")
print()


print(
    f"Random chance:                "
    f"{chance * 100:6.2f}%"
)


if previous is not None:

    bow = (
        previous[
            "bag_of_words"
        ][
            "accuracy"
        ]
    )

    positional_linear = (
        previous[
            "position_aware"
        ][
            "accuracy"
        ]
    )

    old_malecns = (
        previous[
            "male_cns"
        ][
            "accuracy"
        ]
    )


    print(
        f"Bag-of-words linear:          "
        f"{bow * 100:6.2f}%"
    )

    print(
        f"Position-aware linear:        "
        f"{positional_linear * 100:6.2f}%"
    )

    print(
        f"Old MaleCNS encoder:          "
        f"{old_malecns * 100:6.2f}%"
    )


print(
    f"Position-aware MaleCNS:       "
    f"{test_accuracy * 100:6.2f}%"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("============================================")
print("             INTERPRETATION")
print("============================================")
print()


if previous is not None:

    positional_linear = (
        previous[
            "position_aware"
        ][
            "accuracy"
        ]
    )


    if (
        test_accuracy
        >
        positional_linear
    ):

        print(
            "✅ Position-aware MaleCNS "
            "outperformed the conventional "
            "position-aware linear encoder."
        )

    elif (
        test_accuracy
        ==
        positional_linear
    ):

        print(
            "⚠️ Position-aware MaleCNS "
            "tied the position-aware "
            "linear baseline."
        )

    else:

        print(
            "⚠️ Position-aware linear "
            "still outperforms MaleCNS."
        )


if (
    test_accuracy
    > chance
):

    print(
        "✅ Positional neural encoding "
        "produced above-chance "
        "Sequence Challenge performance."
    )

else:

    print(
        "❌ Positional neural encoding "
        "did not beat random chance."
    )


# ============================================================
# Save results
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

    "readout_type": (
        "indirect_visual"
    ),

    "encoder_type": (
        "token_position_population"
    ),
}


if previous is not None:

    results[
        "previous_comparison"
    ] = {
        "bag_of_words": float(
            previous[
                "bag_of_words"
            ][
                "accuracy"
            ]
        ),

        "position_aware_linear": float(
            previous[
                "position_aware"
            ][
                "accuracy"
            ]
        ),

        "old_malecns": float(
            previous[
                "male_cns"
            ][
                "accuracy"
            ]
        ),
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
    "✅ Position-aware MaleCNS comparison completed"
)