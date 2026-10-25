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

TRAIN_FILE = "artifacts/features/sequence_train.npz"
VAL_FILE = "artifacts/features/sequence_validation.npz"
TEST_FILE = "artifacts/features/sequence_test.npz"

OUTPUT_FILE = (
    "artifacts/metrics/"
    "sequence_model_comparison.json"
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

MAX_TOKENS = 6


# ============================================================
# Load logical dataset
# ============================================================

examples = load_dataset(
    DATASET_FILE
)

example_by_id = {
    example.id: example
    for example in examples
}


vocabulary = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


token_to_index = {
    token: index
    for index, token in enumerate(vocabulary)
}


print()
print("============================================")
print("     PyFly SEQUENCE MODEL COMPARISON")
print("============================================")
print()

print(
    f"Vocabulary size: {len(vocabulary)}"
)

print(
    f"Logical prompts: {len(examples)}"
)


# ============================================================
# Load reservoir features
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


X_brain_train, y_train, train_ids, train_seeds = load_split(
    TRAIN_FILE
)

X_brain_val, y_val, val_ids, val_seeds = load_split(
    VAL_FILE
)

X_brain_test, y_test, test_ids, test_seeds = load_split(
    TEST_FILE
)


classes = sorted(
    np.unique(y_train)
)


print()
print(f"Classes: {len(classes)}")

for class_name in classes:
    print(
        f"  {class_name}"
    )


print()
print(
    f"Reservoir train: {X_brain_train.shape}"
)

print(
    f"Reservoir val:   {X_brain_val.shape}"
)

print(
    f"Reservoir test:  {X_brain_test.shape}"
)


# ============================================================
# Leakage check
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
# Encoder 1:
# BAG OF WORDS
#
# Ignores order completely.
# ============================================================

def encode_bow(example_id):

    example = example_by_id[
        example_id
    ]

    vector = np.zeros(
        len(vocabulary),
        dtype=np.float32,
    )

    for token in example.prompt_tokens:

        vector[
            token_to_index[token]
        ] += 1.0

    return vector


# ============================================================
# Encoder 2:
# POSITION-AWARE ONE-HOT
#
# Keeps token identity AND position.
#
# Example:
#
# position 0 = subtract
# position 1 = b
# position 2 = from
# position 3 = a
#
# This has NO FlyBrain.
# ============================================================

def encode_position(example_id):

    example = example_by_id[
        example_id
    ]

    matrix = np.zeros(
        (
            MAX_TOKENS,
            len(vocabulary),
        ),
        dtype=np.float32,
    )

    for position, token in enumerate(
        example.prompt_tokens[
            :MAX_TOKENS
        ]
    ):

        matrix[
            position,
            token_to_index[token],
        ] = 1.0

    return matrix.reshape(-1)


# ============================================================
# Generate baseline feature matrices
# ============================================================

def build_features(
    ids,
    encoder,
):

    return np.stack([
        encoder(example_id)
        for example_id in ids
    ])


X_bow_train = build_features(
    train_ids,
    encode_bow,
)

X_bow_val = build_features(
    val_ids,
    encode_bow,
)

X_bow_test = build_features(
    test_ids,
    encode_bow,
)


X_pos_train = build_features(
    train_ids,
    encode_position,
)

X_pos_val = build_features(
    val_ids,
    encode_position,
)

X_pos_test = build_features(
    test_ids,
    encode_position,
)


print()
print("============================================")
print("           BASELINE DIMENSIONS")
print("============================================")
print()

print(
    f"Bag-of-words:   "
    f"{X_bow_train.shape[1]} features"
)

print(
    f"Position-aware: "
    f"{X_pos_train.shape[1]} features"
)

print(
    f"MaleCNS:        "
    f"{X_brain_train.shape[1]:,} features"
)


# ============================================================
# Tune simple Ridge baseline
# ============================================================

def tune_ridge(
    name,
    X_train,
    X_val,
):

    print()
    print(
        "============================================"
    )

    print(
        f" TUNING {name}"
    )

    print(
        "============================================"
    )

    print()


    best_alpha = None
    best_accuracy = -1


    for alpha in RIDGE_ALPHAS:

        model = RidgeClassifier(
            alpha=alpha
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
            f"alpha={alpha:7.3f} "
            f"validation="
            f"{accuracy * 100:6.2f}%"
        )


        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_alpha = alpha


    print()
    print(
        f"Best alpha: {best_alpha}"
    )

    print(
        f"Best validation: "
        f"{best_accuracy * 100:.2f}%"
    )


    return (
        best_alpha,
        best_accuracy,
    )


# ============================================================
# Bag-of-words tuning
# ============================================================

bow_alpha, bow_val_accuracy = tune_ridge(
    "BAG OF WORDS",
    X_bow_train,
    X_bow_val,
)


# ============================================================
# Position-aware tuning
# ============================================================

pos_alpha, pos_val_accuracy = tune_ridge(
    "POSITION-AWARE LINEAR",
    X_pos_train,
    X_pos_val,
)


# ============================================================
# MaleCNS tuning
# ============================================================

print()
print("============================================")
print("          TUNING MaleCNS RESERVOIR")
print("============================================")
print()


brain_best_accuracy = -1
brain_best_components = None
brain_best_alpha = None


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
            X_brain_train,
            y_train,
        )


        predictions = model.predict(
            X_brain_val
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


        if accuracy > brain_best_accuracy:

            brain_best_accuracy = accuracy

            brain_best_components = (
                components
            )

            brain_best_alpha = alpha


print()
print(
    f"Best PCA components: "
    f"{brain_best_components}"
)

print(
    f"Best alpha: "
    f"{brain_best_alpha}"
)

print(
    f"Best validation: "
    f"{brain_best_accuracy * 100:.2f}%"
)


# ============================================================
# Final training:
# TRAIN + VALIDATION
# ============================================================

y_final = np.concatenate(
    [
        y_train,
        y_val,
    ]
)


# ------------------------------------------------------------
# Bag of words
# ------------------------------------------------------------

X_bow_final = np.concatenate(
    [
        X_bow_train,
        X_bow_val,
    ],
    axis=0,
)


bow_model = RidgeClassifier(
    alpha=bow_alpha
)


bow_model.fit(
    X_bow_final,
    y_final,
)


bow_predictions = bow_model.predict(
    X_bow_test
)


# ------------------------------------------------------------
# Position-aware
# ------------------------------------------------------------

X_pos_final = np.concatenate(
    [
        X_pos_train,
        X_pos_val,
    ],
    axis=0,
)


pos_model = RidgeClassifier(
    alpha=pos_alpha
)


pos_model.fit(
    X_pos_final,
    y_final,
)


pos_predictions = pos_model.predict(
    X_pos_test
)


# ------------------------------------------------------------
# MaleCNS
# ------------------------------------------------------------

X_brain_final = np.concatenate(
    [
        X_brain_train,
        X_brain_val,
    ],
    axis=0,
)


brain_model = Pipeline(
    [
        (
            "scale",
            StandardScaler(),
        ),

        (
            "pca",
            PCA(
                n_components=brain_best_components,
                svd_solver="randomized",
                random_state=42,
            ),
        ),

        (
            "ridge",
            RidgeClassifier(
                alpha=brain_best_alpha,
            ),
        ),
    ]
)


brain_model.fit(
    X_brain_final,
    y_final,
)


brain_predictions = brain_model.predict(
    X_brain_test
)


# ============================================================
# Metrics
# ============================================================

def metrics(
    true,
    predicted,
):

    return {
        "accuracy": float(
            accuracy_score(
                true,
                predicted,
            )
        ),

        "macro_f1": float(
            f1_score(
                true,
                predicted,
                average="macro",
            )
        ),
    }


bow_metrics = metrics(
    y_test,
    bow_predictions,
)

pos_metrics = metrics(
    y_test,
    pos_predictions,
)

brain_metrics = metrics(
    y_test,
    brain_predictions,
)


chance = (
    1 / len(classes)
)


# ============================================================
# Final comparison
# ============================================================

print()
print()
print("============================================")
print("      PyFly SEQUENCE CHALLENGE RESULT")
print("============================================")
print()


print(
    f"Random chance:"
    f"             "
    f"{chance * 100:6.2f}%"
)

print(
    f"Bag-of-words:"
    f"              "
    f"{bow_metrics['accuracy'] * 100:6.2f}%"
)

print(
    f"Position-aware linear:"
    f"      "
    f"{pos_metrics['accuracy'] * 100:6.2f}%"
)

print(
    f"MaleCNS FlyBrain:"
    f"          "
    f"{brain_metrics['accuracy'] * 100:6.2f}%"
)


print()
print("Macro F1:")

print(
    f"Bag-of-words:          "
    f"{bow_metrics['macro_f1'] * 100:.2f}%"
)

print(
    f"Position-aware linear: "
    f"{pos_metrics['macro_f1'] * 100:.2f}%"
)

print(
    f"MaleCNS FlyBrain:      "
    f"{brain_metrics['macro_f1'] * 100:.2f}%"
)


# ============================================================
# MaleCNS predictions
# ============================================================

print()
print("MaleCNS test predictions:")
print()


for example_id, seed, true, predicted in zip(
    test_ids,
    test_seeds,
    y_test,
    brain_predictions,
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
# MaleCNS confusion matrix
# ============================================================

brain_cm = confusion_matrix(
    y_test,
    brain_predictions,
    labels=classes,
)


print()
print("Classes:")

print(classes)

print()
print("MaleCNS confusion matrix:")

print(brain_cm)


# ============================================================
# Save results
# ============================================================

results = {
    "chance_accuracy": float(
        chance
    ),

    "bag_of_words": {
        **bow_metrics,
        "best_alpha": float(
            bow_alpha
        ),
        "validation_accuracy": float(
            bow_val_accuracy
        ),
    },

    "position_aware": {
        **pos_metrics,
        "best_alpha": float(
            pos_alpha
        ),
        "validation_accuracy": float(
            pos_val_accuracy
        ),
    },

    "male_cns": {
        **brain_metrics,

        "pca_components": int(
            brain_best_components
        ),

        "best_alpha": float(
            brain_best_alpha
        ),

        "validation_accuracy": float(
            brain_best_accuracy
        ),
    },
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
    "✅ Sequence model comparison completed"
)