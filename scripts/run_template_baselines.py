import json
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyfly.dataset import load_dataset


DATASET_FILE = "data/raw/python_mre.jsonl"

TRAIN_FILE = "artifacts/features/template_train.npz"
VAL_FILE = "artifacts/features/template_validation.npz"
TEST_FILE = "artifacts/features/template_test.npz"

FINAL_METRICS = "artifacts/metrics/template_final_metrics.json"


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


# Use the complete fixed vocabulary.
# This uses token identity but NO target labels.
vocabulary = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})

token_to_index = {
    token: index
    for index, token in enumerate(vocabulary)
}


def encode_prompt(example_id):
    """
    Simple bag-of-words / encoder-only representation.
    No FlyBrain involved.
    """

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
# Load reservoir feature files
# ============================================================

def load_features(path):

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


X_brain_train, y_train, train_ids, train_seeds = load_features(
    TRAIN_FILE
)

X_brain_val, y_val, val_ids, val_seeds = load_features(
    VAL_FILE
)

X_brain_test, y_test, test_ids, test_seeds = load_features(
    TEST_FILE
)


print()
print("========================================")
print("       PyFly BASELINE COMPARISON")
print("========================================")
print()

print(
    f"Vocabulary size: {len(vocabulary)}"
)

print(
    f"Train episodes: {len(train_ids)}"
)

print(
    f"Validation episodes: {len(val_ids)}"
)

print(
    f"Test episodes: {len(test_ids)}"
)


# ============================================================
# BASELINE 1
# Encoder-only bag-of-words
# ============================================================

print()
print("========================================")
print("       ENCODER-ONLY BASELINE")
print("========================================")
print()


X_encoder_train = np.stack([
    encode_prompt(example_id)
    for example_id in train_ids
])

X_encoder_val = np.stack([
    encode_prompt(example_id)
    for example_id in val_ids
])

X_encoder_test = np.stack([
    encode_prompt(example_id)
    for example_id in test_ids
])


alphas = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


best_encoder_alpha = None
best_encoder_val = -1


for alpha in alphas:

    model = RidgeClassifier(
        alpha=alpha
    )

    model.fit(
        X_encoder_train,
        y_train,
    )

    pred = model.predict(
        X_encoder_val
    )

    accuracy = accuracy_score(
        y_val,
        pred,
    )

    print(
        f"alpha={alpha:7.3f} "
        f"validation={accuracy * 100:6.2f}%"
    )

    if accuracy > best_encoder_val:

        best_encoder_val = accuracy
        best_encoder_alpha = alpha


# Refit encoder-only model on train + validation

X_encoder_train_final = np.concatenate(
    [
        X_encoder_train,
        X_encoder_val,
    ],
    axis=0,
)

y_encoder_train_final = np.concatenate(
    [
        y_train,
        y_val,
    ]
)


encoder_model = RidgeClassifier(
    alpha=best_encoder_alpha
)

encoder_model.fit(
    X_encoder_train_final,
    y_encoder_train_final,
)


encoder_predictions = encoder_model.predict(
    X_encoder_test
)

encoder_test_accuracy = accuracy_score(
    y_test,
    encoder_predictions,
)


print()
print(
    f"Best alpha: "
    f"{best_encoder_alpha}"
)

print(
    f"Best validation accuracy: "
    f"{best_encoder_val * 100:.2f}%"
)

print(
    f"ENCODER-ONLY TEST accuracy: "
    f"{encoder_test_accuracy * 100:.2f}%"
)


# ============================================================
# BASELINE 2
# Shuffled-label FlyBrain control
# ============================================================

print()
print("========================================")
print("       SHUFFLED-LABEL CONTROL")
print("========================================")
print()


with open(
    FINAL_METRICS,
    "r",
    encoding="utf-8",
) as file:

    final_metrics = json.load(
        file
    )


brain_components = int(
    final_metrics[
        "pca_components"
    ]
)

brain_alpha = float(
    final_metrics[
        "ridge_alpha"
    ]
)


print(
    f"Using PCA components: "
    f"{brain_components}"
)

print(
    f"Using Ridge alpha: "
    f"{brain_alpha}"
)


X_brain_train_final = np.concatenate(
    [
        X_brain_train,
        X_brain_val,
    ],
    axis=0,
)

y_brain_train_final = np.concatenate(
    [
        y_train,
        y_val,
    ]
)


shuffle_accuracies = []


for shuffle_seed in range(10):

    rng = np.random.default_rng(
        shuffle_seed
    )

    shuffled_labels = (
        y_brain_train_final.copy()
    )

    rng.shuffle(
        shuffled_labels
    )


    shuffled_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),

            (
                "pca",
                PCA(
                    n_components=brain_components,
                    random_state=42,
                ),
            ),

            (
                "ridge",
                RidgeClassifier(
                    alpha=brain_alpha,
                ),
            ),
        ]
    )


    shuffled_model.fit(
        X_brain_train_final,
        shuffled_labels,
    )


    predictions = shuffled_model.predict(
        X_brain_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    shuffle_accuracies.append(
        accuracy
    )


    print(
        f"Shuffle seed {shuffle_seed}: "
        f"{accuracy * 100:.2f}%"
    )


shuffle_mean = float(
    np.mean(
        shuffle_accuracies
    )
)

shuffle_std = float(
    np.std(
        shuffle_accuracies
    )
)


# ============================================================
# Summary
# ============================================================

chance = (
    1 / len(
        np.unique(y_test)
    )
)


brain_test_accuracy = float(
    final_metrics[
        "test_accuracy"
    ]
)


print()
print()
print("========================================")
print("       PyFly BASELINE SUMMARY")
print("========================================")
print()

print(
    f"Random chance:            "
    f"{chance * 100:.2f}%"
)

print(
    f"Shuffled-label FlyBrain:  "
    f"{shuffle_mean * 100:.2f}% "
    f"± {shuffle_std * 100:.2f}%"
)

print(
    f"Encoder-only linear:      "
    f"{encoder_test_accuracy * 100:.2f}%"
)

print(
    f"MaleCNS FlyBrain:         "
    f"{brain_test_accuracy * 100:.2f}%"
)


print()
print("========================================")

if brain_test_accuracy > encoder_test_accuracy:

    print(
        "FlyBrain is above the encoder-only "
        "baseline on this test."
    )

elif brain_test_accuracy < encoder_test_accuracy:

    print(
        "Encoder-only baseline currently "
        "outperforms FlyBrain."
    )

else:

    print(
        "FlyBrain and encoder-only baseline "
        "are tied."
    )

print("========================================")

print()
print(
    "✅ Baseline experiment completed"
)