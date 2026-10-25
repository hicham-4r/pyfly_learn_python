from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
# Step 76B-2
# Compare six linear decoder families on the NEW rich features.
#
# Model selection uses TRAIN + VALIDATION ONLY.
# The test set is evaluated only after the winning validation
# configuration has been selected.
# ============================================================

TRAIN_FILE = Path(
    "artifacts/features/generator_rich_train.npz"
)

VALIDATION_FILE = Path(
    "artifacts/features/generator_rich_validation.npz"
)

TEST_FILE = Path(
    "artifacts/features/generator_rich_test.npz"
)


# Previous best Step-62 global decoder.
OLD_METRICS_FILE = Path(
    "artifacts/metrics/generator_decoder_comparison.json"
)


OUTPUT_MODEL = Path(
    "artifacts/models/generator_rich_best_decoder.joblib"
)

OUTPUT_METRICS = Path(
    "artifacts/metrics/generator_rich_decoder_comparison.json"
)


# ============================================================
# Search space
#
# Keep the same search space used in the previous global
# decoder comparison so the result is directly comparable.
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


# Semantic positions used by the current PyFly generator
# evaluation convention.
SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# Current records from the handoff / previous best model.
RECORD_TOP1 = 0.7789473684210526
RECORD_SEMANTIC_ACCURACY = 0.65


# ============================================================
# Load one feature split
# ============================================================

def load_split(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    result = {
        "X": data["X"].astype(
            np.float32,
            copy=False,
        ),
        "targets": data["targets"].astype(str),
        "sample_ids": data["sample_ids"].astype(str),
        "example_ids": data["example_ids"].astype(str),
        "intents": data["intents"].astype(str),
        "positions": data["target_positions"].astype(
            np.int32,
            copy=False,
        ),
        "feedback": data["feedback_tokens"].astype(str),
        "seeds": data["seeds"].astype(
            np.int32,
            copy=False,
        ),
    }

    if "feature_block_names" in data:
        result["feature_block_names"] = (
            data["feature_block_names"].astype(str)
        )

    if "block_size" in data:
        result["block_size"] = int(
            np.asarray(data["block_size"]).item()
        )

    return result


# ============================================================
# Classifier factory
# ============================================================

def build_classifier(
    family: str,
    parameter: float,
):
    if family == "Ridge":
        return RidgeClassifier(
            alpha=parameter,
        )

    if family == "Balanced Ridge":
        return RidgeClassifier(
            alpha=parameter,
            class_weight="balanced",
        )

    if family == "Logistic":
        return LogisticRegression(
            C=parameter,
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

    if family == "Balanced Logistic":
        return LogisticRegression(
            C=parameter,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

    if family == "Linear SVM":
        return LinearSVC(
            C=parameter,
            max_iter=10000,
            random_state=42,
        )

    if family == "Balanced Linear SVM":
        return LinearSVC(
            C=parameter,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        )

    raise ValueError(
        f"Unknown decoder family: {family}"
    )


# ============================================================
# Candidate specification
# ============================================================

def build_candidates() -> list[dict]:
    candidates = []

    for components in PCA_COMPONENTS:
        for alpha in RIDGE_ALPHAS:
            candidates.append(
                {
                    "family": "Ridge",
                    "components": components,
                    "parameter": alpha,
                    "parameter_name": "alpha",
                }
            )

            candidates.append(
                {
                    "family": "Balanced Ridge",
                    "components": components,
                    "parameter": alpha,
                    "parameter_name": "alpha",
                }
            )

        for c_value in LOGISTIC_C_VALUES:
            candidates.append(
                {
                    "family": "Logistic",
                    "components": components,
                    "parameter": c_value,
                    "parameter_name": "C",
                }
            )

            candidates.append(
                {
                    "family": "Balanced Logistic",
                    "components": components,
                    "parameter": c_value,
                    "parameter_name": "C",
                }
            )

        for c_value in SVM_C_VALUES:
            candidates.append(
                {
                    "family": "Linear SVM",
                    "components": components,
                    "parameter": c_value,
                    "parameter_name": "C",
                }
            )

            candidates.append(
                {
                    "family": "Balanced Linear SVM",
                    "components": components,
                    "parameter": c_value,
                    "parameter_name": "C",
                }
            )

    for candidate in candidates:
        candidate["name"] = (
            f"{candidate['family']} "
            f"PCA={candidate['components']} "
            f"{candidate['parameter_name']}="
            f"{candidate['parameter']}"
        )

    return candidates


# ============================================================
# Ranking helper
#
# Validation model selection:
#   1. macro F1
#   2. accuracy
#   3. fewer PCA components
#
# The test split is NEVER used here.
# ============================================================

def is_better(
    candidate_result: dict,
    current_best: dict | None,
) -> bool:
    if current_best is None:
        return True

    candidate_key = (
        candidate_result["validation_macro_f1"],
        candidate_result["validation_accuracy"],
        -candidate_result["components"],
    )

    best_key = (
        current_best["validation_macro_f1"],
        current_best["validation_accuracy"],
        -current_best["components"],
    )

    return candidate_key > best_key


# ============================================================
# Top-k helper
# ============================================================

def top_k_accuracy(
    scores: np.ndarray,
    true_labels: np.ndarray,
    class_names: np.ndarray,
    k: int,
) -> float:
    k = min(
        k,
        len(class_names),
    )

    top_indices = np.argsort(
        scores,
        axis=1,
    )[:, -k:]

    correct = 0

    for row_index, target in enumerate(true_labels):
        candidates = {
            str(class_names[class_index])
            for class_index in top_indices[row_index]
        }

        if target in candidates:
            correct += 1

    return correct / len(true_labels)


# ============================================================
# Header
# ============================================================

print()
print("====================================================")
print("  PyFly STEP 76B-2 - RICH DECODER COMPARISON")
print("====================================================")
print()


# ============================================================
# Load data
# ============================================================

train = load_split(TRAIN_FILE)
validation = load_split(VALIDATION_FILE)
test = load_split(TEST_FILE)

X_train = train["X"]
y_train = train["targets"]

X_val = validation["X"]
y_val = validation["targets"]

X_test = test["X"]
y_test = test["targets"]


print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# Rich-feature structure check
# ============================================================

expected_blocks = [
    "prompt_position_0_mean",
    "prompt_position_1_mean",
    "prompt_position_2_mean",
    "prompt_position_3_mean",
    "prompt_position_4_mean",
    "prompt_position_5_mean",
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
]


if "feature_block_names" in train:
    actual_blocks = train["feature_block_names"].tolist()

    if actual_blocks != expected_blocks:
        raise RuntimeError(
            "Unexpected rich-feature block layout.\n"
            f"Expected: {expected_blocks}\n"
            f"Actual:   {actual_blocks}"
        )

    print()
    print("Rich feature block layout: PASS")
    print(f"Feature blocks: {len(actual_blocks)}")


if "block_size" in train:
    expected_dimensions = (
        len(expected_blocks)
        * train["block_size"]
    )

    if X_train.shape[1] != expected_dimensions:
        raise RuntimeError(
            "Rich feature dimension mismatch: "
            f"expected {expected_dimensions}, "
            f"got {X_train.shape[1]}"
        )

    print(f"Block size:     {train['block_size']:,}")
    print(f"Dimensions:     {expected_dimensions:,}")


# ============================================================
# Basic integrity checks
# ============================================================

if X_train.shape[1] != X_val.shape[1]:
    raise RuntimeError(
        "Train/validation feature dimensions differ."
    )

if X_train.shape[1] != X_test.shape[1]:
    raise RuntimeError(
        "Train/test feature dimensions differ."
    )

if not np.isfinite(X_train).all():
    raise RuntimeError("Non-finite values in training features.")

if not np.isfinite(X_val).all():
    raise RuntimeError("Non-finite values in validation features.")

if not np.isfinite(X_test).all():
    raise RuntimeError("Non-finite values in test features.")


# ============================================================
# Logical-prompt leakage check
# ============================================================

train_ids = set(train["example_ids"].tolist())
val_ids = set(validation["example_ids"].tolist())
test_ids = set(test["example_ids"].tolist())

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print()
print("Logical-prompt leakage check: PASS")


# ============================================================
# Classes and baselines
# ============================================================

classes = sorted(
    np.unique(y_train)
)

chance = 1.0 / len(classes)

print()
print(f"Classes:        {len(classes)}")
print(f"Uniform chance: {chance * 100:.2f}%")


unique_tokens, counts = np.unique(
    y_train,
    return_counts=True,
)

majority_token = unique_tokens[
    np.argmax(counts)
]

majority_predictions = np.full(
    len(y_test),
    majority_token,
)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions,
)

print(f"Majority token: {majority_token}")
print(
    f"Majority test accuracy: "
    f"{majority_accuracy * 100:.2f}%"
)


print()
print("Training target distribution:")
print()

for token in classes:
    count = int(
        np.sum(y_train == token)
    )

    print(
        f"{token:12} {count:4}"
    )


# ============================================================
# Preprocessing cache for validation search
#
# IMPORTANT OPTIMIZATION:
# StandardScaler is fitted ON TRAIN ONLY.
# Each PCA is fitted ON TRAIN ONLY exactly once for each
# component count and reused by every classifier at that count.
#
# This gives the same information boundary as fitting a pipeline
# candidate-by-candidate but avoids repeating the expensive PCA
# dozens of times on 76,349-dimensional features.
# ============================================================

print()
print("====================================================")
print("           PRECOMPUTING PCA REPRESENTATIONS")
print("====================================================")
print()


search_scaler = StandardScaler()

X_train_scaled = search_scaler.fit_transform(
    X_train
)

X_val_scaled = search_scaler.transform(
    X_val
)


pca_cache = {}

for components in PCA_COMPONENTS:
    if components >= min(
        X_train.shape[0],
        X_train.shape[1],
    ):
        print(
            f"Skipping PCA={components}: "
            "too many components."
        )
        continue

    print(
        f"Fitting PCA={components} ..."
    )

    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=42,
    )

    train_reduced = pca.fit_transform(
        X_train_scaled
    )

    val_reduced = pca.transform(
        X_val_scaled
    )

    pca_cache[components] = {
        "pca": pca,
        "train": train_reduced,
        "validation": val_reduced,
        "explained_variance": float(
            np.sum(
                pca.explained_variance_ratio_
            )
        ),
    }

    print(
        f"  explained variance: "
        f"{pca_cache[components]['explained_variance'] * 100:.2f}%"
    )


# Scaled full-dimensional arrays are no longer needed for search.
del X_train_scaled
del X_val_scaled


# ============================================================
# Validation search
# ============================================================

print()
print("====================================================")
print("               VALIDATION SEARCH")
print("====================================================")
print()


candidates = build_candidates()

best = None
family_best = {}
all_results = []

valid_candidate_count = sum(
    1
    for candidate in candidates
    if candidate["components"] in pca_cache
)

candidate_number = 0

for candidate in candidates:
    components = candidate["components"]

    if components not in pca_cache:
        continue

    candidate_number += 1

    X_train_reduced = pca_cache[components]["train"]
    X_val_reduced = pca_cache[components]["validation"]

    classifier = build_classifier(
        candidate["family"],
        candidate["parameter"],
    )

    classifier.fit(
        X_train_reduced,
        y_train,
    )

    predictions = classifier.predict(
        X_val_reduced
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
        "family": candidate["family"],
        "name": candidate["name"],
        "components": int(components),
        "parameter": float(candidate["parameter"]),
        "parameter_name": candidate["parameter_name"],
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(macro_f1),
        "pca_explained_variance": float(
            pca_cache[components]["explained_variance"]
        ),
    }

    all_results.append(result)

    print(
        f"[{candidate_number:03}/{valid_candidate_count:03}] "
        f"{candidate['family']:20} "
        f"PCA={components:3} "
        f"{candidate['parameter_name']}="
        f"{candidate['parameter']:8.4f} "
        f"acc={accuracy * 100:6.2f}% "
        f"F1={macro_f1 * 100:6.2f}%"
    )

    family = candidate["family"]
    current_family_best = family_best.get(family)

    if is_better(
        result,
        current_family_best,
    ):
        family_best[family] = result

    if is_better(
        result,
        best,
    ):
        best = result


if best is None:
    raise RuntimeError(
        "No valid decoder candidate was evaluated."
    )


# ============================================================
# Family summary
# ============================================================

print()
print()
print("====================================================")
print("          BEST VALIDATION MODEL BY FAMILY")
print("====================================================")
print()

family_order = [
    "Ridge",
    "Balanced Ridge",
    "Logistic",
    "Balanced Logistic",
    "Linear SVM",
    "Balanced Linear SVM",
]

for family in family_order:
    result = family_best[family]

    print(
        f"{family:22} "
        f"acc={result['validation_accuracy'] * 100:6.2f}% "
        f"F1={result['validation_macro_f1'] * 100:6.2f}% "
        f"PCA={result['components']:3} "
        f"{result['parameter_name']}="
        f"{result['parameter']}"
    )


# ============================================================
# Selected configuration
# ============================================================

print()
print("====================================================")
print("              SELECTED DECODER")
print("====================================================")
print()

print(f"Family:              {best['family']}")
print(f"Configuration:       {best['name']}")
print(
    f"Validation accuracy: "
    f"{best['validation_accuracy'] * 100:.2f}%"
)
print(
    f"Validation macro F1: "
    f"{best['validation_macro_f1'] * 100:.2f}%"
)


# ============================================================
# Refit winning model on TRAIN + VALIDATION
#
# This produces a normal sklearn Pipeline that accepts RAW
# 76,349-dimensional Step-76 features. That is important for
# later closed-loop generation.
# ============================================================

print()
print(
    "Refitting selected decoder on train + validation..."
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


winning_classifier = build_classifier(
    best["family"],
    best["parameter"],
)


winning_model = Pipeline(
    [
        (
            "scale",
            StandardScaler(),
        ),
        (
            "pca",
            PCA(
                n_components=best["components"],
                svd_solver="randomized",
                random_state=42,
            ),
        ),
        (
            "classifier",
            winning_classifier,
        ),
    ]
)


winning_model.fit(
    X_final_train,
    y_final_train,
)


# ============================================================
# TEST — used only after validation selection is complete
# ============================================================

test_predictions = winning_model.predict(
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
# Decision scores / Top-K
# ============================================================

scores = np.asarray(
    winning_model.decision_function(
        X_test
    )
)


if scores.ndim == 1:
    scores = np.column_stack(
        [
            -scores,
            scores,
        ]
    )


classifier = winning_model.named_steps[
    "classifier"
]

model_classes = np.asarray(
    classifier.classes_
).astype(str)


top2_accuracy = top_k_accuracy(
    scores,
    y_test,
    model_classes,
    2,
)


top3_accuracy = top_k_accuracy(
    scores,
    y_test,
    model_classes,
    3,
)


# ============================================================
# Accuracy by output position
# ============================================================

print()
print("====================================================")
print("             ACCURACY BY POSITION")
print("====================================================")
print()


position_metrics = {}
position_macro_f1 = {}

for position in sorted(
    np.unique(test["positions"])
):
    mask = (
        test["positions"]
        == position
    )

    position_accuracy = accuracy_score(
        y_test[mask],
        test_predictions[mask],
    )

    position_f1 = f1_score(
        y_test[mask],
        test_predictions[mask],
        average="macro",
        zero_division=0,
    )

    position_metrics[int(position)] = float(
        position_accuracy
    )

    position_macro_f1[int(position)] = float(
        position_f1
    )

    print(
        f"Position {position}: "
        f"acc={position_accuracy * 100:6.2f}% "
        f"F1={position_f1 * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Semantic-position metrics
# ============================================================

semantic_mask = np.isin(
    test["positions"],
    SEMANTIC_POSITIONS,
)

semantic_accuracy = accuracy_score(
    y_test[semantic_mask],
    test_predictions[semantic_mask],
)

semantic_macro_f1 = f1_score(
    y_test[semantic_mask],
    test_predictions[semantic_mask],
    average="macro",
    zero_division=0,
)


print()
print(
    f"Semantic-position accuracy: "
    f"{semantic_accuracy * 100:.2f}%"
)
print(
    f"Semantic-position macro F1: "
    f"{semantic_macro_f1 * 100:.2f}%"
)


# ============================================================
# Accuracy by token
# ============================================================

print()
print("====================================================")
print("               ACCURACY BY TOKEN")
print("====================================================")
print()


token_metrics = {}

for token in sorted(
    np.unique(y_test)
):
    mask = (
        y_test == token
    )

    token_accuracy = accuracy_score(
        y_test[mask],
        test_predictions[mask],
    )

    token_metrics[token] = float(
        token_accuracy
    )

    print(
        f"{token:12} "
        f"{token_accuracy * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Classification report / confusion matrix
# ============================================================

print()
print("Classification report:")
print()

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)


cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=model_classes,
)

print()
print("Class order:")
print(model_classes.tolist())

print()
print("Confusion matrix:")
print(cm)


# ============================================================
# Load old Step-62 BEST decoder metrics
# ============================================================

old_metrics = None

if OLD_METRICS_FILE.exists():
    with OLD_METRICS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        old_metrics = json.load(file)


# ============================================================
# Final comparison
# ============================================================

print()
print()
print("====================================================")
print("         STEP 76B-2 FINAL COMPARISON")
print("====================================================")
print()

print(
    f"Uniform chance:           "
    f"{chance * 100:6.2f}%"
)

print(
    f"Majority baseline:        "
    f"{majority_accuracy * 100:6.2f}%"
)


old_top1 = RECORD_TOP1
old_top2 = None
old_top3 = None
old_macro_f1 = None
old_semantic = RECORD_SEMANTIC_ACCURACY
old_position_metrics = {}


if old_metrics is not None:
    old_top1 = float(
        old_metrics.get(
            "test_top1_accuracy",
            RECORD_TOP1,
        )
    )

    old_top2 = old_metrics.get(
        "test_top2_accuracy"
    )

    old_top3 = old_metrics.get(
        "test_top3_accuracy"
    )

    old_macro_f1 = old_metrics.get(
        "test_macro_f1"
    )

    old_semantic = float(
        old_metrics.get(
            "semantic_position_accuracy",
            RECORD_SEMANTIC_ACCURACY,
        )
    )

    old_position_metrics = (
        old_metrics.get(
            "accuracy_by_position",
            {}
        )
    )


print()
print("OLD Step-62 best global decoder:")
print(
    f"  Top-1:                "
    f"{old_top1 * 100:6.2f}%"
)

if old_top2 is not None:
    print(
        f"  Top-2:                "
        f"{float(old_top2) * 100:6.2f}%"
    )

if old_top3 is not None:
    print(
        f"  Top-3:                "
        f"{float(old_top3) * 100:6.2f}%"
    )

if old_macro_f1 is not None:
    print(
        f"  Macro F1:             "
        f"{float(old_macro_f1) * 100:6.2f}%"
    )

print(
    f"  Semantic accuracy:    "
    f"{old_semantic * 100:6.2f}%"
)


print()
print(f"NEW rich {best['family']}:")
print(
    f"  Top-1:                "
    f"{test_accuracy * 100:6.2f}%"
)
print(
    f"  Top-2:                "
    f"{top2_accuracy * 100:6.2f}%"
)
print(
    f"  Top-3:                "
    f"{top3_accuracy * 100:6.2f}%"
)
print(
    f"  Macro F1:             "
    f"{test_macro_f1 * 100:6.2f}%"
)
print(
    f"  Semantic accuracy:    "
    f"{semantic_accuracy * 100:6.2f}%"
)
print(
    f"  Semantic macro F1:    "
    f"{semantic_macro_f1 * 100:6.2f}%"
)


print()
print(
    f"Top-1 change:           "
    f"{(test_accuracy - old_top1) * 100:+.2f} pp"
)
print(
    f"Semantic change:        "
    f"{(semantic_accuracy - old_semantic) * 100:+.2f} pp"
)

if old_macro_f1 is not None:
    print(
        f"Macro-F1 change:        "
        f"{(test_macro_f1 - float(old_macro_f1)) * 100:+.2f} pp"
    )


# ============================================================
# Important P0 / P2 comparison
# ============================================================

print()
print("Important semantic positions:")
print()

for position in [0, 2]:
    new_value = position_metrics.get(position)

    old_value = old_position_metrics.get(
        str(position)
    )

    if old_value is None:
        old_value = old_position_metrics.get(
            position
        )

    if old_value is None:
        print(
            f"  P{position}: NEW="
            f"{new_value * 100:6.2f}%"
        )
    else:
        print(
            f"  P{position}: "
            f"OLD={float(old_value) * 100:6.2f}%  "
            f"NEW={new_value * 100:6.2f}%  "
            f"CHANGE="
            f"{(new_value - float(old_value)) * 100:+.2f} pp"
        )


# ============================================================
# Decide whether closed loop is justified
# ============================================================

beats_top1 = (
    test_accuracy
    > old_top1
)

beats_semantic = (
    semantic_accuracy
    > old_semantic
)


print()
print("====================================================")
print("               STEP 76 DECISION")
print("====================================================")
print()

print(
    f"Beat old Top-1?          "
    f"{'YES' if beats_top1 else 'NO'}"
)
print(
    f"Beat old semantic acc?   "
    f"{'YES' if beats_semantic else 'NO'}"
)


if beats_top1 or beats_semantic:
    print()
    print(
        "Rich features improved at least one primary target."
    )
    print(
        "Next: run the rich decoder in closed-loop generation."
    )
else:
    print()
    print(
        "Rich features did not beat either primary target."
    )
    print(
        "Do NOT prioritize closed-loop rerun yet."
    )


# ============================================================
# Sample predictions
# ============================================================

print()
print("====================================================")
print("              SAMPLE PREDICTIONS")
print("====================================================")
print()

for index in range(
    min(
        50,
        len(y_test),
    )
):
    true_token = y_test[index]
    predicted_token = test_predictions[index]

    mark = (
        "OK"
        if true_token == predicted_token
        else "X"
    )

    ranked_indices = np.argsort(
        scores[index]
    )[::-1]

    top3_tokens = [
        str(model_classes[class_index])
        for class_index in ranked_indices[:3]
    ]

    print(
        f"{mark:2} "
        f"{test['example_ids'][index]:16} "
        f"seed={test['seeds'][index]} "
        f"pos={test['positions'][index]} "
        f"feedback={test['feedback'][index]:10} "
        f"target={true_token:10} "
        f"pred={predicted_token:10} "
        f"top3={top3_tokens}"
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

for family in family_order:
    result = family_best[family]

    serializable_family_results[family] = {
        "validation_accuracy": float(
            result["validation_accuracy"]
        ),
        "validation_macro_f1": float(
            result["validation_macro_f1"]
        ),
        "pca_components": int(
            result["components"]
        ),
        "parameter_name": result[
            "parameter_name"
        ],
        "parameter": float(
            result["parameter"]
        ),
        "pca_explained_variance": float(
            result["pca_explained_variance"]
        ),
    }


metrics = {
    "experiment": (
        "Step 76B rich prompt + generator-state decoder"
    ),
    "feature_file_train": str(TRAIN_FILE),
    "feature_file_validation": str(VALIDATION_FILE),
    "feature_file_test": str(TEST_FILE),
    "feature_dimensions": int(
        X_train.shape[1]
    ),
    "selected_family": best["family"],
    "selected_configuration": best["name"],
    "validation_accuracy": float(
        best["validation_accuracy"]
    ),
    "validation_macro_f1": float(
        best["validation_macro_f1"]
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
    "semantic_positions": [
        int(position)
        for position in SEMANTIC_POSITIONS
    ],
    "semantic_position_accuracy": float(
        semantic_accuracy
    ),
    "semantic_position_macro_f1": float(
        semantic_macro_f1
    ),
    "majority_baseline": float(
        majority_accuracy
    ),
    "uniform_chance": float(
        chance
    ),
    "accuracy_by_position": position_metrics,
    "macro_f1_by_position": position_macro_f1,
    "accuracy_by_token": token_metrics,
    "family_validation_results": (
        serializable_family_results
    ),
    "all_validation_results": all_results,
    "previous_best": {
        "top1": float(old_top1),
        "semantic_accuracy": float(old_semantic),
        "macro_f1": (
            None
            if old_macro_f1 is None
            else float(old_macro_f1)
        ),
        "top1_change": float(
            test_accuracy - old_top1
        ),
        "semantic_accuracy_change": float(
            semantic_accuracy - old_semantic
        ),
        "macro_f1_change": (
            None
            if old_macro_f1 is None
            else float(
                test_macro_f1
                - float(old_macro_f1)
            )
        ),
    },
    "decision": {
        "beats_previous_top1": bool(
            beats_top1
        ),
        "beats_previous_semantic_accuracy": bool(
            beats_semantic
        ),
        "rerun_closed_loop": bool(
            beats_top1 or beats_semantic
        ),
    },
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
# Final save summary
# ============================================================

print()
print(f"Saved winning model: {OUTPUT_MODEL}")
print(f"Saved metrics:       {OUTPUT_METRICS}")
print()
print("Step 76B-2 rich decoder comparison completed.")
