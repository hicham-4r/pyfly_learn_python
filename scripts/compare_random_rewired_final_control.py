from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
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
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# Step 78B-2
# DEGREE-PRESERVING RANDOM-REWIRED RESERVOIR READOUT
#
# Model selection:
#   DEVELOPMENT validation only.
#
# Final procedure:
#   1. Select decoder family/hyperparameters on old validation.
#   2. Optional old development-test diagnostic.
#   3. Refit the already-selected decoder on ALL 48
#      development prompts.
#   4. Score the FROZEN final holdout exactly once.
#
# IMPORTANT:
#   No final-holdout result is used for model selection.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Random-reservoir features from Step 78B-1
# ============================================================

TRAIN_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_random_rewired_train.npz"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_random_rewired_validation.npz"
)

DEV_TEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_random_rewired_test.npz"
)

FINAL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_random_rewired_final_holdout.npz"
)

FEATURE_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_reservoir_feature_manifest.json"
)


# ============================================================
# Frozen final-evaluation protocol
# ============================================================

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
)


# ============================================================
# Other final controls/results for comparison
# ============================================================

MALECNS_FINAL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_teacher_forced_metrics.json"
)

ENCODER_ONLY_FINAL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "encoder_only_final_control.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "random_rewired_final_control.joblib"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_final_control.json"
)


# ============================================================
# EXACT Step-62 search grid
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


# ============================================================
# Frozen experiment constants
# ============================================================

REWIRE_SEED = 78001

EXPECTED_FEATURE_DIMENSIONS = 41_111

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873

EXPECTED_ROWS = {
    "train": 1_140,
    "validation": 190,
    "development_test": 190,
    "final_holdout": 760,
}

EXPECTED_LOGICAL_PROMPTS = {
    "train": 36,
    "validation": 6,
    "development_test": 6,
    "final_holdout": 24,
}

EXPECTED_NEURAL_SEEDS = {
    0,
    1,
    2,
    3,
    4,
}

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def normalized_relative(
    path: Path,
) -> str:

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/",
    )


def load_split(
    path: Path,
    expected_rows: int,
    expected_logical_prompts: int,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            "Missing Step-78B-1 feature file: "
            f"{path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "X",
        "targets",
        "sample_ids",
        "example_ids",
        "intents",
        "target_positions",
        "feedback_tokens",
        "seeds",
        "input_neurons",
        "readout_population",
        "feature_block_names",
        "rewire_seed",
    }

    missing = (
        required
        -
        set(
            data.files
        )
    )

    if missing:

        raise ValueError(
            f"{path.name} is missing arrays: "
            f"{sorted(missing)}"
        )

    result = {
        "X": (
            data[
                "X"
            ].astype(
                np.float32
            )
        ),

        "targets": (
            data[
                "targets"
            ].astype(str)
        ),

        "sample_ids": (
            data[
                "sample_ids"
            ].astype(str)
        ),

        "example_ids": (
            data[
                "example_ids"
            ].astype(str)
        ),

        "intents": (
            data[
                "intents"
            ].astype(str)
        ),

        "positions": (
            data[
                "target_positions"
            ].astype(
                np.int32
            )
        ),

        "feedback": (
            data[
                "feedback_tokens"
            ].astype(str)
        ),

        "seeds": (
            data[
                "seeds"
            ].astype(
                np.int32
            )
        ),

        "input_neurons": (
            data[
                "input_neurons"
            ].astype(
                np.int64
            )
        ),

        "readout_population": (
            data[
                "readout_population"
            ].astype(
                np.int64
            )
        ),

        "feature_block_names": (
            data[
                "feature_block_names"
            ].astype(str)
        ),

        "rewire_seed": int(
            np.asarray(
                data[
                    "rewire_seed"
                ]
            ).item()
        ),
    }

    X = result[
        "X"
    ]

    row_count = (
        X.shape[
            0
        ]
    )

    if row_count != expected_rows:

        raise ValueError(
            f"{path.name}: expected "
            f"{expected_rows} rows, "
            f"got {row_count}"
        )

    if (
        X.ndim
        !=
        2
    ):

        raise ValueError(
            f"{path.name}: X must be 2-D."
        )

    if (
        X.shape[
            1
        ]
        !=
        EXPECTED_FEATURE_DIMENSIONS
    ):

        raise ValueError(
            f"{path.name}: expected "
            f"{EXPECTED_FEATURE_DIMENSIONS:,} "
            "features, got "
            f"{X.shape[1]:,}"
        )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            f"{path.name}: non-finite "
            "features found."
        )

    for key in [
        "targets",
        "sample_ids",
        "example_ids",
        "intents",
        "positions",
        "feedback",
        "seeds",
    ]:

        if (
            len(
                result[
                    key
                ]
            )
            !=
            row_count
        ):

            raise ValueError(
                f"{path.name}: row mismatch "
                f"for {key}"
            )

    if (
        len(
            set(
                result[
                    "example_ids"
                ].tolist()
            )
        )
        !=
        expected_logical_prompts
    ):

        raise ValueError(
            f"{path.name}: expected "
            f"{expected_logical_prompts} "
            "logical prompts."
        )

    actual_seeds = set(
        np.unique(
            result[
                "seeds"
            ]
        ).tolist()
    )

    if (
        actual_seeds
        !=
        EXPECTED_NEURAL_SEEDS
    ):

        raise ValueError(
            f"{path.name}: unexpected "
            f"neural seeds "
            f"{sorted(actual_seeds)}"
        )

    if (
        result[
            "rewire_seed"
        ]
        !=
        REWIRE_SEED
    ):

        raise ValueError(
            f"{path.name}: random "
            "rewiring seed changed."
        )

    if (
        len(
            result[
                "input_neurons"
            ]
        )
        !=
        EXPECTED_DIRECT_INPUT_NEURONS
    ):

        raise ValueError(
            f"{path.name}: direct-input "
            "population changed."
        )

    if (
        len(
            result[
                "readout_population"
            ]
        )
        !=
        EXPECTED_READOUT_NEURONS
    ):

        raise ValueError(
            f"{path.name}: readout "
            "population changed."
        )

    return result


# ============================================================
# Verify Step-78B-1 feature hashes + frozen holdout
# ============================================================

def verify_integrity() -> dict:

    required = [
        FEATURE_MANIFEST_FILE,
        PROTOCOL_FILE,
        FINAL_HOLDOUT_FILE,
        TRAIN_FILE,
        VALIDATION_FILE,
        DEV_TEST_FILE,
        FINAL_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: "
                f"{path}"
            )

    manifest = load_json(
        FEATURE_MANIFEST_FILE
    )

    if (
        manifest.get(
            "step"
        )
        !=
        "78B-1"
    ):

        raise ValueError(
            "Random-reservoir feature "
            "manifest has wrong step."
        )

    if (
        manifest.get(
            "control"
        )
        !=
        "degree_preserving_random_rewired_reservoir"
    ):

        raise ValueError(
            "Unexpected random-control type."
        )

    if (
        int(
            manifest.get(
                "rewire_seed",
                -1,
            )
        )
        !=
        REWIRE_SEED
    ):

        raise ValueError(
            "Random rewiring seed changed."
        )

    if (
        int(
            manifest.get(
                "feature_dimensions",
                -1,
            )
        )
        !=
        EXPECTED_FEATURE_DIMENSIONS
    ):

        raise ValueError(
            "Random-control feature "
            "dimensions changed."
        )

    # --------------------------------------------------------
    # Check every feature file against the 78B-1 manifest.
    # --------------------------------------------------------

    split_path_map = {
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
        "test": DEV_TEST_FILE,
        "final_holdout": FINAL_FILE,
    }

    manifest_splits = (
        manifest.get(
            "splits",
            {}
        )
    )

    for split_name, path in (
        split_path_map.items()
    ):

        info = manifest_splits.get(
            split_name
        )

        if not info:

            raise ValueError(
                "78B-1 manifest is missing "
                f"split {split_name}"
            )

        expected_hash = info.get(
            "sha256"
        )

        if (
            not expected_hash
            or
            sha256_file(
                path
            )
            !=
            expected_hash
        ):

            raise ValueError(
                f"{split_name}: feature "
                "file hash changed after "
                "Step 78B-1."
            )

    # --------------------------------------------------------
    # Frozen final holdout hash.
    # --------------------------------------------------------

    protocol = load_json(
        PROTOCOL_FILE
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Final protocol is no longer "
            "marked frozen."
        )

    frozen_hashes = (
        protocol.get(
            "frozen_file_hashes",
            {}
        )
    )

    expected_holdout_hash = None

    for stored_path, info in (
        frozen_hashes.items()
    ):

        if (
            stored_path.replace(
                "\\",
                "/",
            )
            ==
            "data/raw/final_holdout.jsonl"
        ):

            expected_holdout_hash = (
                info.get(
                    "sha256"
                )
            )

            break

    if not expected_holdout_hash:

        raise ValueError(
            "Frozen holdout hash missing "
            "from protocol."
        )

    if (
        sha256_file(
            FINAL_HOLDOUT_FILE
        )
        !=
        expected_holdout_hash
    ):

        raise ValueError(
            "Final holdout changed after "
            "Step 77A."
        )

    return manifest


# ============================================================
# Classifier builders
# ============================================================

def make_classifier(
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
        f"Unknown decoder family: "
        f"{family}"
    )


def build_candidate_specs():

    specs = []

    for components in PCA_COMPONENTS:

        for alpha in RIDGE_ALPHAS:

            specs.append(
                {
                    "family": "Ridge",
                    "components": components,
                    "parameter": float(
                        alpha
                    ),
                    "name": (
                        f"Ridge "
                        f"PCA={components} "
                        f"alpha={alpha}"
                    ),
                }
            )

            specs.append(
                {
                    "family": (
                        "Balanced Ridge"
                    ),
                    "components": components,
                    "parameter": float(
                        alpha
                    ),
                    "name": (
                        f"Balanced Ridge "
                        f"PCA={components} "
                        f"alpha={alpha}"
                    ),
                }
            )

        for c_value in LOGISTIC_C_VALUES:

            specs.append(
                {
                    "family": "Logistic",
                    "components": components,
                    "parameter": float(
                        c_value
                    ),
                    "name": (
                        f"Logistic "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),
                }
            )

            specs.append(
                {
                    "family": (
                        "Balanced Logistic"
                    ),
                    "components": components,
                    "parameter": float(
                        c_value
                    ),
                    "name": (
                        f"Balanced Logistic "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),
                }
            )

        for c_value in SVM_C_VALUES:

            specs.append(
                {
                    "family": "Linear SVM",
                    "components": components,
                    "parameter": float(
                        c_value
                    ),
                    "name": (
                        f"Linear SVM "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),
                }
            )

            specs.append(
                {
                    "family": (
                        "Balanced Linear SVM"
                    ),
                    "components": components,
                    "parameter": float(
                        c_value
                    ),
                    "name": (
                        f"Balanced Linear SVM "
                        f"PCA={components} "
                        f"C={c_value}"
                    ),
                }
            )

    return specs


# ============================================================
# Final sklearn pipeline from a selected specification
# ============================================================

def build_pipeline(
    spec: dict,
) -> Pipeline:

    return Pipeline(
        [
            (
                "scale",

                StandardScaler(),
            ),

            (
                "pca",

                PCA(
                    n_components=(
                        spec[
                            "components"
                        ]
                    ),

                    svd_solver=(
                        "randomized"
                    ),

                    random_state=42,
                ),
            ),

            (
                "classifier",

                make_classifier(
                    spec[
                        "family"
                    ],
                    spec[
                        "parameter"
                    ],
                ),
            ),
        ]
    )


# ============================================================
# Score helpers
# ============================================================

def model_scores(
    model,
    X,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    scores = np.asarray(
        model.decision_function(
            X
        )
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    classes = np.asarray(
        classifier.classes_
    ).astype(str)

    if scores.ndim == 1:

        scores = np.column_stack(
            [
                -scores,
                scores,
            ]
        )

    return (
        scores,
        classes,
    )


def top_k_accuracy(
    scores,
    true_labels,
    classes,
    k,
) -> float:

    k = min(
        int(
            k
        ),
        len(
            classes
        ),
    )

    top_indices = (
        np.argsort(
            scores,
            axis=1,
        )[
            :,
            -k:
        ]
    )

    correct = 0

    for row_index, target in enumerate(
        true_labels
    ):

        candidates = {
            str(
                classes[
                    class_index
                ]
            )

            for class_index
            in top_indices[
                row_index
            ]
        }

        if target in candidates:

            correct += 1

    return float(
        correct
        /
        len(
            true_labels
        )
    )


def calculate_metrics(
    model,
    data: dict,
) -> dict:

    X = data[
        "X"
    ]

    y = data[
        "targets"
    ]

    predictions = (
        model.predict(
            X
        ).astype(str)
    )

    (
        scores,
        classes,
    ) = model_scores(
        model,
        X,
    )

    positions = data[
        "positions"
    ]

    semantic_mask = np.isin(
        positions,
        SEMANTIC_POSITIONS,
    )

    position_metrics = {}

    for position in sorted(
        np.unique(
            positions
        )
    ):

        mask = (
            positions
            ==
            position
        )

        position_metrics[
            str(
                int(
                    position
                )
            )
        ] = float(
            accuracy_score(
                y[
                    mask
                ],
                predictions[
                    mask
                ],
            )
        )

    token_metrics = {}

    for token in sorted(
        np.unique(
            y
        )
    ):

        mask = (
            y
            ==
            token
        )

        token_metrics[
            str(
                token
            )
        ] = float(
            accuracy_score(
                y[
                    mask
                ],
                predictions[
                    mask
                ],
            )
        )

    intent_metrics = {}

    for intent in sorted(
        np.unique(
            data[
                "intents"
            ]
        )
    ):

        mask = (
            data[
                "intents"
            ]
            ==
            intent
        )

        intent_metrics[
            str(
                intent
            )
        ] = float(
            accuracy_score(
                y[
                    mask
                ],
                predictions[
                    mask
                ],
            )
        )

    cm = confusion_matrix(
        y,
        predictions,
        labels=classes,
    )

    return {
        "top1_accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),

        "top2_accuracy": (
            top_k_accuracy(
                scores,
                y,
                classes,
                2,
            )
        ),

        "top3_accuracy": (
            top_k_accuracy(
                scores,
                y,
                classes,
                3,
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "semantic_accuracy": float(
            accuracy_score(
                y[
                    semantic_mask
                ],
                predictions[
                    semantic_mask
                ],
            )
        ),

        "semantic_macro_f1": float(
            f1_score(
                y[
                    semantic_mask
                ],
                predictions[
                    semantic_mask
                ],
                average="macro",
                zero_division=0,
            )
        ),

        "accuracy_by_position": (
            position_metrics
        ),

        "accuracy_by_token": (
            token_metrics
        ),

        "accuracy_by_intent": (
            intent_metrics
        ),

        "class_order": (
            classes.tolist()
        ),

        "confusion_matrix": (
            cm.tolist()
        ),
    }


# ============================================================
# Mean +/- SD across neural seeds
# ============================================================

def per_seed_final_metrics(
    model,
    final_data: dict,
) -> dict:

    per_seed = {}

    for seed in sorted(
        EXPECTED_NEURAL_SEEDS
    ):

        mask = (
            final_data[
                "seeds"
            ]
            ==
            seed
        )

        subset = {
            "X": (
                final_data[
                    "X"
                ][
                    mask
                ]
            ),

            "targets": (
                final_data[
                    "targets"
                ][
                    mask
                ]
            ),

            "positions": (
                final_data[
                    "positions"
                ][
                    mask
                ]
            ),

            "intents": (
                final_data[
                    "intents"
                ][
                    mask
                ]
            ),
        }

        metrics = (
            calculate_metrics(
                model,
                subset,
            )
        )

        per_seed[
            str(
                seed
            )
        ] = {
            "top1_accuracy": (
                metrics[
                    "top1_accuracy"
                ]
            ),

            "macro_f1": (
                metrics[
                    "macro_f1"
                ]
            ),

            "semantic_accuracy": (
                metrics[
                    "semantic_accuracy"
                ]
            ),
        }

    summary = {}

    for metric_name in [
        "top1_accuracy",
        "macro_f1",
        "semantic_accuracy",
    ]:

        values = np.asarray(
            [
                per_seed[
                    str(
                        seed
                    )
                ][
                    metric_name
                ]

                for seed in sorted(
                    EXPECTED_NEURAL_SEEDS
                )
            ],
            dtype=float,
        )

        summary[
            metric_name
        ] = {
            "mean": float(
                np.mean(
                    values
                )
            ),

            "sd": float(
                np.std(
                    values,
                    ddof=0,
                )
            ),
        }

    return {
        "per_seed": per_seed,
        "mean_sd": summary,
    }


# ============================================================
# Read comparison metrics robustly
# ============================================================

def load_malecns_final() -> dict | None:

    if not MALECNS_FINAL_FILE.exists():

        return None

    data = load_json(
        MALECNS_FINAL_FILE
    )

    if (
        "aggregate"
        in data
    ):

        return data[
            "aggregate"
        ]

    return data


def load_encoder_only_final() -> dict | None:

    if not ENCODER_ONLY_FINAL_FILE.exists():

        return None

    data = load_json(
        ENCODER_ONLY_FINAL_FILE
    )

    return data.get(
        "final_teacher_forced"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        " PyFly STEP 78B-2 - RANDOM RESERVOIR READOUT"
    )

    print(
        "===================================================="
    )

    print()

    manifest = verify_integrity()

    print(
        "Step-78B-1 feature integrity: PASS"
    )

    print(
        "Frozen final holdout:         PASS"
    )

    print(
        f"Random rewiring seed:         "
        f"{REWIRE_SEED}"
    )

    # ========================================================
    # Load features
    # ========================================================

    train = load_split(
        TRAIN_FILE,
        EXPECTED_ROWS[
            "train"
        ],
        EXPECTED_LOGICAL_PROMPTS[
            "train"
        ],
    )

    validation = load_split(
        VALIDATION_FILE,
        EXPECTED_ROWS[
            "validation"
        ],
        EXPECTED_LOGICAL_PROMPTS[
            "validation"
        ],
    )

    dev_test = load_split(
        DEV_TEST_FILE,
        EXPECTED_ROWS[
            "development_test"
        ],
        EXPECTED_LOGICAL_PROMPTS[
            "development_test"
        ],
    )

    final_data = load_split(
        FINAL_FILE,
        EXPECTED_ROWS[
            "final_holdout"
        ],
        EXPECTED_LOGICAL_PROMPTS[
            "final_holdout"
        ],
    )

    # --------------------------------------------------------
    # Logical-prompt leakage checks
    # --------------------------------------------------------

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

    dev_test_ids = set(
        dev_test[
            "example_ids"
        ].tolist()
    )

    final_ids = set(
        final_data[
            "example_ids"
        ].tolist()
    )

    if not train_ids.isdisjoint(
        validation_ids
    ):

        raise ValueError(
            "Train/validation prompt leakage."
        )

    if not train_ids.isdisjoint(
        dev_test_ids
    ):

        raise ValueError(
            "Train/dev-test prompt leakage."
        )

    if not validation_ids.isdisjoint(
        dev_test_ids
    ):

        raise ValueError(
            "Validation/dev-test prompt leakage."
        )

    development_ids = (
        train_ids
        |
        validation_ids
        |
        dev_test_ids
    )

    if (
        len(
            development_ids
        )
        !=
        48
    ):

        raise ValueError(
            "Expected 48 total development "
            f"prompts, got "
            f"{len(development_ids)}"
        )

    if not final_ids.isdisjoint(
        development_ids
    ):

        raise ValueError(
            "Frozen final prompts overlap "
            "development prompts."
        )

    print()

    print(
        f"Train:                    "
        f"{train['X'].shape}"
    )

    print(
        f"Validation:               "
        f"{validation['X'].shape}"
    )

    print(
        f"Development test:         "
        f"{dev_test['X'].shape}"
    )

    print(
        f"Frozen final holdout:     "
        f"{final_data['X'].shape}"
    )

    print()

    print(
        "Logical-prompt leakage:    PASS"
    )

    # ========================================================
    # Classes / chance
    # ========================================================

    classes = sorted(
        np.unique(
            train[
                "targets"
            ]
        )
    )

    uniform_chance = (
        1.0
        /
        len(
            classes
        )
    )

    unique_tokens, counts = np.unique(
        train[
            "targets"
        ],
        return_counts=True,
    )

    majority_token = str(
        unique_tokens[
            np.argmax(
                counts
            )
        ]
    )

    final_majority_accuracy = float(
        np.mean(
            final_data[
                "targets"
            ]
            ==
            majority_token
        )
    )

    print()

    print(
        f"Target classes:            "
        f"{len(classes)}"
    )

    print(
        f"Uniform chance:            "
        f"{uniform_chance * 100:.2f}%"
    )

    print(
        f"Development majority:      "
        f"{majority_token}"
    )

    print(
        f"Final majority accuracy:   "
        f"{final_majority_accuracy * 100:.2f}%"
    )

    # ========================================================
    # Efficient validation search
    #
    # Scale ONCE.
    # Fit each PCA dimensionality ONCE.
    # Then train the six decoder families on cached PCA data.
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "     RANDOM RESERVOIR VALIDATION SEARCH"
    )

    print(
        "===================================================="
    )

    print()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        train[
            "X"
        ]
    )

    X_val_scaled = scaler.transform(
        validation[
            "X"
        ]
    )

    pca_cache = {}

    for components in PCA_COMPONENTS:

        if (
            components
            >=
            min(
                X_train_scaled.shape
            )
        ):

            continue

        print(
            f"Computing PCA={components}..."
        )

        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=42,
        )

        X_train_pca = pca.fit_transform(
            X_train_scaled
        )

        X_val_pca = pca.transform(
            X_val_scaled
        )

        pca_cache[
            components
        ] = {
            "train": (
                X_train_pca
            ),

            "validation": (
                X_val_pca
            ),
        }

    candidate_specs = (
        build_candidate_specs()
    )

    best = None
    family_best = {}
    all_results = []

    valid_specs = [
        spec

        for spec in candidate_specs

        if spec[
            "components"
        ]
        in pca_cache
    ]

    for index, spec in enumerate(
        valid_specs,
        start=1,
    ):

        components = spec[
            "components"
        ]

        classifier = make_classifier(
            spec[
                "family"
            ],
            spec[
                "parameter"
            ],
        )

        classifier.fit(
            pca_cache[
                components
            ][
                "train"
            ],
            train[
                "targets"
            ],
        )

        predictions = (
            classifier.predict(
                pca_cache[
                    components
                ][
                    "validation"
                ]
            )
        )

        validation_accuracy = float(
            accuracy_score(
                validation[
                    "targets"
                ],
                predictions,
            )
        )

        validation_macro_f1 = float(
            f1_score(
                validation[
                    "targets"
                ],
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        result = {
            **spec,

            "validation_accuracy": (
                validation_accuracy
            ),

            "validation_macro_f1": (
                validation_macro_f1
            ),
        }

        all_results.append(
            result
        )

        print(
            f"[{index:03}/"
            f"{len(valid_specs):03}] "
            f"{spec['family']:20} "
            f"PCA={components:3} "
            f"param="
            f"{spec['parameter']:8.4f} "
            f"acc="
            f"{validation_accuracy * 100:6.2f}% "
            f"F1="
            f"{validation_macro_f1 * 100:6.2f}%"
        )

        # ----------------------------------------------------
        # Best inside family
        # ----------------------------------------------------

        current_family = (
            family_best.get(
                spec[
                    "family"
                ]
            )
        )

        if (
            current_family is None
            or
            validation_macro_f1
            >
            current_family[
                "validation_macro_f1"
            ]
            or (
                validation_macro_f1
                ==
                current_family[
                    "validation_macro_f1"
                ]
                and
                validation_accuracy
                >
                current_family[
                    "validation_accuracy"
                ]
            )
        ):

            family_best[
                spec[
                    "family"
                ]
            ] = (
                result.copy()
            )

        # ----------------------------------------------------
        # Global best
        # ----------------------------------------------------

        if (
            best is None
            or
            validation_macro_f1
            >
            best[
                "validation_macro_f1"
            ]
            or (
                validation_macro_f1
                ==
                best[
                    "validation_macro_f1"
                ]
                and
                validation_accuracy
                >
                best[
                    "validation_accuracy"
                ]
            )
        ):

            best = (
                result.copy()
            )

    if best is None:

        raise RuntimeError(
            "No random-reservoir decoder "
            "candidate was selected."
        )

    print()

    print(
        "===================================================="
    )

    print(
        "       SELECTED RANDOM-RESERVOIR DECODER"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Family:                 "
        f"{best['family']}"
    )

    print(
        f"Configuration:          "
        f"{best['name']}"
    )

    print(
        f"Validation accuracy:    "
        f"{best['validation_accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation macro F1:    "
        f"{best['validation_macro_f1'] * 100:.2f}%"
    )

    print()

    print(
        "Best configuration by family:"
    )

    for family in [
        "Ridge",
        "Balanced Ridge",
        "Logistic",
        "Balanced Logistic",
        "Linear SVM",
        "Balanced Linear SVM",
    ]:

        row = family_best[
            family
        ]

        print(
            f"  {family:20} "
            f"acc="
            f"{row['validation_accuracy'] * 100:6.2f}% "
            f"F1="
            f"{row['validation_macro_f1'] * 100:6.2f}% "
            f"{row['name']}"
        )

    # ========================================================
    # Old development-test diagnostic
    #
    # Refit winner on train + validation ONLY.
    # ========================================================

    dev_diagnostic_model = (
        build_pipeline(
            best
        )
    )

    X_train_val = np.concatenate(
        [
            train[
                "X"
            ],
            validation[
                "X"
            ],
        ],
        axis=0,
    )

    y_train_val = np.concatenate(
        [
            train[
                "targets"
            ],
            validation[
                "targets"
            ],
        ],
        axis=0,
    )

    print()

    print(
        "Refitting selected random-reservoir "
        "decoder on train + validation for "
        "development-test diagnostic..."
    )

    dev_diagnostic_model.fit(
        X_train_val,
        y_train_val,
    )

    dev_test_metrics = (
        calculate_metrics(
            dev_diagnostic_model,
            dev_test,
        )
    )

    print()

    print(
        "Development-test diagnostic "
        "(NOT final):"
    )

    print(
        f"  Top-1:             "
        f"{dev_test_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{dev_test_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy: "
        f"{dev_test_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    # ========================================================
    # FINAL refit on ALL 48 development prompts
    # ========================================================

    final_model = (
        build_pipeline(
            best
        )
    )

    X_all_development = np.concatenate(
        [
            train[
                "X"
            ],
            validation[
                "X"
            ],
            dev_test[
                "X"
            ],
        ],
        axis=0,
    )

    y_all_development = np.concatenate(
        [
            train[
                "targets"
            ],
            validation[
                "targets"
            ],
            dev_test[
                "targets"
            ],
        ],
        axis=0,
    )

    print()

    print(
        "Refitting the already-selected "
        "random-reservoir decoder on ALL "
        "48 development prompts..."
    )

    final_model.fit(
        X_all_development,
        y_all_development,
    )

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        OUTPUT_MODEL,
    )

    # ========================================================
    # FROZEN FINAL HOLDOUT — score once
    # ========================================================

    final_metrics = (
        calculate_metrics(
            final_model,
            final_data,
        )
    )

    seed_metrics = (
        per_seed_final_metrics(
            final_model,
            final_data,
        )
    )

    # ========================================================
    # Final report
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        " RANDOM-REWIRED FINAL TEACHER-FORCED RESULTS"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Top-1:              "
        f"{final_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Top-2:              "
        f"{final_metrics['top2_accuracy'] * 100:.2f}%"
    )

    print(
        f"Top-3:              "
        f"{final_metrics['top3_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:           "
        f"{final_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Semantic accuracy:  "
        f"{final_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    print(
        f"Semantic macro F1:  "
        f"{final_metrics['semantic_macro_f1'] * 100:.2f}%"
    )

    print()

    print(
        "Mean +/- SD across neural seeds:"
    )

    for label, metric_name in [
        (
            "Top-1",
            "top1_accuracy",
        ),
        (
            "Macro F1",
            "macro_f1",
        ),
        (
            "Semantic accuracy",
            "semantic_accuracy",
        ),
    ]:

        row = (
            seed_metrics[
                "mean_sd"
            ][
                metric_name
            ]
        )

        print(
            f"  {label:18} "
            f"{row['mean'] * 100:6.2f}% "
            f"+/- "
            f"{row['sd'] * 100:5.2f} pp"
        )

    print()

    print(
        "Accuracy by output position:"
    )

    for position, value in (
        final_metrics[
            "accuracy_by_position"
        ].items()
    ):

        print(
            f"  P{position}: "
            f"{value * 100:6.2f}%"
        )

    print()

    print(
        "Accuracy by target token:"
    )

    for token, value in (
        final_metrics[
            "accuracy_by_token"
        ].items()
    ):

        print(
            f"  {token:12} "
            f"{value * 100:6.2f}%"
        )

    print()

    print(
        "Per-seed Top-1 / semantic / Macro F1:"
    )

    for seed in sorted(
        EXPECTED_NEURAL_SEEDS
    ):

        row = (
            seed_metrics[
                "per_seed"
            ][
                str(
                    seed
                )
            ]
        )

        print(
            f"  seed={seed}: "
            f"Top-1="
            f"{row['top1_accuracy'] * 100:6.2f}%  "
            f"Semantic="
            f"{row['semantic_accuracy'] * 100:6.2f}%  "
            f"MacroF1="
            f"{row['macro_f1'] * 100:6.2f}%"
        )

    # ========================================================
    # Three-way final comparison
    # ========================================================

    malecns = (
        load_malecns_final()
    )

    encoder_only = (
        load_encoder_only_final()
    )

    print()

    print(
        "===================================================="
    )

    print(
        "      FINAL TEACHER-FORCED CONTROL COMPARISON"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"{'System':28}"
        f"{'Top-1':>10}"
        f"{'Macro F1':>12}"
        f"{'Semantic':>12}"
    )

    print(
        "-" * 62
    )

    if encoder_only is not None:

        print(
            f"{'Encoder-only':28}"
            f"{encoder_only['top1_accuracy'] * 100:9.2f}%"
            f"{encoder_only['macro_f1'] * 100:11.2f}%"
            f"{encoder_only['semantic_accuracy'] * 100:11.2f}%"
        )

    if malecns is not None:

        print(
            f"{'Real MaleCNS':28}"
            f"{malecns['top1_accuracy'] * 100:9.2f}%"
            f"{malecns['macro_f1'] * 100:11.2f}%"
            f"{malecns['semantic_accuracy'] * 100:11.2f}%"
        )

    print(
        f"{'Random-rewired MaleCNS':28}"
        f"{final_metrics['top1_accuracy'] * 100:9.2f}%"
        f"{final_metrics['macro_f1'] * 100:11.2f}%"
        f"{final_metrics['semantic_accuracy'] * 100:11.2f}%"
    )

    if malecns is not None:

        print()

        print(
            "Random-rewired minus real MaleCNS:"
        )

        print(
            f"  Top-1:    "
            f"{(final_metrics['top1_accuracy'] - malecns['top1_accuracy']) * 100:+.2f} pp"
        )

        print(
            f"  Macro F1: "
            f"{(final_metrics['macro_f1'] - malecns['macro_f1']) * 100:+.2f} pp"
        )

        print(
            f"  Semantic: "
            f"{(final_metrics['semantic_accuracy'] - malecns['semantic_accuracy']) * 100:+.2f} pp"
        )

    # ========================================================
    # Save metrics
    # ========================================================

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "step": "78B-2",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "control": (
            "degree_preserving_random_rewired_reservoir"
        ),

        "rewire_seed": (
            REWIRE_SEED
        ),

        "selection": {
            "selection_split": (
                "development validation only"
            ),

            "selection_metric": (
                "macro F1; accuracy tie-break"
            ),

            "search_grid": {
                "pca_components": (
                    PCA_COMPONENTS
                ),

                "ridge_alphas": (
                    RIDGE_ALPHAS
                ),

                "logistic_c_values": (
                    LOGISTIC_C_VALUES
                ),

                "svm_c_values": (
                    SVM_C_VALUES
                ),
            },

            "selected_family": (
                best[
                    "family"
                ]
            ),

            "selected_configuration": (
                best[
                    "name"
                ]
            ),

            "validation_accuracy": (
                best[
                    "validation_accuracy"
                ]
            ),

            "validation_macro_f1": (
                best[
                    "validation_macro_f1"
                ]
            ),

            "family_best": (
                family_best
            ),
        },

        "development_test_diagnostic": (
            dev_test_metrics
        ),

        "final_teacher_forced": (
            final_metrics
        ),

        "final_neural_seed_uncertainty": (
            seed_metrics
        ),

        "comparison": {
            "encoder_only": (
                encoder_only
            ),

            "real_malecns": (
                malecns
            ),

            "random_rewired_malecns": (
                final_metrics
            ),
        },

        "integrity": {
            "feature_manifest_sha256": (
                sha256_file(
                    FEATURE_MANIFEST_FILE
                )
            ),

            "protocol_sha256": (
                sha256_file(
                    PROTOCOL_FILE
                )
            ),

            "final_holdout_sha256": (
                sha256_file(
                    FINAL_HOLDOUT_FILE
                )
            ),

            "model_sha256": (
                sha256_file(
                    OUTPUT_MODEL
                )
            ),

            "feature_files": {
                "train": (
                    sha256_file(
                        TRAIN_FILE
                    )
                ),

                "validation": (
                    sha256_file(
                        VALIDATION_FILE
                    )
                ),

                "development_test": (
                    sha256_file(
                        DEV_TEST_FILE
                    )
                ),

                "final_holdout": (
                    sha256_file(
                        FINAL_FILE
                    )
                ),
            },
        },

        "rules": [
            (
                "The random-reservoir decoder was "
                "selected using development validation "
                "only."
            ),

            (
                "The final holdout was not used for "
                "decoder-family or hyperparameter selection."
            ),

            (
                "The final model was refit on all 48 "
                "development prompts only."
            ),

            (
                "The exact Step-62 decoder search grid "
                "was used for fairness with real MaleCNS."
            ),
        ],
    }

    OUTPUT_METRICS.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{normalized_relative(OUTPUT_MODEL)}"
    )

    print(
        "  "
        f"{normalized_relative(OUTPUT_METRICS)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 78B-2 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Next: Step 78B-3 - run this selected "
        "random-rewired decoder in true closed-loop "
        "generation for the frozen 24-prompt holdout."
    )


if __name__ == "__main__":
    main()
