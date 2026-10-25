from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Step 77B
# Refit the FROZEN Step-62 winning decoder on ALL 48
# development prompts.
#
# IMPORTANT:
#   * No architecture search.
#   * No hyperparameter search.
#   * No final-holdout scoring.
#   * The final holdout is only integrity-checked by hash.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_train.npz"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_validation.npz"
)

DEV_TEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_test.npz"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

DECODER_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "generator_decoder_comparison.json"
)

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
)

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "generator_final_frozen_decoder.joblib"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_decoder_refit_manifest.json"
)


# ============================================================
# Frozen winner from Step 62 / Step 77A
# ============================================================

EXPECTED_FAMILY = "Balanced Logistic"

EXPECTED_CONFIGURATION = (
    "Balanced Logistic PCA=100 C=0.1"
)

PCA_COMPONENTS = 100
LOGISTIC_C = 0.1

PCA_RANDOM_STATE = 42
CLASSIFIER_RANDOM_STATE = 42

EXPECTED_LOGICAL_PROMPTS = 48

EXPECTED_NEURAL_SEEDS = {
    0,
    1,
    2,
    3,
    4,
}

EXPECTED_TOTAL_ROWS = 1520

EXPECTED_FEATURE_BLOCKS = [
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
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


def load_split(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            "Missing frozen Step-62 "
            f"feature file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
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
            f"{path.name} is missing "
            f"required arrays: "
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
    }

    row_count = (
        result[
            "X"
        ].shape[
            0
        ]
    )

    for key in (
        "targets",
        "sample_ids",
        "example_ids",
        "intents",
        "positions",
        "feedback",
        "seeds",
    ):

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
                f"Row-count mismatch "
                f"in {path.name}: "
                f"X has {row_count}, "
                f"{key} has "
                f"{len(result[key])}"
            )

    if (
        result[
            "X"
        ].ndim
        !=
        2
    ):

        raise ValueError(
            "Expected a 2-D feature "
            f"matrix in {path.name}, "
            f"got "
            f"{result['X'].shape}"
        )

    if not np.isfinite(
        result[
            "X"
        ]
    ).all():

        raise ValueError(
            "Non-finite feature values "
            f"found in {path.name}"
        )

    return result


# ============================================================
# Build EXACT frozen model
# ============================================================

def make_frozen_model() -> Pipeline:

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
                        PCA_COMPONENTS
                    ),

                    svd_solver=(
                        "randomized"
                    ),

                    random_state=(
                        PCA_RANDOM_STATE
                    ),
                ),
            ),

            (
                "classifier",

                LogisticRegression(
                    C=(
                        LOGISTIC_C
                    ),

                    solver="lbfgs",

                    class_weight=(
                        "balanced"
                    ),

                    max_iter=5000,

                    random_state=(
                        CLASSIFIER_RANDOM_STATE
                    ),
                ),
            ),
        ]
    )


# ============================================================
# Protocol verification
# ============================================================

def verify_frozen_protocol():

    if not PROTOCOL_FILE.exists():

        raise FileNotFoundError(
            "Missing Step-77A protocol "
            f"file: {PROTOCOL_FILE}"
        )

    if not DECODER_METRICS_FILE.exists():

        raise FileNotFoundError(
            "Missing Step-62 decoder "
            f"metrics: {DECODER_METRICS_FILE}"
        )

    if not FINAL_HOLDOUT_FILE.exists():

        raise FileNotFoundError(
            "Missing frozen final "
            f"holdout: {FINAL_HOLDOUT_FILE}"
        )

    protocol = load_json(
        PROTOCOL_FILE
    )

    decoder_metrics = load_json(
        DECODER_METRICS_FILE
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Final protocol is not "
            "marked as frozen."
        )

    decoder_selection = (
        protocol.get(
            "decoder_selection",
            {},
        )
    )

    protocol_family = (
        decoder_selection.get(
            "selected_family"
        )
    )

    protocol_configuration = (
        decoder_selection.get(
            "selected_configuration"
        )
    )

    metrics_family = (
        decoder_metrics.get(
            "selected_family"
        )
    )

    metrics_configuration = (
        decoder_metrics.get(
            "selected_configuration"
        )
    )

    if (
        protocol_family
        !=
        EXPECTED_FAMILY
    ):

        raise ValueError(
            "Frozen protocol decoder "
            "family changed: "
            f"{protocol_family!r}"
        )

    if (
        protocol_configuration
        !=
        EXPECTED_CONFIGURATION
    ):

        raise ValueError(
            "Frozen protocol decoder "
            "configuration changed: "
            f"{protocol_configuration!r}"
        )

    if (
        metrics_family
        !=
        EXPECTED_FAMILY
    ):

        raise ValueError(
            "Step-62 metrics decoder "
            "family changed: "
            f"{metrics_family!r}"
        )

    if (
        metrics_configuration
        !=
        EXPECTED_CONFIGURATION
    ):

        raise ValueError(
            "Step-62 metrics decoder "
            "configuration changed: "
            f"{metrics_configuration!r}"
        )

    architecture = (
        protocol.get(
            "architecture",
            {},
        )
    )

    if (
        architecture.get(
            "recurrent_weights"
        )
        !=
        "frozen"
    ):

        raise ValueError(
            "Protocol no longer marks "
            "MaleCNS recurrent weights "
            "as frozen."
        )

    if (
        architecture.get(
            "feature_blocks"
        )
        !=
        EXPECTED_FEATURE_BLOCKS
    ):

        raise ValueError(
            "Frozen feature-block "
            "definition changed."
        )

    # ========================================================
    # Verify final holdout hash.
    #
    # We are NOT using holdout labels for model selection.
    # ========================================================

    frozen_hashes = (
        protocol.get(
            "frozen_file_hashes",
            {},
        )
    )

    expected_holdout_hash = None

    for (
        stored_path,
        info,
    ) in frozen_hashes.items():

        normalized = (
            stored_path.replace(
                "\\",
                "/",
            )
        )

        if (
            normalized
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
            "Frozen protocol does not "
            "contain the SHA-256 for "
            "final_holdout.jsonl"
        )

    current_holdout_hash = (
        sha256_file(
            FINAL_HOLDOUT_FILE
        )
    )

    if (
        current_holdout_hash
        !=
        expected_holdout_hash
    ):

        raise ValueError(
            "Final holdout SHA-256 "
            "no longer matches the "
            "Step-77A frozen protocol.\n"
            f"Expected: "
            f"{expected_holdout_hash}\n"
            f"Current:  "
            f"{current_holdout_hash}"
        )

    return (
        protocol,
        decoder_metrics,
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
        "      PyFly STEP 77B - REFIT FROZEN DECODER"
    )

    print(
        "===================================================="
    )

    print()

    (
        protocol,
        decoder_metrics,

    ) = verify_frozen_protocol()

    print(
        "Frozen protocol integrity: PASS"
    )

    print(
        f"Frozen family:        "
        f"{EXPECTED_FAMILY}"
    )

    print(
        f"Frozen configuration: "
        f"{EXPECTED_CONFIGURATION}"
    )

    print()


    # ========================================================
    # Load all development features
    # ========================================================

    train = load_split(
        TRAIN_FILE
    )

    validation = load_split(
        VALIDATION_FILE
    )

    dev_test = load_split(
        DEV_TEST_FILE
    )


    splits = {

        "train": train,

        "validation": (
            validation
        ),

        "development_test": (
            dev_test
        ),
    }


    for (
        name,
        split,
    ) in splits.items():

        print(
            f"{name:17} "
            f"rows="
            f"{split['X'].shape[0]:4} "
            f"features="
            f"{split['X'].shape[1]:,} "
            f"logical_prompts="
            f"{len(set(split['example_ids'])):2}"
        )


    # ========================================================
    # Split leakage checks
    # ========================================================

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


    if not train_ids.isdisjoint(
        validation_ids
    ):

        raise ValueError(
            "Train and validation "
            "logical prompts overlap."
        )


    if not train_ids.isdisjoint(
        dev_test_ids
    ):

        raise ValueError(
            "Train and development-test "
            "logical prompts overlap."
        )


    if not validation_ids.isdisjoint(
        dev_test_ids
    ):

        raise ValueError(
            "Validation and development-test "
            "logical prompts overlap."
        )


    all_example_ids = (
        train_ids
        |
        validation_ids
        |
        dev_test_ids
    )


    if (
        len(
            all_example_ids
        )
        !=
        EXPECTED_LOGICAL_PROMPTS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_LOGICAL_PROMPTS} "
            "unique development prompts, "
            f"got "
            f"{len(all_example_ids)}"
        )


    feature_dimensions = {

        split[
            "X"
        ].shape[
            1
        ]

        for split
        in splits.values()
    }


    if (
        len(
            feature_dimensions
        )
        !=
        1
    ):

        raise ValueError(
            "Development feature dimensions "
            "do not match across splits: "
            f"{sorted(feature_dimensions)}"
        )


    # ========================================================
    # Combine all 48 development prompts
    # ========================================================

    X_all = np.concatenate(
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


    y_all = np.concatenate(
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


    example_ids_all = np.concatenate(
        [
            train[
                "example_ids"
            ],

            validation[
                "example_ids"
            ],

            dev_test[
                "example_ids"
            ],
        ],

        axis=0,
    )


    seeds_all = np.concatenate(
        [
            train[
                "seeds"
            ],

            validation[
                "seeds"
            ],

            dev_test[
                "seeds"
            ],
        ],

        axis=0,
    )


    if (
        X_all.shape[
            0
        ]
        !=
        EXPECTED_TOTAL_ROWS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_TOTAL_ROWS} "
            "total neural rows, "
            f"got "
            f"{X_all.shape[0]}"
        )


    actual_seeds = set(
        np.unique(
            seeds_all
        ).tolist()
    )


    if (
        actual_seeds
        !=
        EXPECTED_NEURAL_SEEDS
    ):

        raise ValueError(
            "Unexpected neural seeds: "
            f"{sorted(actual_seeds)}"
        )


    if (
        len(
            set(
                example_ids_all.tolist()
            )
        )
        !=
        EXPECTED_LOGICAL_PROMPTS
    ):

        raise ValueError(
            "Combined development data "
            "does not contain exactly "
            "48 logical prompts."
        )


    if not np.isfinite(
        X_all
    ).all():

        raise ValueError(
            "Combined development features "
            "contain non-finite values."
        )


    if (
        PCA_COMPONENTS
        >=
        min(
            X_all.shape
        )
    ):

        raise ValueError(
            f"PCA={PCA_COMPONENTS} "
            f"is invalid for "
            f"shape {X_all.shape}"
        )


    print()

    print(
        "All-development refit set:"
    )


    print(
        f"  logical prompts:     "
        f"{len(all_example_ids)}"
    )


    print(
        f"  neural rows:         "
        f"{X_all.shape[0]:,}"
    )


    print(
        f"  feature dimensions:  "
        f"{X_all.shape[1]:,}"
    )


    print(
        f"  neural seeds:        "
        f"{sorted(actual_seeds)}"
    )


    print(
        f"  target classes:      "
        f"{len(np.unique(y_all))}"
    )


    # ========================================================
    # Target distribution
    # ========================================================

    target_counts = Counter(
        y_all.tolist()
    )


    print()

    print(
        "All-development "
        "target distribution:"
    )


    for token in sorted(
        target_counts
    ):

        print(
            f"  "
            f"{token:12} "
            f"{target_counts[token]:4}"
        )


    # ========================================================
    # Build exact frozen model
    # ========================================================

    model = make_frozen_model()


    print()

    print(
        "Refitting the frozen decoder "
        "on all 48 development prompts..."
    )


    model.fit(
        X_all,
        y_all,
    )


    # ========================================================
    # Training sanity check ONLY
    # ========================================================

    training_predictions = (
        model.predict(
            X_all
        )
    )


    training_accuracy = float(
        np.mean(
            training_predictions
            ==
            y_all
        )
    )


    classifier = (
        model.named_steps[
            "classifier"
        ]
    )


    pca = (
        model.named_steps[
            "pca"
        ]
    )


    print()

    print(
        "Refit complete."
    )


    print(
        f"PCA components fitted: "
        f"{pca.n_components_}"
    )


    print(
        f"Classifier classes:    "
        f"{len(classifier.classes_)}"
    )


    print(
        "Training sanity accuracy: "
        f"{training_accuracy * 100:.2f}% "
        "(NOT a final metric)"
    )


    # ========================================================
    # Save model
    # ========================================================

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    joblib.dump(
        model,
        OUTPUT_MODEL,
    )


    # ========================================================
    # Save refit manifest
    # ========================================================

    manifest = {

        "step": (
            "77B"
        ),

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FROZEN_DECODER_REFIT_COMPLETE"
        ),

        "model_file": str(
            OUTPUT_MODEL.relative_to(
                PROJECT_ROOT
            )
        ),

        "model_sha256": (
            sha256_file(
                OUTPUT_MODEL
            )
        ),

        "protocol_file": str(
            PROTOCOL_FILE.relative_to(
                PROJECT_ROOT
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

        "selected_family": (
            EXPECTED_FAMILY
        ),

        "selected_configuration": (
            EXPECTED_CONFIGURATION
        ),

        "pipeline": {

            "scaler": (
                "StandardScaler"
            ),

            "pca_components": (
                PCA_COMPONENTS
            ),

            "pca_solver": (
                "randomized"
            ),

            "pca_random_state": (
                PCA_RANDOM_STATE
            ),

            "classifier": (
                "LogisticRegression"
            ),

            "C": (
                LOGISTIC_C
            ),

            "solver": (
                "lbfgs"
            ),

            "class_weight": (
                "balanced"
            ),

            "max_iter": 5000,

            "classifier_random_state": (
                CLASSIFIER_RANDOM_STATE
            ),
        },

        "development_refit": {

            "logical_prompts": (
                len(
                    all_example_ids
                )
            ),

            "neural_rows": int(
                X_all.shape[
                    0
                ]
            ),

            "feature_dimensions": int(
                X_all.shape[
                    1
                ]
            ),

            "neural_seeds": (
                sorted(
                    actual_seeds
                )
            ),

            "target_classes": sorted(
                np.unique(
                    y_all
                ).astype(str).tolist()
            ),

            "target_distribution": {

                str(
                    token
                ): int(
                    count
                )

                for (
                    token,
                    count,
                )
                in sorted(
                    target_counts.items()
                )
            },
        },

        "development_history": {

            "original_selected_validation_accuracy": float(
                decoder_metrics[
                    "validation_accuracy"
                ]
            ),

            "original_selected_validation_macro_f1": float(
                decoder_metrics[
                    "validation_macro_f1"
                ]
            ),

            "development_diagnostic_test_top1": float(
                decoder_metrics[
                    "test_top1_accuracy"
                ]
            ),

            "development_diagnostic_test_macro_f1": float(
                decoder_metrics[
                    "test_macro_f1"
                ]
            ),

            "development_diagnostic_semantic_accuracy": float(
                decoder_metrics[
                    "semantic_position_accuracy"
                ]
            ),
        },

        "training_sanity_accuracy_not_final_metric": (
            training_accuracy
        ),

        "rules": [

            (
                "No final-holdout labels "
                "were used for model selection."
            ),

            (
                "No decoder family search "
                "was performed in Step 77B."
            ),

            (
                "No hyperparameter search "
                "was performed in Step 77B."
            ),

            (
                "The selected Step-62 "
                "architecture and decoder "
                "were frozen before this refit."
            ),

            (
                "The final holdout must not "
                "be used to retune architecture "
                "or hyperparameters after scoring."
            ),
        ],
    }


    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    OUTPUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # Final report
    # ========================================================

    print()

    print(
        "Created:"
    )


    print(
        "  "
        f"{OUTPUT_MODEL.relative_to(PROJECT_ROOT)}"
    )


    print(
        "  "
        f"{OUTPUT_MANIFEST.relative_to(PROJECT_ROOT)}"
    )


    print()

    print(
        f"Model SHA-256: "
        f"{sha256_file(OUTPUT_MODEL)}"
    )


    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 77B COMPLETE"
    )

    print(
        "===================================================="
    )

    print()


    print(
        "The final frozen decoder is now "
        "trained on all 48 development prompts."
    )


    print()


    print(
        "Next: Step 77C - extract the frozen "
        "Step-62 MaleCNS features for the "
        "24-prompt final holdout."
    )


    print(
        "Do not perform any new architecture "
        "or hyperparameter search after "
        "final scoring."
    )


if __name__ == "__main__":
    main()