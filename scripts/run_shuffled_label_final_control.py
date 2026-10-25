from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Step 78C
# FINAL SHUFFLED-LABEL CONTROL
#
# Purpose:
#   Test whether the real MaleCNS features only look useful
#   because the decoder has access to the true feature/label
#   association.
#
# Procedure:
#   * Use the REAL MaleCNS Step-62 features.
#   * Keep architecture/hyperparameters FIXED:
#       StandardScaler
#       PCA=100
#       Balanced Logistic Regression
#       C=0.1
#   * Refit on ALL 48 development prompts.
#   * Shuffle TARGET LABELS at the unique teacher-forcing
#     sample level, separately within each output position.
#   * The same shuffled label is used for all five neural
#     seeds of a teacher-forcing sample.
#   * Evaluate against the TRUE frozen final-holdout labels.
#
# No final-holdout result is used for tuning.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Real MaleCNS Step-62 feature files
# ============================================================

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

FINAL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_final_holdout.npz"
)


# ============================================================
# Frozen protocol/model manifests
# ============================================================

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

REFIT_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_decoder_refit_manifest.json"
)

REAL_FINAL_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_teacher_forced_metrics.json"
)


# ============================================================
# Output
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "shuffled_label_final_control.json"
)


# ============================================================
# Frozen decoder configuration
# ============================================================

PCA_COMPONENTS = 100
LOGISTIC_C = 0.1

EXPECTED_CONFIGURATION = (
    "Balanced Logistic PCA=100 C=0.1"
)

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Precommitted shuffled-label seeds
# ============================================================

SHUFFLE_SEEDS = list(
    range(
        78010,
        78030,
    )
)


# ============================================================
# Expected data sizes
# ============================================================

EXPECTED_ROWS = {
    "train": 1140,
    "validation": 190,
    "development_test": 190,
    "final": 760,
}

EXPECTED_FEATURE_DIMENSIONS = 41111
EXPECTED_DEVELOPMENT_PROMPTS = 48
EXPECTED_FINAL_PROMPTS = 24
EXPECTED_NEURAL_SEEDS = {
    0,
    1,
    2,
    3,
    4,
}


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
    expected_rows: int,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required feature file: "
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
        "target_positions",
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
            f"{path.name} missing arrays: "
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

        "positions": (
            data[
                "target_positions"
            ].astype(
                np.int32
            )
        ),

        "seeds": (
            data[
                "seeds"
            ].astype(
                np.int32
            )
        ),
    }

    X = result[
        "X"
    ]

    if (
        X.shape
        !=
        (
            expected_rows,
            EXPECTED_FEATURE_DIMENSIONS,
        )
    ):

        raise ValueError(
            f"{path.name}: expected "
            f"({expected_rows}, "
            f"{EXPECTED_FEATURE_DIMENSIONS}), "
            f"got {X.shape}"
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
        "positions",
        "seeds",
    ]:

        if (
            len(
                result[
                    key
                ]
            )
            !=
            expected_rows
        ):

            raise ValueError(
                f"{path.name}: row mismatch "
                f"for {key}"
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

    return result


# ============================================================
# Integrity verification
# ============================================================

def verify_integrity() -> tuple[dict, dict]:

    required = [
        TRAIN_FILE,
        VALIDATION_FILE,
        DEV_TEST_FILE,
        FINAL_FILE,
        PROTOCOL_FILE,
        REFIT_MANIFEST_FILE,
        REAL_FINAL_METRICS_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: "
                f"{path}"
            )

    protocol = load_json(
        PROTOCOL_FILE
    )

    refit_manifest = load_json(
        REFIT_MANIFEST_FILE
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Final protocol is not frozen."
        )

    decoder_selection = (
        protocol.get(
            "decoder_selection",
            {}
        )
    )

    if (
        decoder_selection.get(
            "selected_configuration"
        )
        !=
        EXPECTED_CONFIGURATION
    ):

        raise ValueError(
            "Frozen decoder configuration "
            "changed."
        )

    if (
        refit_manifest.get(
            "selected_configuration"
        )
        !=
        EXPECTED_CONFIGURATION
    ):

        raise ValueError(
            "Step-77B decoder configuration "
            "changed."
        )

    return (
        protocol,
        refit_manifest,
    )


# ============================================================
# Build one position-stratified shuffled label vector
#
# IMPORTANT:
#   Neural seeds are repeated measurements of the SAME logical
#   teacher-forcing sample. We shuffle at the UNIQUE sample_id
#   level, then assign the same shuffled label to all 5 seeds.
# ============================================================

def shuffled_labels(
    sample_ids: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Validate that each unique sample has one target/position.
    # --------------------------------------------------------

    unique_samples = {}

    for (
        sample_id,
        position,
        target,
    ) in zip(
        sample_ids,
        positions,
        targets,
    ):

        key = str(
            sample_id
        )

        pair = (
            int(
                position
            ),
            str(
                target
            ),
        )

        if key in unique_samples:

            if (
                unique_samples[
                    key
                ]
                !=
                pair
            ):

                raise ValueError(
                    "A teacher-forcing sample "
                    "has inconsistent target "
                    "metadata across neural seeds: "
                    f"{key}"
                )

        else:

            unique_samples[
                key
            ] = pair

    # --------------------------------------------------------
    # Group unique samples by output position.
    # --------------------------------------------------------

    samples_by_position = {}

    for (
        sample_id,
        (
            position,
            target,
        ),
    ) in unique_samples.items():

        samples_by_position.setdefault(
            position,
            []
        ).append(
            (
                sample_id,
                target,
            )
        )

    shuffled_map = {}

    for position in sorted(
        samples_by_position
    ):

        rows = sorted(
            samples_by_position[
                position
            ],
            key=lambda item:
                item[
                    0
                ],
        )

        ids = [
            item[
                0
            ]

            for item
            in rows
        ]

        labels = np.asarray(
            [
                item[
                    1
                ]

                for item
                in rows
            ]
        )

        permutation = rng.permutation(
            len(
                labels
            )
        )

        shuffled = labels[
            permutation
        ]

        for (
            sample_id,
            shuffled_target,
        ) in zip(
            ids,
            shuffled,
        ):

            shuffled_map[
                sample_id
            ] = str(
                shuffled_target
            )

    return np.asarray(
        [
            shuffled_map[
                str(
                    sample_id
                )
            ]

            for sample_id
            in sample_ids
        ]
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    positions: np.ndarray,
) -> dict:

    semantic_mask = np.isin(
        positions,
        SEMANTIC_POSITIONS,
    )

    return {
        "top1_accuracy": float(
            accuracy_score(
                true_labels,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "semantic_accuracy": float(
            accuracy_score(
                true_labels[
                    semantic_mask
                ],
                predictions[
                    semantic_mask
                ],
            )
        ),

        "semantic_macro_f1": float(
            f1_score(
                true_labels[
                    semantic_mask
                ],
                predictions[
                    semantic_mask
                ],
                average="macro",
                zero_division=0,
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        "    PyFly STEP 78C - SHUFFLED-LABEL CONTROL"
    )

    print(
        "===================================================="
    )

    print()

    (
        protocol,
        refit_manifest,
    ) = verify_integrity()

    print(
        "Frozen protocol integrity: PASS"
    )

    print(
        f"Frozen decoder config:     "
        f"{EXPECTED_CONFIGURATION}"
    )

    print(
        f"Shuffle repetitions:       "
        f"{len(SHUFFLE_SEEDS)}"
    )

    print(
        f"Shuffle seeds:             "
        f"{SHUFFLE_SEEDS[0]}.."
        f"{SHUFFLE_SEEDS[-1]}"
    )

    # ========================================================
    # Load all real MaleCNS features
    # ========================================================

    train = load_split(
        TRAIN_FILE,
        EXPECTED_ROWS[
            "train"
        ],
    )

    validation = load_split(
        VALIDATION_FILE,
        EXPECTED_ROWS[
            "validation"
        ],
    )

    dev_test = load_split(
        DEV_TEST_FILE,
        EXPECTED_ROWS[
            "development_test"
        ],
    )

    final_data = load_split(
        FINAL_FILE,
        EXPECTED_ROWS[
            "final"
        ],
    )

    # ========================================================
    # Combine ALL 48 development prompts
    # ========================================================

    X_dev = np.concatenate(
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

    y_dev_true = np.concatenate(
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
        ]
    )

    sample_ids_dev = np.concatenate(
        [
            train[
                "sample_ids"
            ],
            validation[
                "sample_ids"
            ],
            dev_test[
                "sample_ids"
            ],
        ]
    )

    positions_dev = np.concatenate(
        [
            train[
                "positions"
            ],
            validation[
                "positions"
            ],
            dev_test[
                "positions"
            ],
        ]
    )

    example_ids_dev = np.concatenate(
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
        ]
    )

    if (
        len(
            set(
                example_ids_dev.tolist()
            )
        )
        !=
        EXPECTED_DEVELOPMENT_PROMPTS
    ):

        raise ValueError(
            "Expected 48 development "
            "logical prompts."
        )

    if (
        len(
            set(
                final_data[
                    "example_ids"
                ].tolist()
            )
        )
        !=
        EXPECTED_FINAL_PROMPTS
    ):

        raise ValueError(
            "Expected 24 final logical prompts."
        )

    if not set(
        example_ids_dev.tolist()
    ).isdisjoint(
        set(
            final_data[
                "example_ids"
            ].tolist()
        )
    ):

        raise ValueError(
            "Development/final prompt leakage."
        )

    print()

    print(
        f"Development rows:          "
        f"{len(X_dev):,}"
    )

    print(
        f"Development prompts:       "
        f"{len(set(example_ids_dev.tolist()))}"
    )

    print(
        f"Unique teacher samples:    "
        f"{len(set(sample_ids_dev.tolist()))}"
    )

    print(
        f"Final rows:                "
        f"{len(final_data['X']):,}"
    )

    print(
        f"Final prompts:             "
        f"{len(set(final_data['example_ids'].tolist()))}"
    )

    # ========================================================
    # Fit UNSUPERVISED preprocessing once.
    #
    # Scaling/PCA do not use target labels, so using one
    # frozen transform across all label shuffles is exactly
    # equivalent while being much faster.
    # ========================================================

    print()

    print(
        "Fitting frozen StandardScaler + PCA=100 "
        "on all development features..."
    )

    scaler = StandardScaler()

    X_dev_scaled = scaler.fit_transform(
        X_dev
    )

    X_final_scaled = scaler.transform(
        final_data[
            "X"
        ]
    )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=42,
    )

    X_dev_pca = pca.fit_transform(
        X_dev_scaled
    )

    X_final_pca = pca.transform(
        X_final_scaled
    )

    print(
        "Preprocessing:             PASS"
    )

    # ========================================================
    # Run 20 shuffled-label refits
    # ========================================================

    results = []

    print()

    print(
        "===================================================="
    )

    print(
        "             SHUFFLED-LABEL REPEATS"
    )

    print(
        "===================================================="
    )

    print()

    for index, shuffle_seed in enumerate(
        SHUFFLE_SEEDS,
        start=1,
    ):

        y_dev_shuffled = (
            shuffled_labels(
                sample_ids=(
                    sample_ids_dev
                ),

                positions=(
                    positions_dev
                ),

                targets=(
                    y_dev_true
                ),

                seed=(
                    shuffle_seed
                ),
            )
        )

        classifier = LogisticRegression(
            C=LOGISTIC_C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

        classifier.fit(
            X_dev_pca,
            y_dev_shuffled,
        )

        predictions = classifier.predict(
            X_final_pca
        ).astype(str)

        metrics = calculate_metrics(
            true_labels=(
                final_data[
                    "targets"
                ]
            ),

            predictions=(
                predictions
            ),

            positions=(
                final_data[
                    "positions"
                ]
            ),
        )

        result = {
            "shuffle_seed": int(
                shuffle_seed
            ),

            **metrics,
        }

        results.append(
            result
        )

        print(
            f"[{index:02}/"
            f"{len(SHUFFLE_SEEDS):02}] "
            f"seed={shuffle_seed} "
            f"Top-1="
            f"{metrics['top1_accuracy'] * 100:6.2f}% "
            f"MacroF1="
            f"{metrics['macro_f1'] * 100:6.2f}% "
            f"Semantic="
            f"{metrics['semantic_accuracy'] * 100:6.2f}%"
        )

    # ========================================================
    # Aggregate shuffled-label control
    # ========================================================

    aggregate = {}

    for metric_name in [
        "top1_accuracy",
        "macro_f1",
        "semantic_accuracy",
        "semantic_macro_f1",
    ]:

        values = np.asarray(
            [
                row[
                    metric_name
                ]

                for row
                in results
            ],
            dtype=float,
        )

        aggregate[
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

            "min": float(
                np.min(
                    values
                )
            ),

            "max": float(
                np.max(
                    values
                )
            ),
        }

    # ========================================================
    # Load real final MaleCNS result for comparison
    # ========================================================

    real_final_json = load_json(
        REAL_FINAL_METRICS_FILE
    )

    real_final = (
        real_final_json.get(
            "aggregate",
            real_final_json,
        )
    )

    # ========================================================
    # Report
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "       FINAL SHUFFLED-LABEL CONTROL RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Shuffled Top-1:       "
        f"{aggregate['top1_accuracy']['mean'] * 100:.2f}% "
        f"+/- "
        f"{aggregate['top1_accuracy']['sd'] * 100:.2f} pp"
    )

    print(
        f"Shuffled Macro F1:    "
        f"{aggregate['macro_f1']['mean'] * 100:.2f}% "
        f"+/- "
        f"{aggregate['macro_f1']['sd'] * 100:.2f} pp"
    )

    print(
        f"Shuffled semantic:    "
        f"{aggregate['semantic_accuracy']['mean'] * 100:.2f}% "
        f"+/- "
        f"{aggregate['semantic_accuracy']['sd'] * 100:.2f} pp"
    )

    print()

    print(
        "Real MaleCNS final:"
    )

    print(
        f"  Top-1:              "
        f"{real_final['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{real_final['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{real_final['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Real MaleCNS minus shuffled-label mean:"
    )

    print(
        f"  Top-1:              "
        f"{(real_final['top1_accuracy'] - aggregate['top1_accuracy']['mean']) * 100:+.2f} pp"
    )

    print(
        f"  Macro F1:           "
        f"{(real_final['macro_f1'] - aggregate['macro_f1']['mean']) * 100:+.2f} pp"
    )

    print(
        f"  Semantic accuracy:  "
        f"{(real_final['semantic_accuracy'] - aggregate['semantic_accuracy']['mean']) * 100:+.2f} pp"
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "step": "78C",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "control": (
            "position_stratified_shuffled_labels"
        ),

        "decoder_configuration": (
            EXPECTED_CONFIGURATION
        ),

        "shuffle_definition": {
            "repetitions": int(
                len(
                    SHUFFLE_SEEDS
                )
            ),

            "shuffle_seeds": (
                SHUFFLE_SEEDS
            ),

            "unit": (
                "unique teacher-forcing sample_id"
            ),

            "stratified_by": (
                "target_position"
            ),

            "same_shuffled_label_across_neural_seeds": (
                True
            ),

            "true_final_labels_used_for_training": (
                False
            ),
        },

        "results": (
            results
        ),

        "aggregate": (
            aggregate
        ),

        "real_malecns_final": (
            real_final
        ),

        "integrity": {
            "protocol_sha256": (
                sha256_file(
                    PROTOCOL_FILE
                )
            ),

            "refit_manifest_sha256": (
                sha256_file(
                    REFIT_MANIFEST_FILE
                )
            ),

            "train_feature_sha256": (
                sha256_file(
                    TRAIN_FILE
                )
            ),

            "validation_feature_sha256": (
                sha256_file(
                    VALIDATION_FILE
                )
            ),

            "development_test_feature_sha256": (
                sha256_file(
                    DEV_TEST_FILE
                )
            ),

            "final_feature_sha256": (
                sha256_file(
                    FINAL_FILE
                )
            ),
        },

        "rules": [
            (
                "The real MaleCNS feature extractor and "
                "frozen Step-62 decoder configuration were "
                "kept fixed."
            ),

            (
                "Only development labels were shuffled."
            ),

            (
                "The final holdout was never used for "
                "hyperparameter or architecture selection."
            ),

            (
                "Labels were shuffled within output position "
                "to preserve fixed-syntax position structure."
            ),

            (
                "Repeated neural seeds of the same logical "
                "teacher-forcing sample received the same "
                "shuffled target."
            ),
        ],
    }

    OUTPUT_FILE.write_text(
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
        f"{OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 78C COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "ALL REQUIRED BASELINE/CONTROL EXPERIMENTS "
        "ARE NOW COMPLETE."
    )

    print()

    print(
        "Next: Phase 2 - Plastic PyFly. "
        "Begin training synapses inside MaleCNS."
    )


if __name__ == "__main__":
    main()
