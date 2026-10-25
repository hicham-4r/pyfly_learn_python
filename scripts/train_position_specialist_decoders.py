from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Files
# ============================================================

TRAIN_FILE = (
    "artifacts/features/"
    "generator_train.npz"
)

VALIDATION_FILE = (
    "artifacts/features/"
    "generator_validation.npz"
)

TEST_FILE = (
    "artifacts/features/"
    "generator_test.npz"
)


MODEL_OUTPUT = Path(
    "artifacts/models/"
    "generator_position_specialists.joblib"
)

METRICS_OUTPUT = Path(
    "artifacts/metrics/"
    "generator_position_specialists_metrics.json"
)


GLOBAL_METRICS_FILE = Path(
    "artifacts/metrics/"
    "generator_decoder_comparison.json"
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


C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# ============================================================
# Load feature split
# ============================================================

def load_split(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    return {
        "X": data[
            "X"
        ].astype(
            np.float32
        ),

        "targets": data[
            "targets"
        ].astype(str),

        "example_ids": data[
            "example_ids"
        ].astype(str),

        "positions": data[
            "target_positions"
        ].astype(
            np.int32
        ),

        "feedback": data[
            "feedback_tokens"
        ].astype(str),

        "seeds": data[
            "seeds"
        ].astype(
            np.int32
        ),
    }


# ============================================================
# Header
# ============================================================

print()
print(
    "===================================================="
)

print(
    "    PyFly POSITION-SPECIALIST DECODER TRAINING"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load data
# ============================================================

train = load_split(
    TRAIN_FILE
)

validation = load_split(
    VALIDATION_FILE
)

test = load_split(
    TEST_FILE
)


print(
    f"Train:      "
    f"{train['X'].shape}"
)

print(
    f"Validation: "
    f"{validation['X'].shape}"
)

print(
    f"Test:       "
    f"{test['X'].shape}"
)


# ============================================================
# Logical prompt leakage check
# ============================================================

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

test_ids = set(
    test[
        "example_ids"
    ].tolist()
)


assert train_ids.isdisjoint(
    validation_ids
)

assert train_ids.isdisjoint(
    test_ids
)

assert validation_ids.isdisjoint(
    test_ids
)


print()
print(
    "Logical-prompt leakage check: PASS"
)


# ============================================================
# Determine positions
# ============================================================

positions = sorted(
    set(
        train[
            "positions"
        ].tolist()
    )
)


print()
print(
    f"Generation positions: "
    f"{positions}"
)


# ============================================================
# Model factory
# ============================================================

def make_model(
    components,
    c_value,
):

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),

            (
                "pca",
                PCA(
                    n_components=components,

                    svd_solver=(
                        "randomized"
                    ),

                    random_state=42,
                ),
            ),

            (
                "classifier",
                LogisticRegression(
                    C=c_value,

                    solver="lbfgs",

                    class_weight=(
                        "balanced"
                    ),

                    max_iter=5000,

                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# Train one position specialist
# ============================================================

def train_position_specialist(
    position,
):

    train_mask = (
        train[
            "positions"
        ]
        ==
        position
    )

    validation_mask = (
        validation[
            "positions"
        ]
        ==
        position
    )

    test_mask = (
        test[
            "positions"
        ]
        ==
        position
    )


    X_train = train[
        "X"
    ][
        train_mask
    ]

    y_train = train[
        "targets"
    ][
        train_mask
    ]


    X_val = validation[
        "X"
    ][
        validation_mask
    ]

    y_val = validation[
        "targets"
    ][
        validation_mask
    ]


    X_test = test[
        "X"
    ][
        test_mask
    ]

    y_test = test[
        "targets"
    ][
        test_mask
    ]


    classes = sorted(
        np.unique(
            y_train
        )
    )


    print()
    print(
        "----------------------------------------------------"
    )

    print(
        f"POSITION {position}"
    )

    print(
        "----------------------------------------------------"
    )

    print(
        f"Train samples:      "
        f"{len(y_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(y_val)}"
    )

    print(
        f"Test samples:       "
        f"{len(y_test)}"
    )

    print(
        f"Classes:            "
        f"{classes}"
    )


    # ========================================================
    # SINGLE-CLASS POSITION
    #
    # Examples:
    #
    # position 1 -> ASSIGN
    # position 5 -> NEWLINE
    # position 6 -> EOS
    #
    # No classifier is necessary.
    # ========================================================

    if len(
        classes
    ) == 1:

        constant_token = (
            classes[
                0
            ]
        )


        predictions = np.full(
            len(
                y_test
            ),

            constant_token,
        )


        accuracy = accuracy_score(
            y_test,
            predictions,
        )


        macro_f1 = f1_score(
            y_test,
            predictions,

            average="macro",

            zero_division=0,
        )


        print()
        print(
            f"Constant specialist: "
            f"{constant_token}"
        )

        print(
            f"Test accuracy: "
            f"{accuracy * 100:.2f}%"
        )


        return {
            "type": (
                "constant"
            ),

            "constant_token": (
                constant_token
            ),

            "position": int(
                position
            ),

            "classes": (
                classes
            ),

            "validation_accuracy": 1.0,

            "validation_macro_f1": 1.0,

            "test_accuracy": float(
                accuracy
            ),

            "test_macro_f1": float(
                macro_f1
            ),

            "test_indices": np.where(
                test_mask
            )[
                0
            ],
        }


    # ========================================================
    # Multi-class specialist
    # ========================================================

    best = None


    for components in (
        PCA_COMPONENTS
    ):

        max_components = min(
            X_train.shape[
                0
            ],

            X_train.shape[
                1
            ],
        )


        if components >= (
            max_components
        ):

            continue


        for c_value in (
            C_VALUES
        ):

            model = make_model(
                components=components,
                c_value=c_value,
            )


            model.fit(
                X_train,
                y_train,
            )


            val_predictions = (
                model.predict(
                    X_val
                )
            )


            val_accuracy = (
                accuracy_score(
                    y_val,
                    val_predictions,
                )
            )


            val_macro_f1 = (
                f1_score(
                    y_val,
                    val_predictions,

                    average="macro",

                    zero_division=0,
                )
            )


            print(
                f"PCA={components:2} "
                f"C={c_value:7.3f} "
                f"val_acc="
                f"{val_accuracy * 100:6.2f}% "
                f"val_F1="
                f"{val_macro_f1 * 100:6.2f}%"
            )


            # ------------------------------------------------
            # Macro F1 first.
            # Accuracy breaks ties.
            # ------------------------------------------------

            if (
                best is None
                or
                val_macro_f1
                >
                best[
                    "validation_macro_f1"
                ]
                or (
                    val_macro_f1
                    ==
                    best[
                        "validation_macro_f1"
                    ]
                    and
                    val_accuracy
                    >
                    best[
                        "validation_accuracy"
                    ]
                )
            ):

                best = {
                    "components": int(
                        components
                    ),

                    "c_value": float(
                        c_value
                    ),

                    "validation_accuracy": float(
                        val_accuracy
                    ),

                    "validation_macro_f1": float(
                        val_macro_f1
                    ),
                }


    # ========================================================
    # Refit selected specialist on train + validation
    # ========================================================

    print()
    print(
        f"Selected PCA: "
        f"{best['components']}"
    )

    print(
        f"Selected C:   "
        f"{best['c_value']}"
    )

    print(
        f"Validation accuracy: "
        f"{best['validation_accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation macro F1: "
        f"{best['validation_macro_f1'] * 100:.2f}%"
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


    final_model = make_model(
        components=(
            best[
                "components"
            ]
        ),

        c_value=(
            best[
                "c_value"
            ]
        ),
    )


    final_model.fit(
        X_final_train,
        y_final_train,
    )


    test_predictions = (
        final_model.predict(
            X_test
        )
    )


    test_accuracy = (
        accuracy_score(
            y_test,
            test_predictions,
        )
    )


    test_macro_f1 = (
        f1_score(
            y_test,
            test_predictions,

            average="macro",

            zero_division=0,
        )
    )


    print()
    print(
        f"TEST accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"TEST macro F1: "
        f"{test_macro_f1 * 100:.2f}%"
    )


    print()
    print(
        classification_report(
            y_test,
            test_predictions,

            zero_division=0,
        )
    )


    return {
        "type": (
            "model"
        ),

        "model": (
            final_model
        ),

        "position": int(
            position
        ),

        "classes": (
            classes
        ),

        "pca_components": int(
            best[
                "components"
            ]
        ),

        "c_value": float(
            best[
                "c_value"
            ]
        ),

        "validation_accuracy": float(
            best[
                "validation_accuracy"
            ]
        ),

        "validation_macro_f1": float(
            best[
                "validation_macro_f1"
            ]
        ),

        "test_accuracy": float(
            test_accuracy
        ),

        "test_macro_f1": float(
            test_macro_f1
        ),

        "test_indices": np.where(
            test_mask
        )[
            0
        ],
    }


# ============================================================
# Train all specialists
# ============================================================

specialists = {}


for position in positions:

    specialists[
        position
    ] = train_position_specialist(
        position
    )


# ============================================================
# Assemble predictions over the FULL test set
# ============================================================

full_predictions = np.empty(
    len(
        test[
            "targets"
        ]
    ),

    dtype=object,
)


for position, specialist in (
    specialists.items()
):

    indices = specialist[
        "test_indices"
    ]


    if specialist[
        "type"
    ] == "constant":

        full_predictions[
            indices
        ] = specialist[
            "constant_token"
        ]


    else:

        position_features = (
            test[
                "X"
            ][
                indices
            ]
        )


        predictions = specialist[
            "model"
        ].predict(
            position_features
        )


        full_predictions[
            indices
        ] = predictions


# ============================================================
# Overall specialist next-token performance
# ============================================================

y_test_all = test[
    "targets"
]


overall_accuracy = accuracy_score(
    y_test_all,
    full_predictions,
)


overall_macro_f1 = f1_score(
    y_test_all,
    full_predictions,

    average="macro",

    zero_division=0,
)


# ============================================================
# Semantic positions
# ============================================================

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


semantic_mask = np.isin(
    test[
        "positions"
    ],
    SEMANTIC_POSITIONS,
)


semantic_accuracy = accuracy_score(
    y_test_all[
        semantic_mask
    ],

    full_predictions[
        semantic_mask
    ],
)


semantic_macro_f1 = f1_score(
    y_test_all[
        semantic_mask
    ],

    full_predictions[
        semantic_mask
    ],

    average="macro",

    zero_division=0,
)


# ============================================================
# Overall results
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "      POSITION-SPECIALIST TEST RESULT"
)

print(
    "===================================================="
)

print()


print(
    f"Overall top-1 accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Overall macro F1:       "
    f"{overall_macro_f1 * 100:.2f}%"
)

print(
    f"Semantic accuracy:      "
    f"{semantic_accuracy * 100:.2f}%"
)

print(
    f"Semantic macro F1:      "
    f"{semantic_macro_f1 * 100:.2f}%"
)


# ============================================================
# Position comparison
# ============================================================

print()
print(
    "===================================================="
)

print(
    "          SPECIALIST ACCURACY BY POSITION"
)

print(
    "===================================================="
)

print()


position_metrics = {}


for position in positions:

    mask = (
        test[
            "positions"
        ]
        ==
        position
    )


    position_accuracy = (
        accuracy_score(
            y_test_all[
                mask
            ],

            full_predictions[
                mask
            ],
        )
    )


    position_metrics[
        int(
            position
        )
    ] = float(
        position_accuracy
    )


    specialist = (
        specialists[
            position
        ]
    )


    print(
        f"Position {position}: "
        f"{position_accuracy * 100:6.2f}% "
        f"classes="
        f"{specialist['classes']}"
    )


# ============================================================
# Load current global decoder result
# ============================================================

global_metrics = None


if GLOBAL_METRICS_FILE.exists():

    with GLOBAL_METRICS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        global_metrics = json.load(
            file
        )


# ============================================================
# Direct comparison
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "          GLOBAL vs SPECIALIST DECODERS"
)

print(
    "===================================================="
)

print()


if global_metrics is not None:

    global_top1 = (
        global_metrics[
            "test_top1_accuracy"
        ]
    )

    global_macro_f1 = (
        global_metrics[
            "test_macro_f1"
        ]
    )

    global_semantic = (
        global_metrics[
            "semantic_position_accuracy"
        ]
    )


    print(
        "Current global Balanced Logistic:"
    )

    print(
        f"  Top-1:             "
        f"{global_top1 * 100:6.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{global_macro_f1 * 100:6.2f}%"
    )

    print(
        f"  Semantic accuracy: "
        f"{global_semantic * 100:6.2f}%"
    )


print()
print(
    "Position specialists:"
)

print(
    f"  Top-1:             "
    f"{overall_accuracy * 100:6.2f}%"
)

print(
    f"  Macro F1:          "
    f"{overall_macro_f1 * 100:6.2f}%"
)

print(
    f"  Semantic accuracy: "
    f"{semantic_accuracy * 100:6.2f}%"
)


if global_metrics is not None:

    print()

    print(
        f"Top-1 change: "
        f"{(overall_accuracy - global_top1) * 100:+.2f} pp"
    )

    print(
        f"Macro-F1 change: "
        f"{(overall_macro_f1 - global_macro_f1) * 100:+.2f} pp"
    )

    print(
        f"Semantic change: "
        f"{(semantic_accuracy - global_semantic) * 100:+.2f} pp"
    )


# ============================================================
# Show predictions
# ============================================================

print()
print(
    "===================================================="
)

print(
    "              SAMPLE PREDICTIONS"
)

print(
    "===================================================="
)

print()


for index in range(
    min(
        60,
        len(
            y_test_all
        ),
    )
):

    true_token = (
        y_test_all[
            index
        ]
    )


    predicted_token = (
        full_predictions[
            index
        ]
    )


    mark = (
        "✓"
        if true_token
        ==
        predicted_token
        else "✗"
    )


    print(
        f"{mark} "
        f"{test['example_ids'][index]:16} "
        f"seed="
        f"{test['seeds'][index]} "
        f"pos="
        f"{test['positions'][index]} "
        f"target="
        f"{true_token:10} "
        f"pred="
        f"{predicted_token}"
    )


# ============================================================
# Save only what closed-loop inference needs
# ============================================================

model_bundle = {
    "positions": positions,

    "specialists": {},
}


for position, specialist in (
    specialists.items()
):

    if specialist[
        "type"
    ] == "constant":

        model_bundle[
            "specialists"
        ][
            position
        ] = {
            "type": (
                "constant"
            ),

            "token": (
                specialist[
                    "constant_token"
                ]
            ),
        }


    else:

        model_bundle[
            "specialists"
        ][
            position
        ] = {
            "type": (
                "model"
            ),

            "model": (
                specialist[
                    "model"
                ]
            ),

            "classes": (
                specialist[
                    "classes"
                ]
            ),
        }


MODEL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    model_bundle,
    MODEL_OUTPUT,
)


# ============================================================
# Save metrics
# ============================================================

METRICS_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


specialist_metrics = {}


for position, specialist in (
    specialists.items()
):

    record = {
        "type": specialist[
            "type"
        ],

        "classes": specialist[
            "classes"
        ],

        "validation_accuracy": float(
            specialist[
                "validation_accuracy"
            ]
        ),

        "validation_macro_f1": float(
            specialist[
                "validation_macro_f1"
            ]
        ),

        "test_accuracy": float(
            specialist[
                "test_accuracy"
            ]
        ),

        "test_macro_f1": float(
            specialist[
                "test_macro_f1"
            ]
        ),
    }


    if specialist[
        "type"
    ] == "constant":

        record[
            "constant_token"
        ] = specialist[
            "constant_token"
        ]

    else:

        record[
            "pca_components"
        ] = int(
            specialist[
                "pca_components"
            ]
        )

        record[
            "c_value"
        ] = float(
            specialist[
                "c_value"
            ]
        )


    specialist_metrics[
        str(
            position
        )
    ] = record


metrics = {
    "overall_test_accuracy": float(
        overall_accuracy
    ),

    "overall_test_macro_f1": float(
        overall_macro_f1
    ),

    "semantic_accuracy": float(
        semantic_accuracy
    ),

    "semantic_macro_f1": float(
        semantic_macro_f1
    ),

    "accuracy_by_position": (
        position_metrics
    ),

    "specialists": (
        specialist_metrics
    ),
}


if global_metrics is not None:

    metrics[
        "global_decoder"
    ] = {
        "top1": float(
            global_top1
        ),

        "macro_f1": float(
            global_macro_f1
        ),

        "semantic_accuracy": float(
            global_semantic
        ),

        "top1_change": float(
            overall_accuracy
            -
            global_top1
        ),

        "macro_f1_change": float(
            overall_macro_f1
            -
            global_macro_f1
        ),

        "semantic_change": float(
            semantic_accuracy
            -
            global_semantic
        ),
    }


with METRICS_OUTPUT.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# Final
# ============================================================

print()
print(
    f"Saved specialist bundle: "
    f"{MODEL_OUTPUT}"
)

print(
    f"Saved metrics: "
    f"{METRICS_OUTPUT}"
)

print()
print(
    "✅ Position-specialist decoder training completed"
)