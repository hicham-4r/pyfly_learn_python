from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# Files
# ============================================================

GENERATOR_TEST_FILE = (
    "artifacts/features/"
    "generator_test.npz"
)

RICH_FAMILY_TEST_FILE = (
    "artifacts/features/"
    "family_rich_test.npz"
)

RICH_FAMILY_MODEL_FILE = (
    "artifacts/models/"
    "family_rich_best.joblib"
)

HIERARCHICAL_BUNDLE_FILE = (
    "artifacts/models/"
    "generator_hierarchical.joblib"
)

OLD_HIERARCHY_METRICS_FILE = (
    "artifacts/metrics/"
    "generator_hierarchical_metrics.json"
)

OUTPUT_MODEL_FILE = Path(
    "artifacts/models/"
    "generator_rich_hierarchy.joblib"
)

OUTPUT_METRICS_FILE = Path(
    "artifacts/metrics/"
    "generator_rich_hierarchy_metrics.json"
)


# ============================================================
# Intent -> broad program family
# ============================================================

def intent_to_family(
    intent: str,
) -> str:

    if intent.startswith(
        "copy_"
    ):
        return "COPY"

    if intent.startswith(
        "sub_"
    ):
        return "SUBTRACT"

    if intent.startswith(
        "div_"
    ):
        return "DIVIDE"

    raise ValueError(
        f"Unknown intent: {intent}"
    )


# ============================================================
# Header
# ============================================================

print()
print(
    "===================================================="
)

print(
    "       PyFly RICH-FAMILY HIERARCHY TEST"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load next-token test features
# ============================================================

generator_data = np.load(
    GENERATOR_TEST_FILE,
    allow_pickle=True,
)


X_generator = (
    generator_data[
        "X"
    ].astype(
        np.float32
    )
)

y_generator = (
    generator_data[
        "targets"
    ].astype(str)
)

generator_example_ids = (
    generator_data[
        "example_ids"
    ].astype(str)
)

generator_intents = (
    generator_data[
        "intents"
    ].astype(str)
)

generator_positions = (
    generator_data[
        "target_positions"
    ].astype(
        np.int32
    )
)

generator_seeds = (
    generator_data[
        "seeds"
    ].astype(
        np.int32
    )
)


print(
    f"Generator test samples: "
    f"{len(X_generator)}"
)


# ============================================================
# Load rich family prompt features
# ============================================================

family_data = np.load(
    RICH_FAMILY_TEST_FILE,
    allow_pickle=True,
)


X_family = (
    family_data[
        "X"
    ].astype(
        np.float32
    )
)

family_targets = (
    family_data[
        "families"
    ].astype(str)
)

family_example_ids = (
    family_data[
        "example_ids"
    ].astype(str)
)

family_seeds = (
    family_data[
        "seeds"
    ].astype(
        np.int32
    )
)


print(
    f"Rich-family samples:    "
    f"{len(X_family)}"
)


# ============================================================
# Load models
# ============================================================

print()
print(
    "Loading rich family classifier..."
)


rich_family_model = joblib.load(
    RICH_FAMILY_MODEL_FILE
)


print(
    "Loading hierarchical token heads..."
)


hierarchy_bundle = joblib.load(
    HIERARCHICAL_BUNDLE_FILE
)


heads = hierarchy_bundle[
    "heads"
]

global_decoder = hierarchy_bundle[
    "global_fallback"
]


print(
    "Models loaded."
)


# ============================================================
# Predict program family using rich prompt features
# ============================================================

rich_family_predictions = (
    rich_family_model.predict(
        X_family
    )
)


rich_family_accuracy = (
    accuracy_score(
        family_targets,
        rich_family_predictions,
    )
)


rich_family_macro_f1 = (
    f1_score(
        family_targets,
        rich_family_predictions,
        average="macro",
        zero_division=0,
    )
)


print()
print(
    "===================================================="
)

print(
    "             RICH FAMILY GATE"
)

print(
    "===================================================="
)

print()


print(
    f"Family accuracy: "
    f"{rich_family_accuracy * 100:.2f}%"
)

print(
    f"Family macro F1: "
    f"{rich_family_macro_f1 * 100:.2f}%"
)


# ============================================================
# Build lookup:
#
# (logical prompt ID, neural seed)
#          ->
# predicted family
#
# There is one rich family feature per logical prompt/seed.
# ============================================================

family_lookup = {}


true_family_lookup = {}


for (
    example_id,
    seed,
    true_family,
    predicted_family,
) in zip(
    family_example_ids,
    family_seeds,
    family_targets,
    rich_family_predictions,
):

    key = (
        str(
            example_id
        ),
        int(
            seed
        ),
    )


    if key in family_lookup:

        raise RuntimeError(
            f"Duplicate family key: "
            f"{key}"
        )


    family_lookup[
        key
    ] = str(
        predicted_family
    )


    true_family_lookup[
        key
    ] = str(
        true_family
    )


print()
print(
    f"Family lookup entries: "
    f"{len(family_lookup)}"
)


# ============================================================
# Sanity-check every generator row has a rich-family prediction
# ============================================================

for example_id, seed in zip(
    generator_example_ids,
    generator_seeds,
):

    key = (
        str(
            example_id
        ),
        int(
            seed
        ),
    )


    assert key in family_lookup, (
        f"Missing rich-family prediction "
        f"for {key}"
    )


print(
    "Generator/family alignment: PASS"
)


# ============================================================
# Predict token from one family-specific head
# ============================================================

def predict_from_family_head(
    feature,
    family,
    position,
):

    family = str(
        family
    )

    position = int(
        position
    )


    # --------------------------------------------------------
    # Unknown family -> fallback
    # --------------------------------------------------------

    if family not in heads:

        prediction = (
            global_decoder.predict(
                feature.reshape(
                    1,
                    -1,
                )
            )[
                0
            ]
        )

        return (
            str(
                prediction
            ),
            True,
        )


    family_heads = heads[
        family
    ]


    # --------------------------------------------------------
    # Unsupported position -> fallback
    # --------------------------------------------------------

    if position not in family_heads:

        prediction = (
            global_decoder.predict(
                feature.reshape(
                    1,
                    -1,
                )
            )[
                0
            ]
        )

        return (
            str(
                prediction
            ),
            True,
        )


    head = family_heads[
        position
    ]


    # --------------------------------------------------------
    # Constant head
    # --------------------------------------------------------

    if head[
        "type"
    ] == "constant":

        return (
            str(
                head[
                    "token"
                ]
            ),
            False,
        )


    # --------------------------------------------------------
    # Learned head
    # --------------------------------------------------------

    prediction = (
        head[
            "model"
        ].predict(
            feature.reshape(
                1,
                -1,
            )
        )[
            0
        ]
    )


    return (
        str(
            prediction
        ),
        False,
    )


# ============================================================
# Global decoder baseline
# ============================================================

global_predictions = (
    global_decoder.predict(
        X_generator
    )
)


global_accuracy = (
    accuracy_score(
        y_generator,
        global_predictions,
    )
)


global_macro_f1 = (
    f1_score(
        y_generator,
        global_predictions,
        average="macro",
        zero_division=0,
    )
)


# ============================================================
# Oracle hierarchy:
#
# Use the TRUE family.
# ============================================================

oracle_predictions = []

oracle_fallbacks = 0


for index in range(
    len(
        X_generator
    )
):

    true_family = (
        intent_to_family(
            generator_intents[
                index
            ]
        )
    )


    token, fallback = (
        predict_from_family_head(
            feature=(
                X_generator[
                    index
                ]
            ),

            family=(
                true_family
            ),

            position=(
                generator_positions[
                    index
                ]
            ),
        )
    )


    oracle_predictions.append(
        token
    )


    if fallback:

        oracle_fallbacks += 1


oracle_predictions = np.asarray(
    oracle_predictions
)


# ============================================================
# NEW hierarchy:
#
# Use RICH predicted family.
# ============================================================

rich_hierarchy_predictions = []

rich_hierarchy_families = []

rich_fallbacks = 0


for index in range(
    len(
        X_generator
    )
):

    key = (
        str(
            generator_example_ids[
                index
            ]
        ),

        int(
            generator_seeds[
                index
            ]
        ),
    )


    predicted_family = (
        family_lookup[
            key
        ]
    )


    token, fallback = (
        predict_from_family_head(
            feature=(
                X_generator[
                    index
                ]
            ),

            family=(
                predicted_family
            ),

            position=(
                generator_positions[
                    index
                ]
            ),
        )
    )


    rich_hierarchy_predictions.append(
        token
    )


    rich_hierarchy_families.append(
        predicted_family
    )


    if fallback:

        rich_fallbacks += 1


rich_hierarchy_predictions = (
    np.asarray(
        rich_hierarchy_predictions
    )
)


rich_hierarchy_families = (
    np.asarray(
        rich_hierarchy_families
    )
)


# ============================================================
# Metric helper
# ============================================================

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


semantic_mask = np.isin(
    generator_positions,
    SEMANTIC_POSITIONS,
)


def evaluate(
    name,
    predictions,
):

    accuracy = (
        accuracy_score(
            y_generator,
            predictions,
        )
    )


    macro_f1 = (
        f1_score(
            y_generator,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


    semantic_accuracy = (
        accuracy_score(
            y_generator[
                semantic_mask
            ],

            predictions[
                semantic_mask
            ],
        )
    )


    semantic_macro_f1 = (
        f1_score(
            y_generator[
                semantic_mask
            ],

            predictions[
                semantic_mask
            ],

            average="macro",
            zero_division=0,
        )
    )


    print()
    print(
        name
    )

    print(
        f"  Top-1:             "
        f"{accuracy * 100:6.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{macro_f1 * 100:6.2f}%"
    )

    print(
        f"  Semantic accuracy: "
        f"{semantic_accuracy * 100:6.2f}%"
    )

    print(
        f"  Semantic macro F1: "
        f"{semantic_macro_f1 * 100:6.2f}%"
    )


    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "semantic_accuracy": float(
            semantic_accuracy
        ),

        "semantic_macro_f1": float(
            semantic_macro_f1
        ),
    }


# ============================================================
# Evaluate all three
# ============================================================

print()
print(
    "===================================================="
)

print(
    "        RICH HIERARCHICAL COMPARISON"
)

print(
    "===================================================="
)


global_result = evaluate(
    "GLOBAL Balanced Logistic:",
    global_predictions,
)


oracle_result = evaluate(
    "ORACLE hierarchy (true family):",
    oracle_predictions,
)


rich_result = evaluate(
    "RICH hierarchy (predicted family):",
    rich_hierarchy_predictions,
)


# ============================================================
# Load old hierarchy result
# ============================================================

old_metrics = None


if Path(
    OLD_HIERARCHY_METRICS_FILE
).exists():

    with open(
        OLD_HIERARCHY_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        old_metrics = json.load(
            file
        )


# ============================================================
# Summary comparison
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "           FAMILY-GATE IMPROVEMENT"
)

print(
    "===================================================="
)

print()


print(
    f"Old family gate:        "
    f"36.67%"
)

print(
    f"Rich family gate:       "
    f"{rich_family_accuracy * 100:.2f}%"
)


if old_metrics is not None:

    old_predicted_accuracy = (
        old_metrics[
            "predicted_hierarchy"
        ][
            "accuracy"
        ]
    )


    old_predicted_f1 = (
        old_metrics[
            "predicted_hierarchy"
        ][
            "macro_f1"
        ]
    )


    old_predicted_semantic = (
        old_metrics[
            "predicted_hierarchy"
        ][
            "semantic_accuracy"
        ]
    )


    print()
    print(
        "OLD predicted hierarchy:"
    )

    print(
        f"  Top-1:             "
        f"{old_predicted_accuracy * 100:6.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{old_predicted_f1 * 100:6.2f}%"
    )

    print(
        f"  Semantic accuracy: "
        f"{old_predicted_semantic * 100:6.2f}%"
    )


print()
print(
    "NEW rich-family hierarchy:"
)

print(
    f"  Top-1:             "
    f"{rich_result['accuracy'] * 100:6.2f}%"
)

print(
    f"  Macro F1:          "
    f"{rich_result['macro_f1'] * 100:6.2f}%"
)

print(
    f"  Semantic accuracy: "
    f"{rich_result['semantic_accuracy'] * 100:6.2f}%"
)


if old_metrics is not None:

    print()
    print(
        f"Top-1 change vs old hierarchy: "
        f"{(rich_result['accuracy'] - old_predicted_accuracy) * 100:+.2f} pp"
    )

    print(
        f"Macro-F1 change: "
        f"{(rich_result['macro_f1'] - old_predicted_f1) * 100:+.2f} pp"
    )

    print(
        f"Semantic change: "
        f"{(rich_result['semantic_accuracy'] - old_predicted_semantic) * 100:+.2f} pp"
    )


print()
print(
    f"Distance from oracle top-1: "
    f"{(oracle_result['accuracy'] - rich_result['accuracy']) * 100:.2f} pp"
)


# ============================================================
# Accuracy by output position
# ============================================================

print()
print(
    "===================================================="
)

print(
    "       RICH HIERARCHY ACCURACY BY POSITION"
)

print(
    "===================================================="
)

print()


position_metrics = {}


for position in sorted(
    np.unique(
        generator_positions
    )
):

    mask = (
        generator_positions
        ==
        position
    )


    position_accuracy = (
        accuracy_score(
            y_generator[
                mask
            ],

            rich_hierarchy_predictions[
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


    print(
        f"Position {position}: "
        f"{position_accuracy * 100:6.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Position 0 detail
# ============================================================

print()
print(
    "===================================================="
)

print(
    "              POSITION-0 DETAIL"
)

print(
    "===================================================="
)

print()


position_zero_mask = (
    generator_positions
    ==
    0
)


for (
    example_id,
    seed,
    intent,
    predicted_family,
    target_token,
    predicted_token,
) in zip(
    generator_example_ids[
        position_zero_mask
    ],

    generator_seeds[
        position_zero_mask
    ],

    generator_intents[
        position_zero_mask
    ],

    rich_hierarchy_families[
        position_zero_mask
    ],

    y_generator[
        position_zero_mask
    ],

    rich_hierarchy_predictions[
        position_zero_mask
    ],
):

    true_family = (
        intent_to_family(
            intent
        )
    )


    mark = (
        "✓"
        if target_token
        ==
        predicted_token
        else "✗"
    )


    family_mark = (
        "✓"
        if true_family
        ==
        predicted_family
        else "✗"
    )


    print(
        f"{mark} "
        f"{example_id:16} "
        f"seed={seed} "
        f"family[{family_mark}] "
        f"{true_family:8}"
        f"→{predicted_family:8} "
        f"target="
        f"{target_token:8} "
        f"pred="
        f"{predicted_token}"
    )


# ============================================================
# Accuracy conditional on family gate being correct
# ============================================================

row_true_families = np.asarray([
    intent_to_family(
        intent
    )
    for intent in generator_intents
])


family_correct_mask = (
    rich_hierarchy_families
    ==
    row_true_families
)


family_wrong_mask = (
    ~family_correct_mask
)


if np.any(
    family_correct_mask
):

    correct_gate_accuracy = (
        accuracy_score(
            y_generator[
                family_correct_mask
            ],

            rich_hierarchy_predictions[
                family_correct_mask
            ],
        )
    )

else:

    correct_gate_accuracy = 0.0


if np.any(
    family_wrong_mask
):

    wrong_gate_accuracy = (
        accuracy_score(
            y_generator[
                family_wrong_mask
            ],

            rich_hierarchy_predictions[
                family_wrong_mask
            ],
        )
    )

else:

    wrong_gate_accuracy = 0.0


print()
print(
    "===================================================="
)

print(
    "          EFFECT OF FAMILY-GATE ERRORS"
)

print(
    "===================================================="
)

print()


print(
    f"Rows with correct family gate: "
    f"{np.sum(family_correct_mask)}"
)

print(
    f"Token accuracy when family correct: "
    f"{correct_gate_accuracy * 100:.2f}%"
)

print()

print(
    f"Rows with wrong family gate: "
    f"{np.sum(family_wrong_mask)}"
)

print(
    f"Token accuracy when family wrong: "
    f"{wrong_gate_accuracy * 100:.2f}%"
)


# ============================================================
# Save combined model bundle for Step 76
# ============================================================

combined_bundle = {
    "rich_family_model": (
        rich_family_model
    ),

    "heads": (
        heads
    ),

    "global_fallback": (
        global_decoder
    ),

    "families": (
        hierarchy_bundle[
            "families"
        ]
    ),

    "family_feature_type": (
        "9_block_position_sensitive_prompt"
    ),
}


OUTPUT_MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    combined_bundle,
    OUTPUT_MODEL_FILE,
)


# ============================================================
# Save metrics
# ============================================================

OUTPUT_METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = {
    "rich_family_gate": {
        "accuracy": float(
            rich_family_accuracy
        ),

        "macro_f1": float(
            rich_family_macro_f1
        ),
    },

    "global_decoder": (
        global_result
    ),

    "oracle_hierarchy": (
        oracle_result
    ),

    "rich_predicted_hierarchy": (
        rich_result
    ),

    "accuracy_by_position": (
        position_metrics
    ),

    "oracle_fallbacks": int(
        oracle_fallbacks
    ),

    "rich_fallbacks": int(
        rich_fallbacks
    ),

    "token_accuracy_when_family_correct": float(
        correct_gate_accuracy
    ),

    "token_accuracy_when_family_wrong": float(
        wrong_gate_accuracy
    ),
}


if old_metrics is not None:

    metrics[
        "old_predicted_hierarchy"
    ] = {
        "accuracy": float(
            old_predicted_accuracy
        ),

        "macro_f1": float(
            old_predicted_f1
        ),

        "semantic_accuracy": float(
            old_predicted_semantic
        ),
    }


with OUTPUT_METRICS_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


print()
print(
    f"Saved combined model: "
    f"{OUTPUT_MODEL_FILE}"
)

print(
    f"Saved metrics: "
    f"{OUTPUT_METRICS_FILE}"
)

print()
print(
    "✅ Rich-family hierarchical evaluation completed"
)