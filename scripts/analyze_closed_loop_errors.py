from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Files
# ============================================================

INPUT_FILE = Path(
    "artifacts/metrics/"
    "closed_loop_best_decoder.json"
)

OUTPUT_FILE = Path(
    "artifacts/metrics/"
    "closed_loop_error_analysis.json"
)


# ============================================================
# Header
# ============================================================

print()
print(
    "===================================================="
)

print(
    "        PyFly CLOSED-LOOP ERROR ANALYSIS"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load closed-loop results
# ============================================================

with INPUT_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    data = json.load(
        file
    )


episodes = data[
    "episodes"
]


print(
    f"Episodes loaded: "
    f"{len(episodes)}"
)


# ============================================================
# Helper:
# derive family from example ID
#
# Examples:
#
# copy_a_x_02 -> copy_a_x
# div_ab_01   -> div_ab
# sub_ba_03   -> sub_ba
# ============================================================

def get_family(
    example_id: str,
) -> str:

    parts = (
        example_id.split(
            "_"
        )
    )

    # Remove final numeric paraphrase ID.
    return "_".join(
        parts[:-1]
    )


# ============================================================
# All families
# ============================================================

families = sorted({
    get_family(
        episode[
            "example_id"
        ]
    )

    for episode in episodes
})


print()
print(
    "Program families:"
)


for family in families:

    print(
        f"  {family}"
    )


# ============================================================
# Maximum sequence position
# ============================================================

max_target_length = max(
    len(
        episode[
            "target_tokens"
        ]
    )

    for episode in episodes
)


print()
print(
    f"Maximum target length: "
    f"{max_target_length}"
)


# ============================================================
# Position accuracy overall
# ============================================================

position_correct = Counter()

position_total = Counter()


# ============================================================
# Position accuracy by family
# ============================================================

family_position_correct = defaultdict(
    Counter
)

family_position_total = defaultdict(
    Counter
)


# ============================================================
# First-error statistics
# ============================================================

first_error_counter = Counter()

family_first_error_counter = defaultdict(
    Counter
)


# ============================================================
# Token confusion statistics
# ============================================================

confusions = Counter()

position_confusions = defaultdict(
    Counter
)


# ============================================================
# Correct-token-in-top3 at first error
# ============================================================

first_error_top3_hits = 0

first_error_top3_total = 0


# ============================================================
# Exact / valid / token metrics by family
# ============================================================

family_rows = defaultdict(
    list
)


# ============================================================
# Analyze episodes
# ============================================================

for episode in episodes:

    example_id = episode[
        "example_id"
    ]

    family = get_family(
        example_id
    )


    target = episode[
        "target_tokens"
    ]

    predicted = episode[
        "generated_tokens"
    ]

    trace = episode[
        "generation_trace"
    ]


    family_rows[
        family
    ].append(
        episode
    )


    # --------------------------------------------------------
    # Compare token positions
    # --------------------------------------------------------

    comparison_length = max(
        len(
            target
        ),
        len(
            predicted
        ),
    )


    first_error = None


    for position in range(
        comparison_length
    ):

        target_token = (
            target[
                position
            ]
            if position < len(
                target
            )
            else "<NO_TARGET>"
        )


        predicted_token = (
            predicted[
                position
            ]
            if position < len(
                predicted
            )
            else "<MISSING>"
        )


        position_total[
            position
        ] += 1


        family_position_total[
            family
        ][
            position
        ] += 1


        if (
            target_token
            ==
            predicted_token
        ):

            position_correct[
                position
            ] += 1


            family_position_correct[
                family
            ][
                position
            ] += 1

        else:

            confusions[
                (
                    target_token,
                    predicted_token,
                )
            ] += 1


            position_confusions[
                position
            ][
                (
                    target_token,
                    predicted_token,
                )
            ] += 1


            if (
                first_error
                is None
            ):

                first_error = (
                    position
                )


    # --------------------------------------------------------
    # Exact sequence has no first error
    # --------------------------------------------------------

    if first_error is None:

        first_error_counter[
            "EXACT"
        ] += 1


        family_first_error_counter[
            family
        ][
            "EXACT"
        ] += 1

    else:

        first_error_counter[
            first_error
        ] += 1


        family_first_error_counter[
            family
        ][
            first_error
        ] += 1


        # ----------------------------------------------------
        # Was the correct token in top-3 at the first error?
        # ----------------------------------------------------

        if first_error < len(
            trace
        ):

            target_token = (
                target[
                    first_error
                ]
                if first_error < len(
                    target
                )
                else "<NO_TARGET>"
            )


            top3 = [
                item[
                    "token"
                ]

                for item in (
                    trace[
                        first_error
                    ][
                        "top3"
                    ]
                )
            ]


            first_error_top3_total += 1


            if target_token in top3:

                first_error_top3_hits += 1


# ============================================================
# Overall position accuracy
# ============================================================

print()
print(
    "===================================================="
)

print(
    "          OVERALL ACCURACY BY POSITION"
)

print(
    "===================================================="
)

print()


overall_position_metrics = {}


for position in sorted(
    position_total.keys()
):

    correct = (
        position_correct[
            position
        ]
    )

    total = (
        position_total[
            position
        ]
    )


    accuracy = (
        correct
        /
        total
    )


    overall_position_metrics[
        position
    ] = float(
        accuracy
    )


    print(
        f"Position {position}: "
        f"{accuracy * 100:6.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# Accuracy by family AND position
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "        ACCURACY BY FAMILY AND POSITION"
)

print(
    "===================================================="
)

print()


# Header
header = (
    f"{'Family':15}"
)


for position in range(
    max_target_length
):

    header += (
        f" P{position:>2}"
    )


print(
    header
)

print(
    "-" * len(
        header
    )
)


family_position_metrics = {}


for family in families:

    line = (
        f"{family:15}"
    )


    family_position_metrics[
        family
    ] = {}


    for position in range(
        max_target_length
    ):

        total = (
            family_position_total[
                family
            ][
                position
            ]
        )


        correct = (
            family_position_correct[
                family
            ][
                position
            ]
        )


        if total == 0:

            line += (
                f" {'--':>4}"
            )

            continue


        accuracy = (
            correct
            /
            total
        )


        family_position_metrics[
            family
        ][
            position
        ] = float(
            accuracy
        )


        line += (
            f" "
            f"{accuracy * 100:3.0f}%"
        )


    print(
        line
    )


# ============================================================
# Family summary
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "              FAMILY SUMMARY"
)

print(
    "===================================================="
)

print()


family_summary = {}


for family in families:

    rows = (
        family_rows[
            family
        ]
    )


    exact_accuracy = float(
        np.mean([
            row[
                "exact_program"
            ]

            for row in rows
        ])
    )


    valid_rate = float(
        np.mean([
            row[
                "valid_program"
            ]

            for row in rows
        ])
    )


    mean_token_accuracy = float(
        np.mean([
            row[
                "token_accuracy"
            ]

            for row in rows
        ])
    )


    family_summary[
        family
    ] = {
        "exact_accuracy": (
            exact_accuracy
        ),

        "valid_rate": (
            valid_rate
        ),

        "mean_token_accuracy": (
            mean_token_accuracy
        ),
    }


    print(
        f"{family:15} "
        f"exact="
        f"{exact_accuracy * 100:6.2f}% "
        f"token="
        f"{mean_token_accuracy * 100:6.2f}% "
        f"valid="
        f"{valid_rate * 100:6.2f}%"
    )


# ============================================================
# First-error positions
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "             FIRST ERROR POSITION"
)

print(
    "===================================================="
)

print()


total_episodes = len(
    episodes
)


first_error_metrics = {}


for key, count in sorted(
    first_error_counter.items(),

    key=lambda item: (
        999
        if item[0] == "EXACT"
        else item[0]
    ),
):

    rate = (
        count
        /
        total_episodes
    )


    first_error_metrics[
        str(
            key
        )
    ] = {
        "count": int(
            count
        ),

        "rate": float(
            rate
        ),
    }


    if key == "EXACT":

        label = (
            "No error / exact"
        )

    else:

        label = (
            f"Position {key}"
        )


    print(
        f"{label:20} "
        f"{count:3} episodes "
        f"({rate * 100:6.2f}%)"
    )


# ============================================================
# First-error by family
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "          FIRST ERROR POSITION BY FAMILY"
)

print(
    "===================================================="
)

print()


family_first_error_metrics = {}


for family in families:

    counter = (
        family_first_error_counter[
            family
        ]
    )


    print(
        f"{family}:"
    )


    family_first_error_metrics[
        family
    ] = {}


    for key, count in sorted(
        counter.items(),

        key=lambda item: (
            999
            if item[0] == "EXACT"
            else item[0]
        ),
    ):

        family_first_error_metrics[
            family
        ][
            str(
                key
            )
        ] = int(
            count
        )


        if key == "EXACT":

            label = (
                "exact"
            )

        else:

            label = (
                f"position {key}"
            )


        print(
            f"  {label:12} "
            f"{count}"
        )


# ============================================================
# Top confusions
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "              TOP TOKEN CONFUSIONS"
)

print(
    "===================================================="
)

print()


confusion_output = []


for (
    target_token,
    predicted_token,
), count in confusions.most_common(
    20
):

    confusion_output.append(
        {
            "target": (
                target_token
            ),

            "predicted": (
                predicted_token
            ),

            "count": int(
                count
            ),
        }
    )


    print(
        f"{target_token:12} "
        f"-> "
        f"{predicted_token:12} "
        f"{count:3}"
    )


# ============================================================
# Confusions by position
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "         MOST COMMON CONFUSION BY POSITION"
)

print(
    "===================================================="
)

print()


position_confusion_output = {}


for position in sorted(
    position_confusions.keys()
):

    counter = (
        position_confusions[
            position
        ]
    )


    if not counter:

        continue


    print(
        f"Position {position}:"
    )


    position_confusion_output[
        position
    ] = []


    for (
        target_token,
        predicted_token,
    ), count in counter.most_common(
        5
    ):

        position_confusion_output[
            position
        ].append(
            {
                "target": (
                    target_token
                ),

                "predicted": (
                    predicted_token
                ),

                "count": int(
                    count
                ),
            }
        )


        print(
            f"  "
            f"{target_token:12} "
            f"-> "
            f"{predicted_token:12} "
            f"{count}"
        )


# ============================================================
# First-error Top-3 diagnostic
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "          FIRST-ERROR TOP-3 DIAGNOSTIC"
)

print(
    "===================================================="
)

print()


if (
    first_error_top3_total
    >
    0
):

    top3_rate = (
        first_error_top3_hits
        /
        first_error_top3_total
    )


    print(
        f"Incorrect top-1 decisions analyzed: "
        f"{first_error_top3_total}"
    )

    print(
        f"Correct token still in top-3:       "
        f"{first_error_top3_hits}"
    )

    print(
        f"First-error top-3 recovery rate:     "
        f"{top3_rate * 100:.2f}%"
    )

else:

    top3_rate = 0.0

    print(
        "No first-error cases found."
    )


# ============================================================
# Special diagnostic:
# first token / BOS
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "            FIRST-TOKEN DIAGNOSTIC"
)

print(
    "===================================================="
)

print()


first_token_by_family = {}


for family in families:

    rows = (
        family_rows[
            family
        ]
    )


    correct = 0

    total = 0

    predictions = Counter()


    for row in rows:

        target = row[
            "target_tokens"
        ]

        predicted = row[
            "generated_tokens"
        ]


        if not target:

            continue


        total += 1


        predicted_token = (
            predicted[
                0
            ]
            if predicted
            else "<MISSING>"
        )


        predictions[
            predicted_token
        ] += 1


        if (
            predicted_token
            ==
            target[
                0
            ]
        ):

            correct += 1


    accuracy = (
        correct
        /
        total
        if total
        else 0.0
    )


    first_token_by_family[
        family
    ] = {
        "accuracy": float(
            accuracy
        ),

        "predictions": {
            token: int(
                count
            )

            for token, count
            in predictions.items()
        },
    }


    common = ", ".join(
        f"{token}:{count}"

        for token, count
        in predictions.most_common()
    )


    print(
        f"{family:15} "
        f"accuracy="
        f"{accuracy * 100:6.2f}% "
        f"predictions=[{common}]"
    )


# ============================================================
# Interpretation hints
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "             AUTOMATIC DIAGNOSTIC"
)

print(
    "===================================================="
)

print()


copy_families = [
    family
    for family in families
    if family.startswith(
        "copy_"
    )
]


arithmetic_families = [
    family
    for family in families
    if (
        family.startswith(
            "div_"
        )
        or
        family.startswith(
            "sub_"
        )
    )
]


copy_first_token_scores = [
    first_token_by_family[
        family
    ][
        "accuracy"
    ]

    for family in copy_families
]


arithmetic_first_token_scores = [
    first_token_by_family[
        family
    ][
        "accuracy"
    ]

    for family in arithmetic_families
]


copy_first_token_accuracy = (
    float(
        np.mean(
            copy_first_token_scores
        )
    )

    if copy_first_token_scores
    else 0.0
)


arithmetic_first_token_accuracy = (
    float(
        np.mean(
            arithmetic_first_token_scores
        )
    )

    if arithmetic_first_token_scores
    else 0.0
)


print(
    f"Copy-family first-token accuracy: "
    f"{copy_first_token_accuracy * 100:.2f}%"
)

print(
    f"Arithmetic first-token accuracy:  "
    f"{arithmetic_first_token_accuracy * 100:.2f}%"
)


print()


if (
    copy_first_token_accuracy
    <
    0.50
):

    print(
        "⚠️ Copy programs frequently fail at "
        "the first generated token."
    )


if (
    arithmetic_first_token_accuracy
    >
    copy_first_token_accuracy
):

    print(
        "✅ Arithmetic program-family selection "
        "is easier than copy-family selection."
    )


if (
    top3_rate
    >= 0.70
):

    print(
        "✅ Most first errors still contain the "
        "correct token in the top-3."
    )

    print(
        "   This suggests ranking / decoding is "
        "still a major bottleneck."
    )

else:

    print(
        "⚠️ Many first errors do not contain the "
        "correct token in the top-3."
    )

    print(
        "   The neural representation itself may "
        "need improvement."
    )


# ============================================================
# Save
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output = {
    "episodes": int(
        total_episodes
    ),

    "overall_position_accuracy": (
        overall_position_metrics
    ),

    "family_position_accuracy": (
        family_position_metrics
    ),

    "family_summary": (
        family_summary
    ),

    "first_error_distribution": (
        first_error_metrics
    ),

    "family_first_error_distribution": (
        family_first_error_metrics
    ),

    "top_confusions": (
        confusion_output
    ),

    "position_confusions": (
        position_confusion_output
    ),

    "first_error_top3_rate": float(
        top3_rate
    ),

    "first_token_by_family": (
        first_token_by_family
    ),

    "copy_first_token_accuracy": float(
        copy_first_token_accuracy
    ),

    "arithmetic_first_token_accuracy": float(
        arithmetic_first_token_accuracy
    ),
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


print()
print(
    f"Saved: "
    f"{OUTPUT_FILE}"
)

print()

print(
    "✅ Closed-loop error analysis completed"
)