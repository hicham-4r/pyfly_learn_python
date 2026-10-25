import json
from collections import Counter

from pyfly.dataset import load_dataset

from pyfly.generator_dataset import (
    build_generator_examples,
    save_generator_jsonl,
)


# ============================================================
# Files
# ============================================================

DATASET_FILE = (
    "data/raw/"
    "sequence_challenge.jsonl"
)

SPLIT_FILE = (
    "data/processed/"
    "sequence_splits.json"
)


OUTPUT_FILES = {
    "train": (
        "data/processed/"
        "generator_train.jsonl"
    ),

    "validation": (
        "data/processed/"
        "generator_validation.jsonl"
    ),

    "test": (
        "data/processed/"
        "generator_test.jsonl"
    ),
}


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


# ============================================================
# Load split assignments
# ============================================================

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8",
) as file:

    split_data = json.load(
        file
    )


print()
print("============================================")
print("     PyFly GENERATOR DATASET CREATION")
print("============================================")
print()


# ============================================================
# Build each split
# ============================================================

all_generator_samples = {}


for split_name in [
    "train",
    "validation",
    "test",
]:

    samples = []


    logical_ids = split_data[
        split_name
    ]


    for example_id in logical_ids:

        example = example_by_id[
            example_id
        ]


        example_samples = (
            build_generator_examples(
                example=example,
                split=split_name,
            )
        )


        samples.extend(
            example_samples
        )


    all_generator_samples[
        split_name
    ] = samples


    save_generator_jsonl(
        samples,
        OUTPUT_FILES[
            split_name
        ],
    )


# ============================================================
# Statistics
# ============================================================

for split_name in [
    "train",
    "validation",
    "test",
]:

    samples = (
        all_generator_samples[
            split_name
        ]
    )


    target_counts = Counter(
        sample.target_token
        for sample in samples
    )


    logical_prompt_ids = {
        sample.example_id
        for sample in samples
    }


    print()
    print(
        "============================================"
    )

    print(
        f" {split_name.upper()}"
    )

    print(
        "============================================"
    )

    print(
        f"Logical prompts: "
        f"{len(logical_prompt_ids)}"
    )

    print(
        f"Next-token samples: "
        f"{len(samples)}"
    )

    print()
    print(
        "Target-token counts:"
    )


    for token in sorted(
        target_counts
    ):

        print(
            f"  {token:12} "
            f"{target_counts[token]}"
        )


# ============================================================
# Leakage checks
# ============================================================

train_ids = {
    sample.example_id
    for sample in (
        all_generator_samples[
            "train"
        ]
    )
}

validation_ids = {
    sample.example_id
    for sample in (
        all_generator_samples[
            "validation"
        ]
    )
}

test_ids = {
    sample.example_id
    for sample in (
        all_generator_samples[
            "test"
        ]
    )
}


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
    "Logical prompt leakage: PASS"
)


# ============================================================
# Show one example
# ============================================================

print()
print("============================================")
print("        EXAMPLE TEACHER FORCING")
print("============================================")
print()


example_id = "sub_ab_01"


example_samples = [
    sample
    for sample in (
        all_generator_samples[
            "train"
        ]
        +
        all_generator_samples[
            "validation"
        ]
        +
        all_generator_samples[
            "test"
        ]
    )
    if sample.example_id == example_id
]


if example_samples:

    print(
        f"Prompt: "
        f"{example_samples[0].prompt_tokens}"
    )

    print(
        f"Code:   "
        f"{example_samples[0].code}"
    )

    print()


    for sample in example_samples:

        print(
            f"{str(sample.context_tokens):55} "
            f"-> "
            f"{sample.target_token}"
        )


print()
print("============================================")

print(
    f"Train next-token samples:      "
    f"{len(all_generator_samples['train'])}"
)

print(
    f"Validation next-token samples: "
    f"{len(all_generator_samples['validation'])}"
)

print(
    f"Test next-token samples:       "
    f"{len(all_generator_samples['test'])}"
)

print("============================================")

print()
print(
    "✅ Teacher-forcing generator dataset PASSED"
)