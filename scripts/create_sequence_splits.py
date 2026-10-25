from collections import Counter

from pyfly.dataset import load_dataset
from pyfly.splits import create_splits, save_splits


DATASET = "data/raw/sequence_challenge.jsonl"

OUTPUT = "data/processed/sequence_splits.json"

SEED = 42


examples = load_dataset(
    DATASET
)

example_by_id = {
    example.id: example
    for example in examples
}


splits = create_splits(
    examples,
    seed=SEED,
)


save_splits(
    splits,
    OUTPUT,
    seed=SEED,
)


print()
print("========================================")
print("    PyFly SEQUENCE DATA SPLITS")
print("========================================")
print()


for split_name in [
    "train",
    "validation",
    "test",
]:

    ids = splits[
        split_name
    ]

    intents = [
        example_by_id[example_id].intent
        for example_id in ids
    ]

    counts = Counter(
        intents
    )


    print(
        f"{split_name.upper():12} "
        f"{len(ids)} prompts"
    )


    for intent in sorted(counts):

        print(
            f"  {intent:12} "
            f"{counts[intent]}"
        )


    print()


# ============================================================
# Leakage checks
# ============================================================

train_ids = set(
    splits["train"]
)

validation_ids = set(
    splits["validation"]
)

test_ids = set(
    splits["test"]
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


all_split_ids = (
    train_ids
    | validation_ids
    | test_ids
)


all_dataset_ids = {
    example.id
    for example in examples
}


assert all_split_ids == all_dataset_ids

assert len(train_ids) == 36
assert len(validation_ids) == 6
assert len(test_ids) == 6


print(
    "No train/validation/test leakage: PASS"
)

print(
    "All 48 sequence prompts assigned: PASS"
)

print()

print(
    f"Saved: {OUTPUT}"
)

print()

print(
    "✅ Sequence Challenge split PASSED"
)