from collections import Counter

from pyfly.dataset import load_dataset


DATASET = "data/raw/sequence_challenge.jsonl"


examples = load_dataset(DATASET)

by_id = {
    example.id: example
    for example in examples
}


print()
print("========================================")
print("     PyFly SEQUENCE CHALLENGE")
print("========================================")
print()

print(f"Total examples: {len(examples)}")

intents = Counter(
    example.intent
    for example in examples
)

print(f"Classes: {len(intents)}")

print()

for intent in sorted(intents):

    print(
        f"{intent:12} : {intents[intent]}"
    )


assert len(examples) == 48
assert len(intents) == 6

for count in intents.values():
    assert count == 8


# ============================================================
# Verify paired examples have identical bag-of-words counts
# ============================================================

pairs = [
    ("sub_ab", "sub_ba"),
    ("copy_a_x", "copy_x_a"),
    ("div_ab", "div_ba"),
]


print()
print("Checking bag-of-words collisions...")
print()


for left_intent, right_intent in pairs:

    for index in range(1, 9):

        left_id = (
            f"{left_intent}_{index:02}"
        )

        right_id = (
            f"{right_intent}_{index:02}"
        )

        left = by_id[left_id]
        right = by_id[right_id]

        left_bow = Counter(
            left.prompt_tokens
        )

        right_bow = Counter(
            right.prompt_tokens
        )

        assert left_bow == right_bow

        assert (
            left.prompt_tokens
            != right.prompt_tokens
        )

        print(
            f"PASS  {left_id:14} "
            f"<-> {right_id:14}"
        )


print()
print(
    "Every paired prompt has identical "
    "bag-of-words counts."
)

print(
    "Only token ORDER distinguishes "
    "the paired meanings."
)

print()
print(
    "✅ Sequence Challenge dataset PASSED"
)