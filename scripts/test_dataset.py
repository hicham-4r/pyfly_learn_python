from collections import Counter

from pyfly.dataset import load_dataset


DATASET_PATH = "data/raw/python_mre.jsonl"


examples = load_dataset(DATASET_PATH)

print()
print("=== PyFly Dataset ===")
print(f"Total examples: {len(examples)}")

intents = Counter(example.intent for example in examples)

print(f"Number of intents: {len(intents)}")

print()
print("Intent counts:")

for intent, count in intents.items():
    print(f"  {intent:15} : {count}")

print()
print("=== First Example ===")

example = examples[0]

print("ID:            ", example.id)
print("Intent:        ", example.intent)
print("Prompt tokens: ", example.prompt_tokens)
print("Target code:   ", example.code)

print()
print("✅ Dataset loaded successfully")