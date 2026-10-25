import numpy as np

from flybrain import FlyBrain

from pyfly.dataset import load_dataset
from pyfly.sensory_map import create_disjoint_sensory_map


print("Loading FlyBrain on RTX GPU...")

brain = FlyBrain(
    device="cuda",
    seed=0,
)

print("Brain loaded.")

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)

# Collect every unique prompt word
prompt_tokens = sorted(
    {
        token
        for example in examples
        for token in example.prompt_tokens
    }
)

print()
print("Prompt vocabulary:")
print(prompt_tokens)

print()
print(f"Vocabulary size: {len(prompt_tokens)}")


# Get the real visual-projection neurons
visual_pool = brain.cells(
    ["visual_projection"]
)

visual_pool = np.asarray(visual_pool)

print(
    f"Visual-projection neuron pool: "
    f"{len(visual_pool):,}"
)


# Build deterministic neural mapping
sensory_map = create_disjoint_sensory_map(
    tokens=prompt_tokens,
    neuron_pool=visual_pool,
    neurons_per_token=24,
    seed=42,
)


print()
print("=== Neural Token Mapping ===")

for token in prompt_tokens:

    neurons = sensory_map.neurons_for(token)

    print(
        f"{token:12} -> "
        f"{neurons[:6].tolist()} ... "
        f"({len(neurons)} neurons)"
    )


# -----------------------------------
# Check that populations don't overlap
# -----------------------------------

all_neurons = []

for token in prompt_tokens:
    all_neurons.extend(
        sensory_map.neurons_for(token).tolist()
    )

total_assigned = len(all_neurons)
unique_assigned = len(set(all_neurons))

print()
print("=== Mapping Statistics ===")

print(
    f"Vocabulary size:         "
    f"{len(prompt_tokens)}"
)

print(
    f"Neurons per token:       "
    f"{sensory_map.neurons_per_token}"
)

print(
    f"Total assigned neurons:  "
    f"{total_assigned}"
)

print(
    f"Unique assigned neurons: "
    f"{unique_assigned}"
)

overlap = total_assigned - unique_assigned

print(
    f"Neuron overlap:          "
    f"{overlap}"
)


assert overlap == 0


# -----------------------------------
# Determinism test
# -----------------------------------

second_map = create_disjoint_sensory_map(
    tokens=prompt_tokens,
    neuron_pool=visual_pool,
    neurons_per_token=24,
    seed=42,
)

for token in prompt_tokens:

    assert np.array_equal(
        sensory_map.neurons_for(token),
        second_map.neurons_for(token),
    )


print("Mapping reproducibility:  PASS")


# Save the mapping
sensory_map.save(
    "data/mappings/sensory_map.npz"
)

print(
    "Saved mapping: "
    "data/mappings/sensory_map.npz"
)

print()
print("✅ PyFly sensory mapping PASSED")