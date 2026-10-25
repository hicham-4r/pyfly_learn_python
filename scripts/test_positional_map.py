import numpy as np

from flybrain import FlyBrain

from pyfly.dataset import load_dataset
from pyfly.positional_map import (
    create_positional_sensory_map,
)


DEVICE = "cuda"

DATASET = (
    "data/raw/"
    "sequence_challenge.jsonl"
)

MAX_POSITIONS = 6

NEURONS_PER_CHANNEL = 32

MAPPING_SEED = 42


print()
print("========================================")
print("      PyFly POSITIONAL NEURAL MAP")
print("========================================")
print()


# ============================================================
# Load brain
# ============================================================

print("Loading FlyBrain...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")


# ============================================================
# Dataset vocabulary
# ============================================================

examples = load_dataset(
    DATASET
)


tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


print()
print(
    f"Vocabulary size: "
    f"{len(tokens)}"
)

print(
    f"Maximum positions: "
    f"{MAX_POSITIONS}"
)

print(
    f"Neurons/channel: "
    f"{NEURONS_PER_CHANNEL}"
)


# ============================================================
# Neural pool
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print(
    f"Visual-projection pool: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Create map
# ============================================================

positional_map = (
    create_positional_sensory_map(
        tokens=tokens,

        neuron_pool=visual_pool,

        max_positions=MAX_POSITIONS,

        neurons_per_channel=(
            NEURONS_PER_CHANNEL
        ),

        seed=MAPPING_SEED,
    )
)


# ============================================================
# Mapping statistics
# ============================================================

number_of_channels = (
    len(tokens)
    * MAX_POSITIONS
)


expected_neurons = (
    number_of_channels
    * NEURONS_PER_CHANNEL
)


all_neurons = []


for position in range(
    MAX_POSITIONS
):

    for token in tokens:

        neurons = (
            positional_map.neurons_for(
                token,
                position,
            )
        )


        assert (
            len(neurons)
            == NEURONS_PER_CHANNEL
        )


        all_neurons.extend(
            neurons.tolist()
        )


unique_neurons = set(
    all_neurons
)


print()
print("========================================")
print("            MAP STATISTICS")
print("========================================")
print()

print(
    f"Channels:              "
    f"{number_of_channels}"
)

print(
    f"Assigned neurons:      "
    f"{len(all_neurons)}"
)

print(
    f"Unique neurons:        "
    f"{len(unique_neurons)}"
)

print(
    f"Expected neurons:      "
    f"{expected_neurons}"
)


assert (
    len(all_neurons)
    == expected_neurons
)

assert (
    len(unique_neurons)
    == expected_neurons
)


print(
    "Channel overlap:       0"
)


# ============================================================
# Check that token identity + position matters
# ============================================================

token_a_pos0 = (
    positional_map.neurons_for(
        "a",
        0,
    )
)

token_a_pos1 = (
    positional_map.neurons_for(
        "a",
        1,
    )
)


assert not np.array_equal(
    token_a_pos0,
    token_a_pos1,
)


print()
print(
    "Same token at different positions:"
)

print(
    "a @ position 0:",
    token_a_pos0[:8].tolist(),
)

print(
    "a @ position 1:",
    token_a_pos1[:8].tolist(),
)

print()

print(
    "Different neural populations: PASS"
)


# ============================================================
# Determinism
# ============================================================

second_map = (
    create_positional_sensory_map(
        tokens=tokens,

        neuron_pool=visual_pool,

        max_positions=MAX_POSITIONS,

        neurons_per_channel=(
            NEURONS_PER_CHANNEL
        ),

        seed=MAPPING_SEED,
    )
)


for position in range(
    MAX_POSITIONS
):

    for token in tokens:

        assert np.array_equal(
            positional_map.neurons_for(
                token,
                position,
            ),

            second_map.neurons_for(
                token,
                position,
            ),
        )


print(
    "Mapping reproducibility: PASS"
)


# ============================================================
# Save
# ============================================================

OUTPUT = (
    "data/mappings/"
    "positional_sensory_map.npz"
)


positional_map.save(
    OUTPUT
)


print()
print(
    f"Saved: {OUTPUT}"
)

print()
print(
    "✅ Positional neural map PASSED"
)