import numpy as np

from flybrain import FlyBrain

from pyfly.dataset import load_dataset

from pyfly.positional_feedback_map import (
    create_positional_feedback_map,
)

from pyfly.tokenizer import (
    build_code_vocabulary,
)


DEVICE = "cuda"

MAX_PROMPT_POSITIONS = 6
MAX_FEEDBACK_POSITIONS = 7

PROMPT_NEURONS = 16
FEEDBACK_NEURONS = 16

MAP_SEED = 123


print()
print(
    "============================================"
)

print(
    " PyFly POSITIONAL FEEDBACK MAP TEST"
)

print(
    "============================================"
)

print()


# ============================================================
# Brain
# ============================================================

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)


visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print(
    f"Visual pool: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Prompt vocabulary
# ============================================================

examples = load_dataset(
    "data/raw/"
    "sequence_challenge.jsonl"
)


prompt_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


# ============================================================
# Feedback vocabulary
# ============================================================

code_vocab = (
    build_code_vocabulary()
)


feedback_tokens = [
    token
    for token in code_vocab.token_to_id
    if token != "PAD"
]


# ============================================================
# Map
# ============================================================

mapping = create_positional_feedback_map(
    prompt_tokens=prompt_tokens,

    feedback_tokens=feedback_tokens,

    neuron_pool=visual_pool,

    max_prompt_positions=(
        MAX_PROMPT_POSITIONS
    ),

    max_feedback_positions=(
        MAX_FEEDBACK_POSITIONS
    ),

    prompt_neurons_per_channel=(
        PROMPT_NEURONS
    ),

    feedback_neurons_per_channel=(
        FEEDBACK_NEURONS
    ),

    seed=MAP_SEED,
)


# ============================================================
# Critical positional test
# ============================================================

name_a_pos2 = (
    mapping.feedback_neurons_for(
        "NAME_A",
        2,
    )
)

name_a_pos4 = (
    mapping.feedback_neurons_for(
        "NAME_A",
        4,
    )
)


print()
print(
    "NAME_A @ feedback position 2:"
)

print(
    name_a_pos2[:8].tolist()
)


print()
print(
    "NAME_A @ feedback position 4:"
)

print(
    name_a_pos4[:8].tolist()
)


assert not np.array_equal(
    name_a_pos2,
    name_a_pos4,
)


assert len(
    np.intersect1d(
        name_a_pos2,
        name_a_pos4,
    )
) == 0


print()
print(
    "Different position populations: PASS"
)


# ============================================================
# Global overlap test
# ============================================================

all_populations = []


for position in range(
    MAX_PROMPT_POSITIONS
):

    for token in prompt_tokens:

        all_populations.append(
            mapping.prompt_neurons_for(
                token,
                position,
            )
        )


for position in range(
    MAX_FEEDBACK_POSITIONS
):

    for token in feedback_tokens:

        all_populations.append(
            mapping.feedback_neurons_for(
                token,
                position,
            )
        )


all_neurons = np.concatenate(
    all_populations
)


unique_neurons = np.unique(
    all_neurons
)


print()
print(
    f"Assigned neurons: "
    f"{len(all_neurons):,}"
)

print(
    f"Unique neurons:   "
    f"{len(unique_neurons):,}"
)

print(
    f"Overlap:          "
    f"{len(all_neurons) - len(unique_neurons)}"
)


assert (
    len(all_neurons)
    ==
    len(unique_neurons)
)


# ============================================================
# Save
# ============================================================

mapping.save(
    "data/mappings/"
    "positional_feedback_map.npz"
)


print()
print(
    "✅ Positional feedback map PASSED"
)