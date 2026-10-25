import numpy as np

from flybrain import FlyBrain

from pyfly.dataset import load_dataset

from pyfly.dual_channel_map import (
    create_dual_channel_map,
)

from pyfly.tokenizer import (
    build_code_vocabulary,
)


DEVICE = "cuda"

DATASET = (
    "data/raw/"
    "sequence_challenge.jsonl"
)

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS = 16
FEEDBACK_NEURONS = 32

MAP_SEED = 123


print()
print("========================================")
print("     PyFly DUAL CHANNEL MAP TEST")
print("========================================")
print()


# ============================================================
# Brain
# ============================================================

print(
    "Loading FlyBrain..."
)

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print(
    "Brain loaded."
)


visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print(
    f"Visual neuron pool: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Prompt vocabulary
# ============================================================

examples = load_dataset(
    DATASET
)


prompt_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


print(
    f"Prompt vocabulary: "
    f"{len(prompt_tokens)}"
)


# ============================================================
# Feedback/code vocabulary
# ============================================================

code_vocab = (
    build_code_vocabulary()
)


feedback_tokens = [
    token
    for token in (
        code_vocab.token_to_id
    )
    if token != "PAD"
]


print(
    f"Feedback vocabulary: "
    f"{len(feedback_tokens)}"
)


# ============================================================
# Build map
# ============================================================

dual_map = create_dual_channel_map(
    prompt_tokens=prompt_tokens,

    feedback_tokens=feedback_tokens,

    neuron_pool=visual_pool,

    max_prompt_positions=(
        MAX_PROMPT_POSITIONS
    ),

    prompt_neurons_per_channel=(
        PROMPT_NEURONS
    ),

    feedback_neurons_per_token=(
        FEEDBACK_NEURONS
    ),

    seed=MAP_SEED,
)


# ============================================================
# Collect populations
# ============================================================

prompt_neurons = []


for position in range(
    MAX_PROMPT_POSITIONS
):

    for token in prompt_tokens:

        neurons = (
            dual_map.prompt_neurons_for(
                token,
                position,
            )
        )

        assert (
            len(neurons)
            == PROMPT_NEURONS
        )

        prompt_neurons.extend(
            neurons.tolist()
        )


feedback_neurons = []


for token in feedback_tokens:

    neurons = (
        dual_map.feedback_neurons_for(
            token
        )
    )

    assert (
        len(neurons)
        == FEEDBACK_NEURONS
    )

    feedback_neurons.extend(
        neurons.tolist()
    )


# ============================================================
# Check uniqueness
# ============================================================

prompt_set = set(
    prompt_neurons
)

feedback_set = set(
    feedback_neurons
)


prompt_overlap = (
    len(prompt_neurons)
    -
    len(prompt_set)
)

feedback_overlap = (
    len(feedback_neurons)
    -
    len(feedback_set)
)


cross_overlap = (
    prompt_set
    &
    feedback_set
)


print()
print("========================================")
print("           CHANNEL STATISTICS")
print("========================================")
print()


print(
    f"Prompt neurons:     "
    f"{len(prompt_neurons):,}"
)

print(
    f"Prompt unique:      "
    f"{len(prompt_set):,}"
)

print(
    f"Prompt overlap:     "
    f"{prompt_overlap}"
)


print()

print(
    f"Feedback neurons:   "
    f"{len(feedback_neurons):,}"
)

print(
    f"Feedback unique:    "
    f"{len(feedback_set):,}"
)

print(
    f"Feedback overlap:   "
    f"{feedback_overlap}"
)


print()

print(
    f"Cross-channel overlap: "
    f"{len(cross_overlap)}"
)


assert prompt_overlap == 0

assert feedback_overlap == 0

assert len(
    cross_overlap
) == 0


# ============================================================
# Example prompt channel
# ============================================================

print()
print("========================================")
print("        EXAMPLE PROMPT CHANNEL")
print("========================================")
print()


example_prompt = [
    "subtract",
    "b",
    "from",
    "a",
]


for position, token in enumerate(
    example_prompt
):

    neurons = (
        dual_map.prompt_neurons_for(
            token,
            position,
        )
    )


    print(
        f"position={position} "
        f"token={token:10} "
        f"neurons="
        f"{neurons[:6].tolist()}..."
    )


# ============================================================
# Example feedback channel
# ============================================================

print()
print("========================================")
print("       EXAMPLE FEEDBACK CHANNEL")
print("========================================")
print()


feedback_example = [
    "BOS",
    "NAME_Y",
    "ASSIGN",
    "NAME_A",
    "MINUS",
    "NAME_B",
]


for token in feedback_example:

    neurons = (
        dual_map.feedback_neurons_for(
            token
        )
    )


    print(
        f"{token:10} "
        f"-> "
        f"{neurons[:6].tolist()}..."
    )


# ============================================================
# Determinism
# ============================================================

second_map = create_dual_channel_map(
    prompt_tokens=prompt_tokens,

    feedback_tokens=feedback_tokens,

    neuron_pool=visual_pool,

    max_prompt_positions=(
        MAX_PROMPT_POSITIONS
    ),

    prompt_neurons_per_channel=(
        PROMPT_NEURONS
    ),

    feedback_neurons_per_token=(
        FEEDBACK_NEURONS
    ),

    seed=MAP_SEED,
)


for position in range(
    MAX_PROMPT_POSITIONS
):

    for token in prompt_tokens:

        assert np.array_equal(
            dual_map.prompt_neurons_for(
                token,
                position,
            ),

            second_map.prompt_neurons_for(
                token,
                position,
            ),
        )


for token in feedback_tokens:

    assert np.array_equal(
        dual_map.feedback_neurons_for(
            token
        ),

        second_map.feedback_neurons_for(
            token
        ),
    )


print()
print(
    "Mapping reproducibility: PASS"
)


# ============================================================
# Save
# ============================================================

OUTPUT = (
    "data/mappings/"
    "dual_channel_map.npz"
)


dual_map.save(
    OUTPUT
)


print(
    f"Saved: {OUTPUT}"
)


print()
print(
    "✅ Dual-channel neural mapping PASSED"
)