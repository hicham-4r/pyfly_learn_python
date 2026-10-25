from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import create_dual_channel_map
from pyfly.tokenizer import build_code_vocabulary


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

DATASET_FILE = (
    "data/raw/"
    "sequence_challenge.jsonl"
)

SPLIT_FILE = (
    "data/processed/"
    "sequence_splits.json"
)

OUTPUT_DIR = Path(
    "artifacts/features"
)


NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


# ============================================================
# MUST MATCH ORIGINAL GENERATOR ENCODING
# ============================================================

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS_PER_CHANNEL = 16

FEEDBACK_NEURONS_PER_TOKEN = 32

PROMPT_TOKEN_STEPS = 8

PROMPT_GAP_STEPS = 2

WARMUP_STEPS = 10

AMPLITUDE = 0.8

TRACE_TAU = 0.25

MAP_SEED = 123


# ============================================================
# Intent -> broad family
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
    "       PyFly RICH FAMILY FEATURE EXTRACTION"
)

print(
    "===================================================="
)

print()


# ============================================================
# Dataset
# ============================================================

examples = load_dataset(
    DATASET_FILE
)


example_by_id = {
    example.id: example
    for example in examples
}


with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8",
) as file:

    split_data = json.load(
        file
    )


print(
    f"Logical examples: "
    f"{len(examples)}"
)


# ============================================================
# Prompt vocabulary
# ============================================================

prompt_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


# ============================================================
# Feedback vocabulary
#
# We include this even though the family classifier does not
# stimulate feedback neurons.
#
# Reason:
# it recreates EXACTLY the same Step-62 mapping and therefore
# EXACTLY the same indirect readout population.
# ============================================================

code_vocab = (
    build_code_vocabulary()
)


feedback_tokens = sorted([
    token
    for token in code_vocab.token_to_id
    if token != "PAD"
])


print(
    f"Prompt vocabulary: "
    f"{len(prompt_tokens)}"
)

print(
    f"Feedback vocabulary: "
    f"{len(feedback_tokens)}"
)


# ============================================================
# FlyBrain
# ============================================================

print()
print(
    "Loading FlyBrain..."
)


brain = FlyBrain(
    device=DEVICE,
    seed=0,
)


print(
    "FlyBrain loaded."
)


visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


# ============================================================
# Recreate exact original generator map
# ============================================================

dual_map = create_dual_channel_map(
    prompt_tokens=prompt_tokens,

    feedback_tokens=feedback_tokens,

    neuron_pool=visual_pool,

    max_prompt_positions=(
        MAX_PROMPT_POSITIONS
    ),

    prompt_neurons_per_channel=(
        PROMPT_NEURONS_PER_CHANNEL
    ),

    feedback_neurons_per_token=(
        FEEDBACK_NEURONS_PER_TOKEN
    ),

    seed=MAP_SEED,
)


input_neurons = (
    dual_map.all_input_neurons()
)


readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()
print(
    f"Visual neurons:         "
    f"{len(visual_pool):,}"
)

print(
    f"Direct input neurons:   "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout:       "
    f"{len(readout_population):,}"
)


# ============================================================
# Trace
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Run one prompt
# ============================================================

def run_prompt(
    prompt_tokens,
    seed,
):

    brain.reset(
        seed=seed
    )

    trace.reset()


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        trace.observe(
            fired
        )


    position_means = []

    all_prompt_states = []


    # ========================================================
    # Process each prompt position separately
    # ========================================================

    for position, token in enumerate(
        prompt_tokens
    ):

        if position >= (
            MAX_PROMPT_POSITIONS
        ):

            raise ValueError(
                f"Prompt too long: "
                f"{prompt_tokens}"
            )


        neurons = (
            dual_map.prompt_neurons_for(
                token,
                position,
            )
        )


        token_states = []


        # ----------------------------------------------------
        # Token stimulation
        # ----------------------------------------------------

        for _ in range(
            PROMPT_TOKEN_STEPS
        ):

            fired = brain.step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ]
            )


            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


            token_states.append(
                state
            )

            all_prompt_states.append(
                state
            )


        token_states = np.stack(
            token_states
        )


        # ----------------------------------------------------
        # Position-sensitive reservoir summary
        # ----------------------------------------------------

        position_mean = (
            token_states.mean(
                axis=0
            )
        )


        position_means.append(
            position_mean
        )


        # ----------------------------------------------------
        # Gap
        # ----------------------------------------------------

        for _ in range(
            PROMPT_GAP_STEPS
        ):

            fired = brain.step()


            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


            all_prompt_states.append(
                state
            )


    # ========================================================
    # Pad unused positions with zeros
    # ========================================================

    zero_state = np.zeros(
        len(
            readout_population
        ),
        dtype=np.float32,
    )


    while len(
        position_means
    ) < MAX_PROMPT_POSITIONS:

        position_means.append(
            zero_state.copy()
        )


    # ========================================================
    # Global summaries
    # ========================================================

    all_prompt_states = np.stack(
        all_prompt_states
    )


    prompt_mean = (
        all_prompt_states.mean(
            axis=0
        )
    )


    prompt_max = (
        all_prompt_states.max(
            axis=0
        )
    )


    prompt_final = (
        all_prompt_states[
            -1
        ]
    )


    # ========================================================
    # Rich family feature:
    #
    # P0 mean
    # P1 mean
    # ...
    # P5 mean
    # global mean
    # global max
    # global final
    #
    # = 9 reservoir blocks
    # ========================================================

    feature = np.concatenate(
        position_means
        +
        [
            prompt_mean,
            prompt_max,
            prompt_final,
        ]
    ).astype(
        np.float32
    )


    return feature


# ============================================================
# Extract one split
# ============================================================

def extract_split(
    split_name,
):

    logical_ids = (
        split_data[
            split_name
        ]
    )


    features = []

    families = []

    intents = []

    example_ids = []

    seeds = []


    total = (
        len(
            logical_ids
        )
        *
        len(
            NEURAL_SEEDS
        )
    )


    counter = 0


    print()
    print(
        "===================================================="
    )

    print(
        f" EXTRACTING {split_name.upper()}"
    )

    print(
        "===================================================="
    )

    print()


    for example_id in logical_ids:

        example = (
            example_by_id[
                example_id
            ]
        )


        family = intent_to_family(
            example.intent
        )


        for seed in NEURAL_SEEDS:

            counter += 1


            print(
                f"[{counter:03}/{total}] "
                f"{example_id:16} "
                f"family="
                f"{family:8} "
                f"seed={seed}"
            )


            feature = run_prompt(
                example.prompt_tokens,
                seed,
            )


            features.append(
                feature
            )

            families.append(
                family
            )

            intents.append(
                example.intent
            )

            example_ids.append(
                example.id
            )

            seeds.append(
                seed
            )


    X = np.stack(
        features
    )


    families = np.asarray(
        families
    )

    intents = np.asarray(
        intents
    )

    example_ids = np.asarray(
        example_ids
    )

    seeds = np.asarray(
        seeds,
        dtype=np.int32,
    )


    assert np.isfinite(
        X
    ).all()


    output_file = (
        OUTPUT_DIR
        /
        f"family_rich_{split_name}.npz"
    )


    np.savez(
        output_file,

        X=X,

        families=families,

        intents=intents,

        example_ids=example_ids,

        seeds=seeds,

        readout_population=(
            readout_population
        ),

        input_neurons=(
            input_neurons
        ),
    )


    print()
    print(
        f"Logical prompts: "
        f"{len(logical_ids)}"
    )

    print(
        f"Neural samples: "
        f"{len(X)}"
    )

    print(
        f"Feature size: "
        f"{X.shape[1]:,}"
    )

    print(
        f"Finite: "
        f"{np.isfinite(X).all()}"
    )

    print(
        f"Saved: "
        f"{output_file}"
    )


# ============================================================
# Run
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


extract_split(
    "train"
)

extract_split(
    "validation"
)

extract_split(
    "test"
)


print()
print(
    "===================================================="
)

print(
    "      RICH FAMILY EXTRACTION COMPLETE"
)

print(
    "===================================================="
)

print()


print(
    "Expected:"
)

print(
    "Train:      180 samples"
)

print(
    "Validation:  30 samples"
)

print(
    "Test:        30 samples"
)


print()
print(
    "✅ Rich family features extracted"
)