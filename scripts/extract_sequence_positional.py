import json
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.positional_map import create_positional_sensory_map


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

DATASET_FILE = "data/raw/sequence_challenge.jsonl"
SPLIT_FILE = "data/processed/sequence_splits.json"

OUTPUT_DIR = Path("artifacts/features")

MAX_POSITIONS = 6

NEURONS_PER_CHANNEL = 32
MAPPING_SEED = 42

NEURAL_SEEDS = [0, 1, 2, 3, 4]

WARMUP_STEPS = 10

TOKEN_STEPS = 8
GAP_STEPS = 2
POST_STEPS = 4

AMPLITUDE = 0.8
TRACE_TAU = 0.25


# ============================================================
# Load dataset
# ============================================================

print()
print("============================================")
print(" PyFly POSITIONAL SEQUENCE EXTRACTION")
print("============================================")
print()


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
    f"Logical prompts: {len(examples)}"
)

print(
    f"Train prompts:   "
    f"{len(split_data['train'])}"
)

print(
    f"Validation:      "
    f"{len(split_data['validation'])}"
)

print(
    f"Test:            "
    f"{len(split_data['test'])}"
)


# ============================================================
# Validate lengths
# ============================================================

longest_prompt = max(
    len(example.prompt_tokens)
    for example in examples
)


print()
print(
    f"Longest prompt: "
    f"{longest_prompt}"
)


assert (
    longest_prompt
    <= MAX_POSITIONS
)


# ============================================================
# Load FlyBrain
# ============================================================

print()
print(
    "Loading FlyBrain on GPU..."
)


brain = FlyBrain(
    device=DEVICE,
    seed=0,
)


print(
    "Brain loaded."
)

print(
    f"Neurons:     {brain.n:,}"
)

print(
    f"Connections: "
    f"{len(brain.weights):,}"
)


# ============================================================
# Vocabulary
# ============================================================

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


# ============================================================
# Visual projection pool
# ============================================================

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
# Position-aware sensory mapping
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
# Collect every directly stimulated neuron
# ============================================================

input_neurons = []


for position in range(
    MAX_POSITIONS
):

    for token in tokens:

        input_neurons.extend(
            positional_map.neurons_for(
                token,
                position,
            ).tolist()
        )


input_neurons = np.asarray(
    sorted(
        set(input_neurons)
    ),
    dtype=np.int64,
)


print()
print(
    f"Positional channels: "
    f"{len(tokens) * MAX_POSITIONS}"
)

print(
    f"Direct input neurons: "
    f"{len(input_neurons):,}"
)


# ============================================================
# IMPORTANT:
# exclude all directly stimulated neurons from readout
# ============================================================

readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print(
    f"Indirect readout neurons: "
    f"{len(readout_population):,}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


if len(readout_population) == 0:

    raise RuntimeError(
        "Positional mapping consumed the "
        "entire visual-projection population."
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
# Run one complete prompt
# ============================================================

def run_prompt(
    prompt_tokens,
    seed,
):

    brain.reset(
        seed=seed
    )

    trace.reset()

    activity = []


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()

        activity.append(
            state
        )


    prompt_start = len(
        activity
    )


    # --------------------------------------------------------
    # Position-aware prompt stimulation
    # --------------------------------------------------------

    for position, token in enumerate(
        prompt_tokens
    ):

        token_neurons = (
            positional_map.neurons_for(
                token,
                position,
            )
        )


        # Token
        for _ in range(
            TOKEN_STEPS
        ):

            fired = brain.step(
                inject=[
                    (
                        token_neurons,
                        AMPLITUDE,
                    )
                ]
            )


            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


            activity.append(
                state
            )


        # Gap
        for _ in range(
            GAP_STEPS
        ):

            fired = brain.step()

            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()


            activity.append(
                state
            )


    prompt_end = len(
        activity
    )


    # --------------------------------------------------------
    # Post-prompt activity
    # --------------------------------------------------------

    post_states = []


    for _ in range(
        POST_STEPS
    ):

        fired = brain.step()

        state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()


        activity.append(
            state
        )

        post_states.append(
            state
        )


    activity = np.stack(
        activity
    )


    prompt_activity = activity[
        prompt_start:
        prompt_end
    ]


    # ========================================================
    # Feature extraction
    #
    # IMPORTANT:
    # We do NOT concatenate explicit per-position states.
    #
    # Positional information enters ONLY through
    # position-specific sensory neuron populations.
    # ========================================================

    mean_prompt = (
        prompt_activity.mean(
            axis=0
        )
    )


    max_prompt = (
        prompt_activity.max(
            axis=0
        )
    )


    final_prompt = activity[
        prompt_end - 1
    ]


    mean_post = np.stack(
        post_states
    ).mean(
        axis=0
    )


    final_post = post_states[
        -1
    ]


    feature = np.concatenate(
        [
            mean_prompt,
            max_prompt,
            final_prompt,
            mean_post,
            final_post,
        ]
    )


    return feature.astype(
        np.float32
    )


# ============================================================
# Extract one split
# ============================================================

def extract_split(
    split_name,
    prompt_ids,
):

    features = []
    intents = []
    example_ids = []
    seeds = []


    total = (
        len(prompt_ids)
        * len(NEURAL_SEEDS)
    )


    counter = 0


    print()
    print(
        "============================================"
    )

    print(
        f" EXTRACTING "
        f"{split_name.upper()}"
    )

    print(
        "============================================"
    )

    print()


    for example_id in prompt_ids:

        example = example_by_id[
            example_id
        ]


        for seed in NEURAL_SEEDS:

            counter += 1


            print(
                f"[{counter:03}/{total}] "
                f"{example.id:16} "
                f"seed={seed} "
                f"len="
                f"{len(example.prompt_tokens)}"
            )


            feature = run_prompt(
                prompt_tokens=(
                    example.prompt_tokens
                ),

                seed=seed,
            )


            features.append(
                feature
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


    intents = np.asarray(
        intents
    )

    example_ids = np.asarray(
        example_ids
    )

    seeds = np.asarray(
        seeds
    )


    print()
    print(
        f"{split_name} episodes:   "
        f"{X.shape[0]}"
    )

    print(
        f"Readout neurons:    "
        f"{len(readout_population):,}"
    )

    print(
        f"Feature dimensions: "
        f"{X.shape[1]:,}"
    )

    print(
        f"Finite:             "
        f"{np.isfinite(X).all()}"
    )


    assert np.isfinite(
        X
    ).all()


    output_file = (
        OUTPUT_DIR
        /
        f"sequence_positional_{split_name}.npz"
    )


    np.savez(
        output_file,

        X=X,

        intents=intents,

        example_ids=example_ids,

        seeds=seeds,

        input_neurons=input_neurons,

        readout_population=(
            readout_population
        ),

        neurons_per_channel=(
            NEURONS_PER_CHANNEL
        ),

        max_positions=(
            MAX_POSITIONS
        ),

        trace_tau=(
            TRACE_TAU
        ),
    )


    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Extract all splits
# ============================================================

extract_split(
    "train",
    split_data[
        "train"
    ],
)


extract_split(
    "validation",
    split_data[
        "validation"
    ],
)


extract_split(
    "test",
    split_data[
        "test"
    ],
)


# ============================================================
# Summary
# ============================================================

print()
print(
    "============================================"
)

print(
    " POSITIONAL EXTRACTION COMPLETE"
)

print(
    "============================================"
)

print()

print(
    "Train:      180 episodes"
)

print(
    "Validation:  30 episodes"
)

print(
    "Test:        30 episodes"
)

print()

print(
    "Total: 240 reservoir episodes"
)

print()

print(
    f"Neurons/channel: "
    f"{NEURONS_PER_CHANNEL}"
)

print(
    f"Trace tau: "
    f"{TRACE_TAU}"
)

print(
    f"Indirect readout neurons: "
    f"{len(readout_population):,}"
)

print()

print(
    "✅ Position-aware MaleCNS extraction PASSED"
)