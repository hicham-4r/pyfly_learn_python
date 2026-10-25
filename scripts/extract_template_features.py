import json
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.encoder import build_prompt_schedule
from pyfly.sensory_map import create_disjoint_sensory_map


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

DATASET_FILE = "data/raw/python_mre.jsonl"
SPLIT_FILE = "data/processed/splits.json"

OUTPUT_DIR = Path(
    "artifacts/features"
)

NEURONS_PER_TOKEN = 64
MAPPING_SEED = 42

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

WARMUP_STEPS = 10
TOKEN_STEPS = 8
GAP_STEPS = 2
POST_STEPS = 4

AMPLITUDE = 0.8
TRACE_TAU = 0.1


# ============================================================
# Load dataset
# ============================================================

print()
print("========================================")
print(" PyFly FULL TEMPLATE FEATURE EXTRACTION")
print("========================================")
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
    f"Test prompts:    "
    f"{len(split_data['test'])}"
)


# ============================================================
# Load FlyBrain
# ============================================================

print()
print("Loading FlyBrain on GPU...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")

print(
    f"Neurons:     {brain.n:,}"
)

print(
    f"Connections: {len(brain.weights):,}"
)


# ============================================================
# Prompt vocabulary
# ============================================================

all_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


print()
print(
    f"Prompt vocabulary size: "
    f"{len(all_tokens)}"
)


# ============================================================
# Visual input population
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


sensory_map = create_disjoint_sensory_map(
    tokens=all_tokens,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


# ============================================================
# Find all directly stimulated neurons
# ============================================================

input_neurons = []

for token in all_tokens:

    input_neurons.extend(
        sensory_map
        .neurons_for(token)
        .tolist()
    )


input_neurons = np.asarray(
    sorted(
        set(input_neurons)
    ),
    dtype=np.int64,
)


# ============================================================
# Indirect readout population
# ============================================================

readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()
print(
    f"Visual neurons:          "
    f"{len(visual_pool):,}"
)

print(
    f"Direct input neurons:    "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout neurons:"
    f" {len(readout_population):,}"
)


# ============================================================
# Trace recorder
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Run complete prompt
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
    # Warmup
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        activity.append(
            np.asarray(
                trace.observe(
                    fired
                )
            ).copy()
        )


    # --------------------------------------------------------
    # Prompt schedule
    # --------------------------------------------------------

    schedule = build_prompt_schedule(
        prompt_tokens=prompt_tokens,
        sensory_map=sensory_map,
        token_steps=TOKEN_STEPS,
        gap_steps=GAP_STEPS,
        amplitude=AMPLITUDE,
    )


    prompt_start = len(
        activity
    )


    for event in schedule:

        for _ in range(
            event.steps
        ):

            if len(
                event.neurons
            ) == 0:

                fired = brain.step()

            else:

                fired = brain.step(
                    inject=[
                        (
                            event.neurons,
                            event.amplitude,
                        )
                    ]
                )


            activity.append(
                np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()
            )


    prompt_end = len(
        activity
    )


    # --------------------------------------------------------
    # Post-prompt propagation
    # --------------------------------------------------------

    for _ in range(
        POST_STEPS
    ):

        fired = brain.step()

        activity.append(
            np.asarray(
                trace.observe(
                    fired
                )
            ).copy()
        )


    activity = np.stack(
        activity
    )


    # ========================================================
    # Feature extraction
    # ========================================================

    prompt_activity = activity[
        prompt_start:
        prompt_end
    ]


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


    final_trace = activity[
        -1
    ]


    feature = np.concatenate(
        [
            mean_prompt,
            max_prompt,
            final_prompt,
            final_trace,
        ]
    )


    return feature


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
        *
        len(NEURAL_SEEDS)
    )


    counter = 0


    print()
    print(
        "========================================"
    )

    print(
        f" EXTRACTING {split_name.upper()}"
    )

    print(
        "========================================"
    )

    print()


    for example_id in prompt_ids:

        example = (
            example_by_id[
                example_id
            ]
        )


        for seed in NEURAL_SEEDS:

            counter += 1


            print(
                f"[{counter:03}/{total}] "
                f"{example.id:22} "
                f"seed={seed}"
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


    intents_array = np.asarray(
        intents
    )

    ids_array = np.asarray(
        example_ids
    )

    seeds_array = np.asarray(
        seeds
    )


    print()
    print(
        f"{split_name} episodes: "
        f"{X.shape[0]}"
    )

    print(
        f"Feature dimensions: "
        f"{X.shape[1]:,}"
    )

    print(
        f"Finite: "
        f"{np.isfinite(X).all()}"
    )


    assert np.isfinite(
        X
    ).all()


    output_file = (
        OUTPUT_DIR
        /
        f"template_{split_name}.npz"
    )


    np.savez(
        output_file,

        X=X,

        intents=(
            intents_array
        ),

        example_ids=(
            ids_array
        ),

        seeds=(
            seeds_array
        ),

        input_neurons=(
            input_neurons
        ),

        readout_population=(
            readout_population
        ),
    )


    print(
        f"Saved: "
        f"{output_file}"
    )


# ============================================================
# Run all splits
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
# Complete
# ============================================================

print()
print("========================================")
print("       EXTRACTION COMPLETE")
print("========================================")
print()

print(
    "Train:      36 × 5 = 180 episodes"
)

print(
    "Validation:  6 × 5 = 30 episodes"
)

print(
    "Test:        6 × 5 = 30 episodes"
)

print()

print(
    "Total: 240 reservoir episodes"
)

print()

print(
    "✅ PyFly full feature extraction PASSED"
)