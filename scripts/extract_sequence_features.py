import json
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.sensory_map import create_disjoint_sensory_map


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

DATASET_FILE = "data/raw/sequence_challenge.jsonl"
SPLIT_FILE = "data/processed/sequence_splits.json"

OUTPUT_DIR = Path("artifacts/features")

NEURONS_PER_TOKEN = 64
MAPPING_SEED = 42

NEURAL_SEEDS = [0, 1, 2, 3, 4]

WARMUP_STEPS = 10

TOKEN_STEPS = 8
GAP_STEPS = 2
POST_STEPS = 4

AMPLITUDE = 0.8
TRACE_TAU = 0.1

# Longest prompt in our challenge has 6 tokens.
MAX_TOKENS = 6


# ============================================================
# Load dataset and splits
# ============================================================

print()
print("============================================")
print(" PyFly SEQUENCE FEATURE EXTRACTION")
print("============================================")
print()


examples = load_dataset(DATASET_FILE)

example_by_id = {
    example.id: example
    for example in examples
}


with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8",
) as file:
    split_data = json.load(file)


print(f"Logical prompts: {len(examples)}")
print(f"Train prompts:   {len(split_data['train'])}")
print(f"Validation:      {len(split_data['validation'])}")
print(f"Test:            {len(split_data['test'])}")


# ============================================================
# Validate prompt lengths
# ============================================================

longest_prompt = max(
    len(example.prompt_tokens)
    for example in examples
)

print()
print(f"Longest prompt: {longest_prompt} tokens")

assert longest_prompt <= MAX_TOKENS


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

print(f"Neurons:     {brain.n:,}")
print(f"Connections: {len(brain.weights):,}")


# ============================================================
# Build vocabulary
# ============================================================

all_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


print()
print(f"Sequence vocabulary size: {len(all_tokens)}")


# ============================================================
# Input population
# ============================================================

visual_pool = np.asarray(
    brain.cells(["visual_projection"])
)


sensory_map = create_disjoint_sensory_map(
    tokens=all_tokens,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


# ============================================================
# Remove directly stimulated neurons from readout
# ============================================================

input_neurons = []

for token in all_tokens:

    input_neurons.extend(
        sensory_map.neurons_for(token).tolist()
    )


input_neurons = np.asarray(
    sorted(set(input_neurons)),
    dtype=np.int64,
)


readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()
print(f"Visual neurons:           {len(visual_pool):,}")
print(f"Direct input neurons:     {len(input_neurons):,}")
print(f"Indirect readout neurons: {len(readout_population):,}")


# ============================================================
# Trace recorder
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Run one order-sensitive prompt
# ============================================================

def run_prompt(prompt_tokens, seed):

    brain.reset(seed=seed)
    trace.reset()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        trace.observe(fired)


    # --------------------------------------------------------
    # Position-sensitive states
    # --------------------------------------------------------

    token_states = []

    all_prompt_states = []


    for token in prompt_tokens:

        token_neurons = sensory_map.neurons_for(token)

        current_token_states = []


        # Present this token
        for _ in range(TOKEN_STEPS):

            fired = brain.step(
                inject=[
                    (
                        token_neurons,
                        AMPLITUDE,
                    )
                ]
            )

            state = np.asarray(
                trace.observe(fired)
            ).copy()

            current_token_states.append(
                state
            )

            all_prompt_states.append(
                state
            )


        # Store average response for THIS token position
        token_mean = np.stack(
            current_token_states
        ).mean(axis=0)

        token_states.append(
            token_mean
        )


        # Inter-token gap
        for _ in range(GAP_STEPS):

            fired = brain.step()

            state = np.asarray(
                trace.observe(fired)
            ).copy()

            all_prompt_states.append(
                state
            )


    # --------------------------------------------------------
    # Post-prompt propagation
    # --------------------------------------------------------

    post_states = []

    for _ in range(POST_STEPS):

        fired = brain.step()

        state = np.asarray(
            trace.observe(fired)
        ).copy()

        post_states.append(
            state
        )


    # ========================================================
    # Build fixed-length temporal feature
    # ========================================================

    zero_state = np.zeros(
        len(readout_population),
        dtype=np.float32,
    )


    padded_token_states = []


    for position in range(MAX_TOKENS):

        if position < len(token_states):

            padded_token_states.append(
                token_states[position]
            )

        else:

            padded_token_states.append(
                zero_state
            )


    # Final state at the end of prompt
    final_prompt_state = (
        token_states[-1]
    )


    # State after propagation
    final_post_state = (
        post_states[-1]
        if post_states
        else final_prompt_state
    )


    # Global prompt mean
    global_prompt_mean = np.stack(
        all_prompt_states
    ).mean(axis=0)


    # --------------------------------------------------------
    # ORDER-SENSITIVE FEATURE:
    #
    # position 1 neural state
    # position 2 neural state
    # ...
    # position 6 neural state
    # global prompt mean
    # final prompt state
    # final post state
    # --------------------------------------------------------

    feature = np.concatenate(
        padded_token_states
        + [
            global_prompt_mean,
            final_prompt_state,
            final_post_state,
        ]
    )


    return feature.astype(
        np.float32
    )


# ============================================================
# Extract split
# ============================================================

def extract_split(
    split_name,
    prompt_ids,
):

    features = []
    intents = []
    example_ids = []
    seeds = []
    prompt_lengths = []


    total = (
        len(prompt_ids)
        * len(NEURAL_SEEDS)
    )


    counter = 0


    print()
    print("============================================")
    print(f" EXTRACTING {split_name.upper()}")
    print("============================================")
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
                f"len={len(example.prompt_tokens)}"
            )


            feature = run_prompt(
                prompt_tokens=example.prompt_tokens,
                seed=seed,
            )


            features.append(feature)

            intents.append(
                example.intent
            )

            example_ids.append(
                example.id
            )

            seeds.append(seed)

            prompt_lengths.append(
                len(example.prompt_tokens)
            )


    X = np.stack(features)


    intents = np.asarray(
        intents
    )

    example_ids = np.asarray(
        example_ids
    )

    seeds = np.asarray(
        seeds
    )

    prompt_lengths = np.asarray(
        prompt_lengths
    )


    print()
    print(f"{split_name} episodes:   {X.shape[0]}")
    print(f"Feature dimensions: {X.shape[1]:,}")
    print(f"Finite:             {np.isfinite(X).all()}")


    assert np.isfinite(X).all()


    output_file = (
        OUTPUT_DIR
        / f"sequence_{split_name}.npz"
    )


    np.savez(
        output_file,

        X=X,

        intents=intents,

        example_ids=example_ids,

        seeds=seeds,

        prompt_lengths=prompt_lengths,

        input_neurons=input_neurons,

        readout_population=readout_population,

        max_tokens=MAX_TOKENS,
    )


    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Extract all splits
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


extract_split(
    "train",
    split_data["train"],
)


extract_split(
    "validation",
    split_data["validation"],
)


extract_split(
    "test",
    split_data["test"],
)


# ============================================================
# Finished
# ============================================================

print()
print("============================================")
print("      SEQUENCE EXTRACTION COMPLETE")
print("============================================")
print()

print("Train:      36 × 5 = 180 episodes")
print("Validation:  6 × 5 = 30 episodes")
print("Test:        6 × 5 = 30 episodes")

print()
print("Total: 240 reservoir episodes")

print()
print(
    "✅ Sequence Challenge feature extraction PASSED"
)