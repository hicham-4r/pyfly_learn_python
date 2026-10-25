import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.encoder import build_prompt_schedule
from pyfly.sensory_map import create_disjoint_sensory_map


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

NEURONS_PER_TOKEN = 64
MAPPING_SEED = 42

SEEDS = [0, 1, 2, 3, 4]

WARMUP_STEPS = 10
TOKEN_STEPS = 8
GAP_STEPS = 2
POST_STEPS = 4

AMPLITUDE = 0.8
TRACE_TAU = 0.1


# ============================================================
# Load brain
# ============================================================

print()
print("========================================")
print("    PyFly PROMPT FEATURE EXTRACTION")
print("========================================")
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
# Dataset
# ============================================================

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)

all_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})

print()
print(f"Logical prompts: {len(examples)}")
print(f"Prompt vocabulary: {len(all_tokens)}")


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
# Remove ALL directly stimulated neurons from readout
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
print(
    f"Visual neurons:            "
    f"{len(visual_pool):,}"
)

print(
    f"Direct encoder neurons:    "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout neurons:  "
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
# Run one COMPLETE prompt
# ============================================================

def run_prompt(prompt_tokens, seed):

    brain.reset(seed=seed)
    trace.reset()

    activity = []

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        activity.append(
            np.asarray(
                trace.observe(fired)
            ).copy()
        )


    # --------------------------------------------------------
    # Temporal prompt schedule
    # --------------------------------------------------------

    schedule = build_prompt_schedule(
        prompt_tokens=prompt_tokens,
        sensory_map=sensory_map,
        token_steps=TOKEN_STEPS,
        gap_steps=GAP_STEPS,
        amplitude=AMPLITUDE,
    )


    prompt_start = len(activity)


    for event in schedule:

        for _ in range(event.steps):

            # GAP
            if len(event.neurons) == 0:

                fired = brain.step()

            # TOKEN
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
                    trace.observe(fired)
                ).copy()
            )


    prompt_end = len(activity)


    # --------------------------------------------------------
    # Post-prompt propagation
    # --------------------------------------------------------

    for _ in range(POST_STEPS):

        fired = brain.step()

        activity.append(
            np.asarray(
                trace.observe(fired)
            ).copy()
        )


    activity = np.stack(activity)


    # ========================================================
    # Feature extraction
    # ========================================================

    prompt_activity = activity[
        prompt_start:prompt_end
    ]


    mean_prompt = prompt_activity.mean(
        axis=0
    )

    max_prompt = prompt_activity.max(
        axis=0
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
# Generate prompt episodes
# ============================================================

features = []
intents = []
example_ids = []
seed_labels = []


total = (
    len(examples)
    * len(SEEDS)
)

counter = 0


print()
print("========================================")
print("       RUNNING COMPLETE PROMPTS")
print("========================================")
print()


for example in examples:

    for seed in SEEDS:

        counter += 1

        print(
            f"[{counter:02}/{total}] "
            f"{example.id:20} "
            f"seed={seed}"
        )

        print(
            f"       prompt = "
            f"{example.prompt_tokens}"
        )


        feature = run_prompt(
            example.prompt_tokens,
            seed,
        )


        features.append(feature)
        intents.append(example.intent)
        example_ids.append(example.id)
        seed_labels.append(seed)


# ============================================================
# Convert
# ============================================================

X = np.stack(features)

intents = np.asarray(
    intents
)

example_ids = np.asarray(
    example_ids
)

seed_labels = np.asarray(
    seed_labels
)


print()
print("========================================")
print("          FEATURE DATASET")
print("========================================")
print()

print(
    f"Episodes:           {X.shape[0]}"
)

print(
    f"Feature dimensions: {X.shape[1]:,}"
)

print(
    f"Classes:            "
    f"{len(np.unique(intents))}"
)

print(
    f"Finite:             "
    f"{np.isfinite(X).all()}"
)


assert np.isfinite(X).all()


# ============================================================
# Save
# ============================================================

OUTPUT = (
    "artifacts/features/"
    "template_prompt_features.npz"
)


np.savez(
    OUTPUT,

    X=X,

    intents=intents,

    example_ids=example_ids,

    seeds=seed_labels,

    input_neurons=input_neurons,

    readout_population=readout_population,
)


print()
print(
    f"Saved: {OUTPUT}"
)

print()
print(
    "✅ Complete-prompt feature extraction PASSED"
)