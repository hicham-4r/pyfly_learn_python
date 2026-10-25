import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.sensory_map import create_disjoint_sensory_map


DEVICE = "cuda"

NEURONS_PER_TOKEN = 64
MAPPING_SEED = 42

SEEDS = [0, 1, 2, 3, 4]

WARMUP_STEPS = 10
TOKEN_STEPS = 8
POST_STEPS = 4

AMPLITUDE = 0.8
TRACE_TAU = 0.1


print()
print("========================================")
print(" PyFly INDIRECT VISUAL READOUT")
print("========================================")
print()

print("Loading FlyBrain...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded.")


# ============================================================
# Dataset
# ============================================================

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)

tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})


# ============================================================
# Visual population
# ============================================================

visual_pool = np.asarray(
    brain.cells(["visual_projection"])
)


sensory_map = create_disjoint_sensory_map(
    tokens=tokens,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


# ============================================================
# Find every neuron directly used by the encoder
# ============================================================

input_neurons = []

for token in tokens:

    input_neurons.extend(
        sensory_map.neurons_for(token).tolist()
    )


input_neurons = np.asarray(
    sorted(set(input_neurons)),
    dtype=np.int64,
)


print(
    f"Directly stimulated neurons: "
    f"{len(input_neurons):,}"
)


# ============================================================
# IMPORTANT:
# Remove all directly stimulated neurons from the readout
# ============================================================

readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print(
    f"Visual neurons total:       "
    f"{len(visual_pool):,}"
)

print(
    f"Indirect visual readout:    "
    f"{len(readout_population):,}"
)


assert len(readout_population) == (
    len(visual_pool)
    - len(input_neurons)
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
# Episode
# ============================================================

def run_episode(token: str, seed: int):

    brain.reset(seed=seed)
    trace.reset()

    activity = []


    # Warmup
    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        activity.append(
            np.asarray(
                trace.observe(fired)
            ).copy()
        )


    # Token stimulation
    token_neurons = sensory_map.neurons_for(
        token
    )

    for _ in range(TOKEN_STEPS):

        fired = brain.step(
            inject=[
                (
                    token_neurons,
                    AMPLITUDE,
                )
            ]
        )

        activity.append(
            np.asarray(
                trace.observe(fired)
            ).copy()
        )


    # Propagation
    for _ in range(POST_STEPS):

        fired = brain.step()

        activity.append(
            np.asarray(
                trace.observe(fired)
            ).copy()
        )


    activity = np.stack(activity)


    stimulus_start = WARMUP_STEPS

    stimulus_end = (
        WARMUP_STEPS
        + TOKEN_STEPS
    )


    stim = activity[
        stimulus_start:stimulus_end
    ]


    mean_stimulus = stim.mean(
        axis=0
    )

    max_stimulus = stim.max(
        axis=0
    )

    final_stimulus = activity[
        stimulus_end - 1
    ]

    final_trace = activity[
        -1
    ]


    return np.concatenate(
        [
            mean_stimulus,
            max_stimulus,
            final_stimulus,
            final_trace,
        ]
    )


# ============================================================
# Generate features
# ============================================================

features = []
labels = []
seed_labels = []


total = (
    len(tokens)
    * len(SEEDS)
)

counter = 0


print()
print("Running indirect visual episodes...")
print()


for token in tokens:

    for seed in SEEDS:

        counter += 1

        print(
            f"[{counter:02}/{total}] "
            f"token={token:10} "
            f"seed={seed}"
        )


        feature = run_episode(
            token,
            seed,
        )


        features.append(
            feature
        )

        labels.append(
            token
        )

        seed_labels.append(
            seed
        )


# ============================================================
# Save
# ============================================================

X = np.stack(features)

labels = np.asarray(labels)

seed_labels = np.asarray(
    seed_labels
)


print()
print("========================================")
print("        INDIRECT FEATURE DATASET")
print("========================================")
print()

print(
    f"Episodes:           {X.shape[0]}"
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


assert np.isfinite(X).all()


OUTPUT = (
    "artifacts/features/"
    "token_visual_indirect_features.npz"
)


np.savez(
    OUTPUT,

    X=X,
    labels=labels,
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
    "✅ Indirect visual experiment completed"
)