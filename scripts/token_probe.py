import numpy as np

from flybrain import FlyBrain, Trace
from pyfly.dataset import load_dataset
from pyfly.sensory_map import create_disjoint_sensory_map


DEVICE = "cuda"
SEED = 123

WARMUP_STEPS = 10
TOKEN_STEPS = 4
POST_STEPS = 10

AMPLITUDE = 0.8
NEURONS_PER_TOKEN = 24


print("Loading FlyBrain on GPU...")

brain = FlyBrain(
    device=DEVICE,
    seed=SEED,
)

print(f"Brain loaded on: {brain.device}")


# --------------------------------------------------
# Build the exact same deterministic sensory mapping
# --------------------------------------------------

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)

prompt_tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})

visual_pool = brain.cells(
    ["visual_projection"]
)

sensory_map = create_disjoint_sensory_map(
    tokens=prompt_tokens,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=42,
)


# --------------------------------------------------
# Record all descending neurons
# --------------------------------------------------

descending = brain.cells(
    ["descending_neuron"]
)

trace = Trace(
    brain,
    idx=descending,
    tau=0.1,
)

print(
    f"Recording {len(descending):,} "
    "descending neurons"
)


# --------------------------------------------------
# Run one isolated token
# --------------------------------------------------

def run_token(token=None):
    brain.reset(seed=SEED)
    trace.reset()

    activity = []

    # Warm-up
    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        features = trace.observe(fired)

        activity.append(features)


    # Token stimulation
    for _ in range(TOKEN_STEPS):

        if token is None:
            injection = []
        else:
            injection = [
                (
                    sensory_map.neurons_for(token),
                    AMPLITUDE,
                )
            ]

        fired = brain.step(
            inject=injection
        )

        features = trace.observe(fired)

        activity.append(features)


    # Post-stimulus period
    for _ in range(POST_STEPS):

        fired = brain.step()

        features = trace.observe(fired)

        activity.append(features)


    return np.stack(activity)


# --------------------------------------------------
# Run three conditions
# --------------------------------------------------

print()
print("Running CONTROL...")
control_activity = run_token(None)

print("Running HELLO...")
hello_activity = run_token("hello")

print("Running ADD...")
add_activity = run_token("add")


# --------------------------------------------------
# Extract feature vectors
# --------------------------------------------------

FINAL_WINDOW = 5

control_feature = control_activity[-FINAL_WINDOW:].mean(axis=0)
hello_feature = hello_activity[-FINAL_WINDOW:].mean(axis=0)
add_feature = add_activity[-FINAL_WINDOW:].mean(axis=0)


# --------------------------------------------------
# Comparison functions
# --------------------------------------------------

def euclidean(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=== PyFly Neural Response Test ===")

print(
    f"Control -> Hello distance: "
    f"{euclidean(control_feature, hello_feature):.4f}"
)

print(
    f"Control -> Add distance:   "
    f"{euclidean(control_feature, add_feature):.4f}"
)

print(
    f"Hello -> Add distance:     "
    f"{euclidean(hello_feature, add_feature):.4f}"
)

print()

print(
    f"Hello/Add cosine similarity: "
    f"{cosine_similarity(hello_feature, add_feature):.4f}"
)

print()

print(
    "Active control features:",
    np.count_nonzero(control_feature),
)

print(
    "Active hello features:  ",
    np.count_nonzero(hello_feature),
)

print(
    "Active add features:    ",
    np.count_nonzero(add_feature),
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

np.savez(
    "artifacts/features/token_probe.npz",

    control_activity=control_activity,
    hello_activity=hello_activity,
    add_activity=add_activity,

    control_feature=control_feature,
    hello_feature=hello_feature,
    add_feature=add_feature,

    descending_neurons=descending,
)


print()
print(
    "Saved:"
    " artifacts/features/token_probe.npz"
)

print()
print("✅ Token neural-response experiment completed")