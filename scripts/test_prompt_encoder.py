import numpy as np

from flybrain import FlyBrain

from pyfly.dataset import load_dataset
from pyfly.encoder import build_prompt_schedule
from pyfly.sensory_map import create_disjoint_sensory_map


print("Loading FlyBrain...")

brain = FlyBrain(
    device="cuda",
    seed=0,
)

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)

tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})

visual_pool = np.asarray(
    brain.cells(["visual_projection"])
)

sensory_map = create_disjoint_sensory_map(
    tokens=tokens,
    neuron_pool=visual_pool,
    neurons_per_token=64,
    seed=42,
)


# Test with:
# add a b

prompt = [
    "add",
    "a",
    "b",
]


schedule = build_prompt_schedule(
    prompt_tokens=prompt,
    sensory_map=sensory_map,
    token_steps=8,
    gap_steps=2,
    amplitude=0.8,
)


print()
print("==============================")
print("    PyFly Prompt Schedule")
print("==============================")
print()

print("Prompt:", prompt)

print()


total_steps = 0


for index, event in enumerate(schedule):

    total_steps += event.steps

    print(
        f"{index:02} | "
        f"{event.token:8} | "
        f"steps={event.steps:2} | "
        f"neurons={len(event.neurons):3} | "
        f"amplitude={event.amplitude}"
    )


print()
print(f"Total simulation steps: {total_steps}")

print()

assert total_steps == 30

assert schedule[0].token == "add"
assert schedule[2].token == "a"
assert schedule[4].token == "b"

assert len(schedule[0].neurons) == 64

print("✅ Temporal prompt encoder PASSED")