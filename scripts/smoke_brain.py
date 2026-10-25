from flybrain import FlyBrain

print("Loading FlyBrain on GPU...")

brain = FlyBrain(device="cuda", seed=0)

print()
print("=== PyFly Brain Information ===")
print(f"Neurons:       {brain.n:,}")
print(f"Connections:   {len(brain.weights):,}")

visual_projection = brain.cells(["visual_projection"])
descending = brain.cells(["descending_neuron"])

print(f"Visual projection neurons: {len(visual_projection):,}")
print(f"Descending neurons:        {len(descending):,}")

assert brain.n == 166_700
assert len(brain.weights) == 25_582_938
assert len(visual_projection) > 0
assert len(descending) > 0

print()
print("✅ PyFly GPU brain smoke test PASSED")