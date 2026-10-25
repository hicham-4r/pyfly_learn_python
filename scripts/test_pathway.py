from flybrain import FlyBrain

print("Loading FlyBrain on GPU...")
brain = FlyBrain(device="cuda", seed=0)

# Looming-sensitive neurons on the left side
left_loom = brain.cells(["LC4", "LPLC2"], side="L")

# DNp01 = giant-fiber escape command neuron
giant_fiber = brain.cells(["DNp01"], side="L")

print(f"LC4 + LPLC2 neurons: {len(left_loom)}")
print(f"DNp01 neurons:       {len(giant_fiber)}")
print()

giant_fiber_set = set(giant_fiber)
spike_times = []

# Run for 50 steps = about 1 second
for step in range(50):
    fired = brain.step(
        inject=[
            (left_loom, 0.8)
        ]
    )

    if giant_fiber_set.intersection(fired):
        time_seconds = step * brain.dt
        spike_times.append(time_seconds)
        print(f"DNp01 fired at {time_seconds:.3f} seconds")

print()
print("=== RESULT ===")

if spike_times:
    print(f"✅ Pathway works")
    print(f"DNp01 spike count: {len(spike_times)}")
    print(f"First spike: {spike_times[0]:.3f} s")
else:
    print("⚠️ DNp01 did not spike in this run")