import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
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
POST_STEPS = 4

AMPLITUDE = 0.8
TRACE_TAU = 0.1


# ============================================================
# Load Brain
# ============================================================

print()
print("========================================")
print("   PyFly VISUAL-PROJECTION READOUT")
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
# Dataset / vocabulary
# ============================================================

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)

tokens = sorted({
    token
    for example in examples
    for token in example.prompt_tokens
})

print()
print(f"Token classes: {len(tokens)}")
print(tokens)


# ============================================================
# Visual-projection population
# ============================================================

visual_pool = np.asarray(
    brain.cells(["visual_projection"])
)

print()
print(
    f"Visual projection neurons: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Deterministic sensory mapping
# ============================================================

sensory_map = create_disjoint_sensory_map(
    tokens=tokens,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


# ============================================================
# IMPORTANT:
# We now RECORD visual_projection itself instead of descending
# ============================================================

readout_population = visual_pool


trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)

print(
    f"Recording {len(readout_population):,} "
    "visual-projection neurons"
)


# ============================================================
# Run episode
# ============================================================

def run_episode(token: str, seed: int):

    brain.reset(seed=seed)
    trace.reset()

    activity = []

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        vector = trace.observe(fired)

        activity.append(
            np.asarray(vector).copy()
        )


    # --------------------------------------------------------
    # Stimulate token
    # --------------------------------------------------------

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

        vector = trace.observe(fired)

        activity.append(
            np.asarray(vector).copy()
        )


    # --------------------------------------------------------
    # Propagation
    # --------------------------------------------------------

    for _ in range(POST_STEPS):

        fired = brain.step()

        vector = trace.observe(fired)

        activity.append(
            np.asarray(vector).copy()
        )


    activity = np.stack(
        activity
    )


    # ========================================================
    # Feature extraction
    # ========================================================

    stimulus_start = WARMUP_STEPS

    stimulus_end = (
        WARMUP_STEPS
        + TOKEN_STEPS
    )


    stimulus_activity = activity[
        stimulus_start:stimulus_end
    ]


    mean_stimulus = stimulus_activity.mean(
        axis=0
    )

    max_stimulus = stimulus_activity.max(
        axis=0
    )

    final_stimulus = activity[
        stimulus_end - 1
    ]

    final_trace = activity[-1]


    feature = np.concatenate(
        [
            mean_stimulus,
            max_stimulus,
            final_stimulus,
            final_trace,
        ]
    )


    return feature


# ============================================================
# Generate all episodes
# ============================================================

features = []
labels = []
seed_labels = []


total = len(tokens) * len(SEEDS)

counter = 0


print()
print("Running experiments...")
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
# Create feature matrix
# ============================================================

X = np.stack(
    features
)

labels = np.asarray(
    labels
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


# ============================================================
# Save separately from descending experiment
# ============================================================

OUTPUT = (
    "artifacts/features/"
    "token_visual_features.npz"
)


np.savez(
    OUTPUT,

    X=X,
    labels=labels,
    seeds=seed_labels,

    readout_type="visual_projection",

    neurons_per_token=NEURONS_PER_TOKEN,
    amplitude=AMPLITUDE,
    token_steps=TOKEN_STEPS,
)


print()
print(
    f"Saved features: {OUTPUT}"
)


# ============================================================
# PCA visualization
# ============================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X
)


print()
print("========================================")
print("                PCA")
print("========================================")
print()

print(
    f"PC1: "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"PC2: "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    f"Total: "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)


plt.figure(
    figsize=(13, 9)
)


for token in tokens:

    mask = labels == token

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=80,
        label=token,
    )

    center = X_pca[
        mask
    ].mean(
        axis=0
    )

    plt.text(
        center[0],
        center[1],
        token,
        fontsize=9,
        fontweight="bold",
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "PyFly — Visual Projection Token Responses"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()


PLOT_OUTPUT = (
    "artifacts/plots/"
    "token_visual_pca.png"
)


plt.savefig(
    PLOT_OUTPUT,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print()
print(
    f"Saved plot: {PLOT_OUTPUT}"
)

print()
print(
    "✅ Visual-projection experiment completed"
)