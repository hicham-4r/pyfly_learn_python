import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.sensory_map import create_disjoint_sensory_map


# ============================================================
# PyFly Token PCA Experiment
# ============================================================

DEVICE = "cuda"

# Neural mapping
NEURONS_PER_TOKEN = 64
MAPPING_SEED = 42

# Neural noise seeds
SEEDS = [0, 1, 2, 3, 4]

# Simulation timing
WARMUP_STEPS = 10
TOKEN_STEPS = 8
POST_STEPS = 4

# Token stimulation strength
AMPLITUDE = 0.8

# Trace settings
TRACE_TAU = 0.1


# ============================================================
# Load FlyBrain
# ============================================================

print()
print("========================================")
print("          PyFly TOKEN PCA")
print("========================================")
print()

print("Loading FlyBrain on RTX GPU...")

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

print("Brain loaded successfully.")

print()
print(f"Neurons:     {brain.n:,}")
print(f"Connections: {len(brain.weights):,}")


# ============================================================
# Load Dataset
# ============================================================

examples = load_dataset(
    "data/raw/python_mre.jsonl"
)


tokens = sorted(
    {
        token
        for example in examples
        for token in example.prompt_tokens
    }
)


print()
print("========================================")
print("         PROMPT VOCABULARY")
print("========================================")
print()

print(f"Number of tokens: {len(tokens)}")
print()

for token in tokens:
    print(f"  {token}")


# ============================================================
# Visual Projection Input Neurons
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print()
print("========================================")
print("        INPUT NEURON POPULATION")
print("========================================")
print()

print(
    f"Visual projection neurons available: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Build Deterministic Token -> Neuron Mapping
# ============================================================

sensory_map = create_disjoint_sensory_map(
    tokens=tokens,
    neuron_pool=visual_pool,
    neurons_per_token=NEURONS_PER_TOKEN,
    seed=MAPPING_SEED,
)


print()
print(
    f"Neurons per token: {NEURONS_PER_TOKEN}"
)

print(
    f"Total token neurons used: "
    f"{len(tokens) * NEURONS_PER_TOKEN}"
)


# ============================================================
# Descending Neurons
# ============================================================

descending = np.asarray(
    brain.cells(
        ["descending_neuron"]
    )
)


print()
print("========================================")
print("        OUTPUT NEURON POPULATION")
print("========================================")
print()

print(
    f"Descending neurons recorded: "
    f"{len(descending):,}"
)


# ============================================================
# Neural Trace
# ============================================================

trace = Trace(
    brain,
    idx=descending,
    tau=TRACE_TAU,
)


# ============================================================
# Run One Token Episode
# ============================================================

def run_episode(token: str, seed: int):

    # Reset complete fly state
    brain.reset(
        seed=seed
    )

    # Reset neural trace
    trace.reset()

    activity = []

    # --------------------------------------------------------
    # 1. Warm-up
    # --------------------------------------------------------

    for _ in range(WARMUP_STEPS):

        fired = brain.step()

        trace_vector = trace.observe(
            fired
        )

        activity.append(
            np.asarray(trace_vector).copy()
        )


    # --------------------------------------------------------
    # 2. Token stimulation
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

        trace_vector = trace.observe(
            fired
        )

        activity.append(
            np.asarray(trace_vector).copy()
        )


    # --------------------------------------------------------
    # 3. Short propagation period
    # --------------------------------------------------------

    for _ in range(POST_STEPS):

        fired = brain.step()

        trace_vector = trace.observe(
            fired
        )

        activity.append(
            np.asarray(trace_vector).copy()
        )


    activity = np.stack(
        activity
    )


    # ========================================================
    # Feature Extraction
    # ========================================================

    stimulus_start = WARMUP_STEPS

    stimulus_end = (
        WARMUP_STEPS
        + TOKEN_STEPS
    )


    # Average activity DURING token injection
    mean_stimulus = activity[
        stimulus_start:stimulus_end
    ].mean(
        axis=0
    )


    # Maximum response during token injection
    max_stimulus = activity[
        stimulus_start:stimulus_end
    ].max(
        axis=0
    )


    # State immediately after stimulation
    final_stimulus = activity[
        stimulus_end - 1
    ]


    # Final state after short propagation
    final_trace = activity[
        -1
    ]


    # Combine neural features
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
# Generate Neural Feature Dataset
# ============================================================

features = []
labels = []
seed_labels = []


print()
print("========================================")
print("       RUNNING NEURAL EPISODES")
print("========================================")
print()


total_episodes = (
    len(tokens)
    * len(SEEDS)
)

episode_number = 0


for token in tokens:

    for seed in SEEDS:

        episode_number += 1

        print(
            f"[{episode_number:02}/{total_episodes}] "
            f"Token={token:10} "
            f"Seed={seed}"
        )


        feature = run_episode(
            token=token,
            seed=seed,
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
# Convert to NumPy
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
    f"Feature dimensions: {X.shape[1]}"
)

print(
    f"Finite values:      {np.isfinite(X).all()}"
)

print(
    f"Mean feature value: {X.mean():.6f}"
)

print(
    f"Feature std:        {X.std():.6f}"
)


assert np.isfinite(X).all(), \
    "Feature matrix contains NaN or Inf"


# ============================================================
# Save Neural Features
# ============================================================

feature_output = (
    "artifacts/features/"
    "token_pca_features.npz"
)


np.savez(
    feature_output,

    X=X,

    labels=labels,

    seeds=seed_labels,

    neurons_per_token=NEURONS_PER_TOKEN,

    amplitude=AMPLITUDE,

    token_steps=TOKEN_STEPS,

    post_steps=POST_STEPS,
)


print()
print(
    f"Saved neural features: "
    f"{feature_output}"
)


# ============================================================
# PCA
# ============================================================

print()
print("========================================")
print("                PCA")
print("========================================")
print()


pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X
)


pc1 = (
    pca.explained_variance_ratio_[0]
    * 100
)

pc2 = (
    pca.explained_variance_ratio_[1]
    * 100
)

total_variance = (
    pca.explained_variance_ratio_.sum()
    * 100
)


print(
    f"PC1 explained variance: "
    f"{pc1:.2f}%"
)

print(
    f"PC2 explained variance: "
    f"{pc2:.2f}%"
)

print(
    f"Total explained variance: "
    f"{total_variance:.2f}%"
)


# ============================================================
# Calculate Token Cluster Centers
# ============================================================

print()
print("========================================")
print("          PCA TOKEN CENTERS")
print("========================================")
print()


for token in tokens:

    mask = (
        labels == token
    )

    center = X_pca[
        mask
    ].mean(
        axis=0
    )


    print(
        f"{token:12} "
        f"PC1={center[0]:9.3f} "
        f"PC2={center[1]:9.3f}"
    )


# ============================================================
# PCA Visualization
# ============================================================

plt.figure(
    figsize=(13, 9)
)


for token in tokens:

    mask = (
        labels == token
    )


    plt.scatter(
        X_pca[
            mask,
            0
        ],

        X_pca[
            mask,
            1
        ],

        label=token,

        s=80,

        alpha=0.8,
    )


    center_x = X_pca[
        mask,
        0
    ].mean()

    center_y = X_pca[
        mask,
        1
    ].mean()


    plt.text(
        center_x,
        center_y,
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
    "PyFly — Token Representations "
    "in MaleCNS Reservoir"
)


plt.legend(
    bbox_to_anchor=(
        1.02,
        1,
    ),
    loc="upper left",
)


plt.grid(
    alpha=0.2
)

plt.tight_layout()


plot_output = (
    "artifacts/plots/"
    "token_probe_pca.png"
)


plt.savefig(
    plot_output,
    dpi=200,
    bbox_inches="tight",
)


plt.close()


print()
print(
    f"Saved PCA plot: "
    f"{plot_output}"
)


# ============================================================
# Complete
# ============================================================

print()
print("========================================")
print("          EXPERIMENT COMPLETE")
print("========================================")
print()

print(
    "✅ Multi-token neural experiment PASSED"
)

print()
print(
    "Configuration:"
)

print(
    f"  neurons/token : "
    f"{NEURONS_PER_TOKEN}"
)

print(
    f"  amplitude     : "
    f"{AMPLITUDE}"
)

print(
    f"  token steps   : "
    f"{TOKEN_STEPS}"
)

print(
    f"  post steps    : "
    f"{POST_STEPS}"
)

print(
    f"  neural seeds  : "
    f"{len(SEEDS)}"
)

print(
    f"  feature size  : "
    f"{X.shape[1]}"
)