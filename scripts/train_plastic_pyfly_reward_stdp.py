from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Phase 2 — Plastic PyFly
# Step 79B — First real reward-modulated synaptic learning run
#
# This is the first task-learning experiment that leaves
# recurrent MaleCNS synapses changed across training examples.
#
# Learning rule:
#
#   pre/post spike timing
#          ↓
#   synaptic eligibility trace
#          ×
#   global token reward (+1 correct, -1 wrong)
#          ↓
#   sign-preserving recurrent-weight update
#
# IMPORTANT:
#   * Original MaleCNS files on disk are NEVER modified.
#   * Only the Step-79A selected recurrent synapse subset moves.
#   * The decoder is fixed BEFORE plastic training.
#   * Training uses generator_train only.
#   * Monitoring uses generator_validation only.
#   * The old final holdout is NOT opened in this script.
#   * 3 epochs and all plasticity hyperparameters below are
#     precommitted for this pilot; there is no tuning here.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"


# ============================================================
# Development data only
# ============================================================

TRAIN_GENERATOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_train.jsonl"
)

VALIDATION_GENERATOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_validation.jsonl"
)

BASELINE_TRAIN_FEATURE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_train.npz"
)


# ============================================================
# Frozen PyFly neural mapping
# ============================================================

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)


# ============================================================
# Step-79A plasticity infrastructure
# ============================================================

PLASTICITY_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "plasticity"
)

SMOKE_TEST_MANIFEST_FILE = (
    PLASTICITY_DIR
    / "plasticity_smoke_test.json"
)

PLASTIC_SUBSET_FILE = (
    PLASTICITY_DIR
    / "plastic_synapse_subset.npz"
)


# ============================================================
# Step-79B outputs
# ============================================================

FIXED_DECODER_FILE = (
    PLASTICITY_DIR
    / "phase2_train_only_fixed_decoder.joblib"
)

TRAINING_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_reward_plasticity_training.json"
)

FINAL_PLASTIC_WEIGHTS_FILE = (
    PLASTICITY_DIR
    / "phase2_epoch_03_plastic_weights.npz"
)


# ============================================================
# Frozen Step-62 stimulation settings
# ============================================================

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS_PER_CHANNEL = 16
FEEDBACK_NEURONS_PER_TOKEN = 32

PROMPT_TOKEN_STEPS = 8
PROMPT_GAP_STEPS = 2

FEEDBACK_TOKEN_STEPS = 8
FEEDBACK_GAP_STEPS = 1

WARMUP_STEPS = 10

AMPLITUDE = 0.8
TRACE_TAU = 0.25

MAP_SEED = 123

FEATURE_BLOCK_NAMES = [
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
]

EXPECTED_FEATURE_DIMENSIONS = 41_111
EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938


# ============================================================
# Fixed decoder used ONLY as the global reward readout
#
# Same winning Step-62 family/configuration, but fitted on
# generator_train ONLY so validation remains a development
# monitor for this Phase-2 pilot.
# ============================================================

DECODER_PCA_COMPONENTS = 100
DECODER_C = 0.1
DECODER_RANDOM_STATE = 42


# ============================================================
# PRECOMMITTED plasticity pilot hyperparameters
#
# These are engineering values, NOT claimed biological values.
# ============================================================

TRAINING_EPOCHS = 3

TRAINING_ORDER_SEED = 79_100

TRAIN_NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

VALIDATION_NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

# Spike-history trace:
# recent spikes have more influence.
NEURON_TRACE_DECAY = 0.85

# Synaptic eligibility trace:
# pre-before-post causal evidence persists briefly.
ELIGIBILITY_DECAY = 0.90

# A smaller anti-causal component gives an STDP-like
# depression term for post-before-pre timing.
ANTI_CAUSAL_STRENGTH = 0.25

# Maximum relative change per reward event is about ±2%.
LEARNING_RATE = 0.02

CORRECT_REWARD = +1.0
INCORRECT_REWARD = -1.0

# Keep each selected synapse within these bounds relative to
# its ORIGINAL Step-79A magnitude.
MIN_RELATIVE_MAGNITUDE = 0.50
MAX_RELATIVE_MAGNITUDE = 1.50

ELIGIBILITY_EPS = 1e-8

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def normalized_relative(
    path: Path,
) -> str:

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).replace(
        "\\",
        "/",
    )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Load exact frozen PyFly mapping
# ============================================================

def load_frozen_dual_map(
    path: Path,
) -> DualChannelMap:

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "prompt_tokens",
        "prompt_positions",
        "prompt_neurons",
        "feedback_tokens",
        "feedback_neurons",
        "prompt_neurons_per_channel",
        "feedback_neurons_per_token",
        "max_prompt_positions",
        "seed",
    }

    missing = (
        required
        -
        set(
            data.files
        )
    )

    if missing:

        raise ValueError(
            f"Frozen mapping missing: "
            f"{sorted(missing)}"
        )

    prompt_tokens = (
        data[
            "prompt_tokens"
        ].astype(str)
    )

    prompt_positions = (
        data[
            "prompt_positions"
        ].astype(
            np.int64
        )
    )

    prompt_neurons = (
        data[
            "prompt_neurons"
        ].astype(
            np.int64
        )
    )

    feedback_tokens = (
        data[
            "feedback_tokens"
        ].astype(str)
    )

    feedback_neurons = (
        data[
            "feedback_neurons"
        ].astype(
            np.int64
        )
    )

    prompt_mapping = {}

    for row, (
        token,
        position,
    ) in enumerate(
        zip(
            prompt_tokens,
            prompt_positions,
        )
    ):

        prompt_mapping[
            (
                str(
                    token
                ),
                int(
                    position
                ),
            )
        ] = (
            prompt_neurons[
                row
            ].copy()
        )

    feedback_mapping = {}

    for row, token in enumerate(
        feedback_tokens
    ):

        feedback_mapping[
            str(
                token
            )
        ] = (
            feedback_neurons[
                row
            ].copy()
        )

    dual_map = DualChannelMap(
        prompt_mapping=(
            prompt_mapping
        ),

        feedback_mapping=(
            feedback_mapping
        ),

        prompt_neurons_per_channel=int(
            np.asarray(
                data[
                    "prompt_neurons_per_channel"
                ]
            ).item()
        ),

        feedback_neurons_per_token=int(
            np.asarray(
                data[
                    "feedback_neurons_per_token"
                ]
            ).item()
        ),

        max_prompt_positions=int(
            np.asarray(
                data[
                    "max_prompt_positions"
                ]
            ).item()
        ),

        seed=int(
            np.asarray(
                data[
                    "seed"
                ]
            ).item()
        ),
    )

    if (
        dual_map.prompt_neurons_per_channel
        !=
        PROMPT_NEURONS_PER_CHANNEL
    ):

        raise ValueError(
            "Prompt mapping width changed."
        )

    if (
        dual_map.feedback_neurons_per_token
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):

        raise ValueError(
            "Feedback mapping width changed."
        )

    if (
        dual_map.max_prompt_positions
        !=
        MAX_PROMPT_POSITIONS
    ):

        raise ValueError(
            "Prompt-position maximum changed."
        )

    if (
        dual_map.seed
        !=
        MAP_SEED
    ):

        raise ValueError(
            "Neural mapping seed changed."
        )

    return dual_map


# ============================================================
# Group teacher-forcing rows by logical prompt
# ============================================================

def group_by_example(
    samples,
):

    grouped = defaultdict(
        list
    )

    for sample in samples:

        grouped[
            sample.example_id
        ].append(
            sample
        )

    for example_id in grouped:

        grouped[
            example_id
        ].sort(
            key=lambda sample:
                sample.target_position
        )

        positions = [
            sample.target_position

            for sample
            in grouped[
                example_id
            ]
        ]

        expected = list(
            range(
                len(
                    positions
                )
            )
        )

        if positions != expected:

            raise ValueError(
                f"Non-contiguous target positions "
                f"for {example_id}: {positions}"
            )

    return grouped


# ============================================================
# Fit fixed train-only decoder BEFORE plastic learning
# ============================================================

def fit_fixed_reward_decoder():

    if not BASELINE_TRAIN_FEATURE_FILE.exists():

        raise FileNotFoundError(
            "Missing frozen baseline train "
            f"features: "
            f"{BASELINE_TRAIN_FEATURE_FILE}"
        )

    data = np.load(
        BASELINE_TRAIN_FEATURE_FILE,
        allow_pickle=False,
    )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    y = data[
        "targets"
    ].astype(str)

    if (
        X.ndim
        !=
        2
        or
        X.shape[
            1
        ]
        !=
        EXPECTED_FEATURE_DIMENSIONS
    ):

        raise ValueError(
            "Unexpected frozen Step-62 "
            f"train feature shape: {X.shape}"
        )

    pipeline = Pipeline(
        [
            (
                "scale",

                StandardScaler(),
            ),

            (
                "pca",

                PCA(
                    n_components=(
                        DECODER_PCA_COMPONENTS
                    ),

                    svd_solver=(
                        "randomized"
                    ),

                    random_state=(
                        DECODER_RANDOM_STATE
                    ),
                ),
            ),

            (
                "classifier",

                LogisticRegression(
                    C=DECODER_C,

                    solver="lbfgs",

                    class_weight=(
                        "balanced"
                    ),

                    max_iter=5000,

                    random_state=(
                        DECODER_RANDOM_STATE
                    ),
                ),
            ),
        ]
    )

    pipeline.fit(
        X,
        y,
    )

    return pipeline


# ============================================================
# Reward-modulated STDP-like eligibility engine
# ============================================================

class RewardPlasticity:

    def __init__(
        self,
        brain: FlyBrain,
        edge_offsets: np.ndarray,
        pre_neurons: np.ndarray,
        post_neurons: np.ndarray,
        original_weights: np.ndarray,
    ) -> None:

        if brain.device != "cuda":

            raise RuntimeError(
                "Step 79B requires CUDA."
            )

        import cupy as cp

        self.cp = cp
        self.brain = brain

        self.edge_offsets = cp.asarray(
            edge_offsets,
            dtype=cp.int64,
        )

        self.pre_neurons = cp.asarray(
            pre_neurons,
            dtype=cp.int64,
        )

        self.post_neurons = cp.asarray(
            post_neurons,
            dtype=cp.int64,
        )

        self.original_weights = cp.asarray(
            original_weights,
            dtype=cp.float32,
        )

        self.original_sign = cp.sign(
            self.original_weights
        )

        self.original_magnitude = cp.abs(
            self.original_weights
        )

        self.minimum_magnitude = (
            self.original_magnitude
            *
            np.float32(
                MIN_RELATIVE_MAGNITUDE
            )
        )

        self.maximum_magnitude = (
            self.original_magnitude
            *
            np.float32(
                MAX_RELATIVE_MAGNITUDE
            )
        )

        self.neuron_trace = cp.zeros(
            brain.n,
            dtype=cp.float32,
        )

        self.spike_mask = cp.zeros(
            brain.n,
            dtype=cp.float32,
        )

        self.edge_eligibility = cp.zeros(
            len(
                edge_offsets
            ),
            dtype=cp.float32,
        )

    # --------------------------------------------------------
    # Start a new logical training episode.
    # --------------------------------------------------------

    def reset_episode(
        self,
    ) -> None:

        self.neuron_trace.fill(
            0
        )

        self.spike_mask.fill(
            0
        )

        self.edge_eligibility.fill(
            0
        )

    # --------------------------------------------------------
    # Observe one brain step.
    #
    # fired is a NumPy vector of neuron indices returned by
    # FlyBrain.step().
    #
    # Causal term:
    #     recent PRE spike × current POST spike
    #
    # Anti-causal term:
    #     recent POST spike × current PRE spike
    # --------------------------------------------------------

    def observe_spikes(
        self,
        fired: np.ndarray,
    ) -> None:

        cp = self.cp

        self.neuron_trace *= np.float32(
            NEURON_TRACE_DECAY
        )

        self.edge_eligibility *= np.float32(
            ELIGIBILITY_DECAY
        )

        self.spike_mask.fill(
            0
        )

        if len(
            fired
        ):

            fired_gpu = cp.asarray(
                fired,
                dtype=cp.int64,
            )

            self.spike_mask[
                fired_gpu
            ] = np.float32(
                1.0
            )

        pre_history = self.neuron_trace[
            self.pre_neurons
        ]

        post_history = self.neuron_trace[
            self.post_neurons
        ]

        pre_now = self.spike_mask[
            self.pre_neurons
        ]

        post_now = self.spike_mask[
            self.post_neurons
        ]

        self.edge_eligibility += (
            pre_history
            *
            post_now
        )

        self.edge_eligibility -= (
            np.float32(
                ANTI_CAUSAL_STRENGTH
            )
            *
            post_history
            *
            pre_now
        )

        if len(
            fired
        ):

            self.neuron_trace[
                fired_gpu
            ] += np.float32(
                1.0
            )

    # --------------------------------------------------------
    # Apply one global reward signal.
    #
    # Only current GPU recurrent weights are changed.
    # Original disk files remain untouched.
    # --------------------------------------------------------

    def apply_reward(
        self,
        reward: float,
    ) -> dict:

        cp = self.cp

        max_abs = float(
            cp.max(
                cp.abs(
                    self.edge_eligibility
                )
            ).get()
        )

        if (
            not np.isfinite(
                max_abs
            )
            or
            max_abs
            <=
            ELIGIBILITY_EPS
        ):

            self.edge_eligibility.fill(
                0
            )

            return {
                "eligible_max_abs": (
                    max_abs
                ),

                "changed_synapses": 0,

                "mean_abs_delta": 0.0,
            }

        normalized = (
            self.edge_eligibility
            /
            np.float32(
                max_abs
            )
        )

        current = self.brain._W.data[
            self.edge_offsets
        ]

        current_magnitude = cp.abs(
            current
        )

        relative_factor = (
            np.float32(
                1.0
            )
            +
            np.float32(
                LEARNING_RATE
                *
                reward
            )
            *
            normalized
        )

        new_magnitude = (
            current_magnitude
            *
            relative_factor
        )

        new_magnitude = cp.clip(
            new_magnitude,
            self.minimum_magnitude,
            self.maximum_magnitude,
        )

        new_weights = (
            self.original_sign
            *
            new_magnitude
        )

        delta = (
            new_weights
            -
            current
        )

        changed_synapses = int(
            cp.count_nonzero(
                cp.abs(
                    delta
                )
                >
                np.float32(
                    1e-10
                )
            ).get()
        )

        mean_abs_delta = float(
            cp.mean(
                cp.abs(
                    delta
                )
            ).get()
        )

        self.brain._W.data[
            self.edge_offsets
        ] = (
            new_weights
        )

        # Reward has now consumed this eligibility trace.
        # Keep the neuron history, but do not apply the same
        # edge eligibility again to the next token reward.
        self.edge_eligibility.fill(
            0
        )

        return {
            "eligible_max_abs": (
                max_abs
            ),

            "changed_synapses": (
                changed_synapses
            ),

            "mean_abs_delta": (
                mean_abs_delta
            ),
        }

    # --------------------------------------------------------
    # Weight statistics / checkpointing
    # --------------------------------------------------------

    def current_weights_cpu(
        self,
    ) -> np.ndarray:

        return (
            self.brain._W.data[
                self.edge_offsets
            ].get()
        )

    def statistics(
        self,
    ) -> dict:

        cp = self.cp

        current = self.brain._W.data[
            self.edge_offsets
        ]

        current_magnitude = cp.abs(
            current
        )

        ratio = (
            current_magnitude
            /
            self.original_magnitude
        )

        changed = cp.abs(
            current
            -
            self.original_weights
        )

        return {
            "selected_synapses": int(
                len(
                    self.edge_offsets
                )
            ),

            "mean_relative_magnitude": float(
                cp.mean(
                    ratio
                ).get()
            ),

            "std_relative_magnitude": float(
                cp.std(
                    ratio
                ).get()
            ),

            "min_relative_magnitude": float(
                cp.min(
                    ratio
                ).get()
            ),

            "max_relative_magnitude": float(
                cp.max(
                    ratio
                ).get()
            ),

            "changed_from_original": int(
                cp.count_nonzero(
                    changed
                    >
                    np.float32(
                        1e-8
                    )
                ).get()
            ),

            "mean_abs_change_from_original": float(
                cp.mean(
                    changed
                ).get()
            ),
        }


# ============================================================
# Main training experiment
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        "       PHASE 2 - PLASTIC PYFLY"
    )

    print(
        "       STEP 79B - REWARD SYNAPTIC LEARNING"
    )

    print(
        "===================================================="
    )

    print()

    # ========================================================
    # Verify Step 79A
    # ========================================================

    required = [
        TRAIN_GENERATOR_FILE,
        VALIDATION_GENERATOR_FILE,
        BASELINE_TRAIN_FEATURE_FILE,
        MAPPING_FILE,
        SMOKE_TEST_MANIFEST_FILE,
        PLASTIC_SUBSET_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: "
                f"{path}"
            )

    smoke = load_json(
        SMOKE_TEST_MANIFEST_FILE
    )

    if (
        smoke.get(
            "status"
        )
        !=
        "PLASTICITY_INFRASTRUCTURE_VERIFIED"
    ):

        raise ValueError(
            "Step 79A plasticity smoke test "
            "has not passed."
        )

    print(
        "Step 79A plasticity infrastructure: PASS"
    )

    print(
        f"Training epochs:                    "
        f"{TRAINING_EPOCHS}"
    )

    print(
        f"Learning rate:                      "
        f"{LEARNING_RATE}"
    )

    print(
        f"Correct / incorrect reward:         "
        f"{CORRECT_REWARD:+.1f} / "
        f"{INCORRECT_REWARD:+.1f}"
    )

    print(
        f"Allowed weight magnitude range:     "
        f"{MIN_RELATIVE_MAGNITUDE:.2f}x .. "
        f"{MAX_RELATIVE_MAGNITUDE:.2f}x original"
    )

    print()

    print(
        "IMPORTANT: old final holdout is NOT "
        "used anywhere in Step 79B."
    )

    # ========================================================
    # Load development teacher-forcing data
    # ========================================================

    train_samples = (
        load_generator_jsonl(
            TRAIN_GENERATOR_FILE
        )
    )

    validation_samples = (
        load_generator_jsonl(
            VALIDATION_GENERATOR_FILE
        )
    )

    train_grouped = (
        group_by_example(
            train_samples
        )
    )

    validation_grouped = (
        group_by_example(
            validation_samples
        )
    )

    print()

    print(
        f"Training logical prompts:           "
        f"{len(train_grouped)}"
    )

    print(
        f"Training teacher-forcing samples:   "
        f"{len(train_samples)}"
    )

    print(
        f"Validation logical prompts:         "
        f"{len(validation_grouped)}"
    )

    print(
        f"Validation teacher-forcing samples: "
        f"{len(validation_samples)}"
    )

    # ========================================================
    # Fixed reward decoder
    # ========================================================

    print()

    print(
        "Fitting fixed TRAIN-ONLY reward decoder "
        "before synaptic learning..."
    )

    decoder = (
        fit_fixed_reward_decoder()
    )

    FIXED_DECODER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        decoder,
        FIXED_DECODER_FILE,
    )

    print(
        "Fixed reward decoder:               "
        "Balanced Logistic PCA=100 C=0.1"
    )

    print(
        "Reward decoder fitting data:         "
        "generator_train ONLY"
    )

    # ========================================================
    # Load frozen neural map + Step-79A synapse checkpoint
    # ========================================================

    dual_map = (
        load_frozen_dual_map(
            MAPPING_FILE
        )
    )

    plastic_checkpoint = np.load(
        PLASTIC_SUBSET_FILE,
        allow_pickle=False,
    )

    edge_offsets = (
        plastic_checkpoint[
            "edge_offsets"
        ].astype(
            np.int64
        )
    )

    pre_neurons = (
        plastic_checkpoint[
            "pre_neurons"
        ].astype(
            np.int64
        )
    )

    post_neurons = (
        plastic_checkpoint[
            "post_neurons"
        ].astype(
            np.int64
        )
    )

    original_weights = (
        plastic_checkpoint[
            "original_weights"
        ].astype(
            np.float32
        )
    )

    if not (
        len(
            edge_offsets
        )
        ==
        len(
            pre_neurons
        )
        ==
        len(
            post_neurons
        )
        ==
        len(
            original_weights
        )
    ):

        raise ValueError(
            "Step-79A plastic subset arrays "
            "have inconsistent lengths."
        )

    print()

    print(
        f"Plastic synapses loaded:            "
        f"{len(edge_offsets):,}"
    )

    # ========================================================
    # Load real MaleCNS on CUDA
    # ========================================================

    print()

    print(
        "Loading real MaleCNS on CUDA..."
    )

    brain = FlyBrain(
        device=DEVICE,
        seed=0,
    )

    if (
        brain.n
        !=
        EXPECTED_NEURONS
    ):

        raise ValueError(
            "Unexpected MaleCNS neuron count."
        )

    if (
        brain._W.nnz
        !=
        EXPECTED_CONNECTIONS
    ):

        raise ValueError(
            "Unexpected MaleCNS recurrent edge count."
        )

    visual_pool = np.asarray(
        brain.cells(
            [
                "visual_projection"
            ]
        ),
        dtype=np.int64,
    )

    input_neurons = (
        dual_map.all_input_neurons()
    )

    readout_population = np.setdiff1d(
        visual_pool,
        input_neurons,
    )

    if (
        len(
            input_neurons
        )
        !=
        EXPECTED_DIRECT_INPUT_NEURONS
    ):

        raise ValueError(
            "Direct input population changed."
        )

    if (
        len(
            readout_population
        )
        !=
        EXPECTED_READOUT_NEURONS
    ):

        raise ValueError(
            "Indirect readout population changed."
        )

    print(
        f"Neurons:                             "
        f"{brain.n:,}"
    )

    print(
        f"Recurrent edges:                     "
        f"{brain._W.nnz:,}"
    )

    print(
        f"Indirect readout neurons:            "
        f"{len(readout_population):,}"
    )

    # ========================================================
    # Verify Step-79A edge offsets against THIS fresh brain
    # ========================================================

    W_indptr = (
        brain._W.indptr.get().astype(
            np.int64,
            copy=False,
        )
    )

    W_indices = (
        brain._W.indices.get().astype(
            np.int64,
            copy=False,
        )
    )

    if np.max(
        edge_offsets
    ) >= brain._W.nnz:

        raise ValueError(
            "Plastic edge offset exceeds "
            "current GPU sparse matrix."
        )

    current_pre = W_indices[
        edge_offsets
    ]

    current_post = (
        np.searchsorted(
            W_indptr,
            edge_offsets,
            side="right",
        )
        -
        1
    )

    if not np.array_equal(
        current_pre,
        pre_neurons,
    ):

        raise ValueError(
            "Step-79A plastic presynaptic "
            "indices no longer match."
        )

    if not np.array_equal(
        current_post,
        post_neurons,
    ):

        raise ValueError(
            "Step-79A plastic postsynaptic "
            "indices no longer match."
        )

    import cupy as cp

    gpu_offsets = cp.asarray(
        edge_offsets,
        dtype=cp.int64,
    )

    fresh_weights = (
        brain._W.data[
            gpu_offsets
        ].get()
    )

    if not np.array_equal(
        fresh_weights,
        original_weights,
    ):

        raise ValueError(
            "Fresh MaleCNS does not match "
            "the Step-79A original plastic weights."
        )

    print(
        "Step-79A synapse offsets verified:    PASS"
    )

    # ========================================================
    # Trace + plasticity engine
    # ========================================================

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    plasticity = RewardPlasticity(
        brain=brain,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_neurons=post_neurons,
        original_weights=original_weights,
    )

    # ========================================================
    # One brain step helper
    # ========================================================

    def brain_step(
        inject=(),
        learn=False,
    ):

        fired = brain.step(
            inject=inject
        )

        state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()

        if learn:

            plasticity.observe_spikes(
                fired
            )

        return (
            fired,
            state,
        )

    # ========================================================
    # Frozen Step-62 prompt stimulation
    # ========================================================

    def stimulate_prompt(
        prompt_tokens,
        learn,
    ):

        states = []

        for (
            position,
            token,
        ) in enumerate(
            prompt_tokens
        ):

            neurons = (
                dual_map.prompt_neurons_for(
                    token,
                    position,
                )
            )

            for _ in range(
                PROMPT_TOKEN_STEPS
            ):

                _, state = brain_step(
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ],
                    learn=learn,
                )

                states.append(
                    state
                )

            for _ in range(
                PROMPT_GAP_STEPS
            ):

                _, state = brain_step(
                    learn=learn,
                )

                states.append(
                    state
                )

        states = np.stack(
            states
        )

        return (
            states.mean(
                axis=0
            ),

            states.max(
                axis=0
            ),

            states[
                -1
            ],
        )

    # ========================================================
    # Frozen Step-62 feedback stimulation
    # ========================================================

    def stimulate_feedback(
        token,
        learn,
    ):

        neurons = (
            dual_map.feedback_neurons_for(
                token
            )
        )

        states = []

        for _ in range(
            FEEDBACK_TOKEN_STEPS
        ):

            _, state = brain_step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ],
                learn=learn,
            )

            states.append(
                state
            )

        states = np.stack(
            states
        )

        feedback_mean = (
            states.mean(
                axis=0
            )
        )

        feedback_max = (
            states.max(
                axis=0
            )
        )

        feedback_final = (
            states[
                -1
            ]
        )

        delayed_state = (
            feedback_final
        )

        for _ in range(
            FEEDBACK_GAP_STEPS
        ):

            _, delayed_state = brain_step(
                learn=learn,
            )

        return (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        )

    # ========================================================
    # One teacher-forcing episode
    # ========================================================

    def run_episode(
        example_samples,
        neural_seed,
        learn,
    ):

        brain.reset(
            seed=neural_seed
        )

        trace.reset()

        # ----------------------------------------------------
        # Warm-up is NOT eligible for learning.
        # ----------------------------------------------------

        for _ in range(
            WARMUP_STEPS
        ):

            fired = brain.step()

            trace.observe(
                fired
            )

        if learn:

            plasticity.reset_episode()

        prompt_tokens = (
            example_samples[
                0
            ].prompt_tokens
        )

        (
            prompt_mean,
            prompt_max,
            prompt_final,
        ) = stimulate_prompt(
            prompt_tokens,
            learn=learn,
        )

        rows = []

        for (
            position,
            sample,
        ) in enumerate(
            example_samples
        ):

            feedback_token = (
                "BOS"
                if position == 0
                else
                example_samples[
                    position - 1
                ].target_token
            )

            if (
                sample.context_tokens[
                    -1
                ]
                !=
                feedback_token
            ):

                raise ValueError(
                    "Teacher-forcing context mismatch "
                    f"for {sample.sample_id}"
                )

            (
                feedback_mean,
                feedback_max,
                feedback_final,
                delayed_state,
            ) = stimulate_feedback(
                feedback_token,
                learn=learn,
            )

            feature = np.concatenate(
                [
                    prompt_mean,
                    prompt_max,
                    prompt_final,
                    feedback_mean,
                    feedback_max,
                    feedback_final,
                    delayed_state,
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            if (
                feature.shape[
                    0
                ]
                !=
                EXPECTED_FEATURE_DIMENSIONS
            ):

                raise RuntimeError(
                    "Plastic feature dimensions changed."
                )

            if not np.isfinite(
                feature
            ).all():

                raise RuntimeError(
                    "Non-finite plastic feature."
                )

            predicted = str(
                decoder.predict(
                    feature.reshape(
                        1,
                        -1,
                    )
                )[
                    0
                ]
            )

            correct = (
                predicted
                ==
                sample.target_token
            )

            update_info = None

            if learn:

                reward = (
                    CORRECT_REWARD
                    if correct
                    else
                    INCORRECT_REWARD
                )

                update_info = (
                    plasticity.apply_reward(
                        reward
                    )
                )

            rows.append(
                {
                    "target": (
                        sample.target_token
                    ),

                    "prediction": (
                        predicted
                    ),

                    "position": int(
                        sample.target_position
                    ),

                    "correct": bool(
                        correct
                    ),

                    "update": (
                        update_info
                    ),
                }
            )

        return rows

    # ========================================================
    # Validation evaluation
    # ========================================================

    def evaluate_validation():

        targets = []
        predictions = []
        positions = []

        for example_id in sorted(
            validation_grouped
        ):

            example_samples = (
                validation_grouped[
                    example_id
                ]
            )

            for neural_seed in (
                VALIDATION_NEURAL_SEEDS
            ):

                rows = run_episode(
                    example_samples=(
                        example_samples
                    ),
                    neural_seed=(
                        neural_seed
                    ),
                    learn=False,
                )

                for row in rows:

                    targets.append(
                        row[
                            "target"
                        ]
                    )

                    predictions.append(
                        row[
                            "prediction"
                        ]
                    )

                    positions.append(
                        row[
                            "position"
                        ]
                    )

        targets = np.asarray(
            targets
        )

        predictions = np.asarray(
            predictions
        )

        positions = np.asarray(
            positions,
            dtype=np.int32,
        )

        semantic_mask = np.isin(
            positions,
            SEMANTIC_POSITIONS,
        )

        position_accuracy = {}

        for position in sorted(
            np.unique(
                positions
            )
        ):

            mask = (
                positions
                ==
                position
            )

            position_accuracy[
                str(
                    int(
                        position
                    )
                )
            ] = float(
                accuracy_score(
                    targets[
                        mask
                    ],
                    predictions[
                        mask
                    ],
                )
            )

        return {
            "rows": int(
                len(
                    targets
                )
            ),

            "top1_accuracy": float(
                accuracy_score(
                    targets,
                    predictions,
                )
            ),

            "macro_f1": float(
                f1_score(
                    targets,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

            "semantic_accuracy": float(
                accuracy_score(
                    targets[
                        semantic_mask
                    ],
                    predictions[
                        semantic_mask
                    ],
                )
            ),

            "semantic_macro_f1": float(
                f1_score(
                    targets[
                        semantic_mask
                    ],
                    predictions[
                        semantic_mask
                    ],
                    average="macro",
                    zero_division=0,
                )
            ),

            "accuracy_by_position": (
                position_accuracy
            ),
        }

    # ========================================================
    # Save one plastic-weight checkpoint
    # ========================================================

    def save_checkpoint(
        epoch: int,
    ) -> Path:

        path = (
            PLASTICITY_DIR
            /
            f"phase2_epoch_{epoch:02}_"
            f"plastic_weights.npz"
        )

        current_weights = (
            plasticity.current_weights_cpu()
        )

        np.savez(
            path,

            edge_offsets=(
                edge_offsets
            ),

            pre_neurons=(
                pre_neurons
            ),

            post_neurons=(
                post_neurons
            ),

            original_weights=(
                original_weights
            ),

            plastic_weights=(
                current_weights
            ),

            epoch=np.asarray(
                epoch,
                dtype=np.int32,
            ),

            learning_rate=np.asarray(
                LEARNING_RATE,
                dtype=np.float32,
            ),

            neuron_trace_decay=np.asarray(
                NEURON_TRACE_DECAY,
                dtype=np.float32,
            ),

            eligibility_decay=np.asarray(
                ELIGIBILITY_DECAY,
                dtype=np.float32,
            ),

            anti_causal_strength=np.asarray(
                ANTI_CAUSAL_STRENGTH,
                dtype=np.float32,
            ),
        )

        return path

    # ========================================================
    # BEFORE-TRAINING validation
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "       VALIDATION BEFORE SYNAPTIC LEARNING"
    )

    print(
        "===================================================="
    )

    before_validation = (
        evaluate_validation()
    )

    print()

    print(
        f"Top-1:              "
        f"{before_validation['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:           "
        f"{before_validation['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Semantic accuracy:  "
        f"{before_validation['semantic_accuracy'] * 100:.2f}%"
    )

    epoch_zero_checkpoint = (
        save_checkpoint(
            0
        )
    )

    # ========================================================
    # ACTUAL SYNAPTIC TRAINING
    # ========================================================

    history = [
        {
            "epoch": 0,

            "training": None,

            "validation": (
                before_validation
            ),

            "plastic_weights": (
                plasticity.statistics()
            ),

            "checkpoint": (
                normalized_relative(
                    epoch_zero_checkpoint
                )
            ),
        }
    ]

    logical_train_ids = sorted(
        train_grouped.keys()
    )

    for epoch in range(
        1,
        TRAINING_EPOCHS + 1,
    ):

        print()

        print(
            "===================================================="
        )

        print(
            f"          PLASTIC TRAINING EPOCH {epoch}/"
            f"{TRAINING_EPOCHS}"
        )

        print(
            "===================================================="
        )

        print()

        rng = np.random.default_rng(
            TRAINING_ORDER_SEED
            +
            epoch
        )

        order = rng.permutation(
            len(
                logical_train_ids
            )
        )

        total_tokens = 0
        correct_tokens = 0

        reward_sum = 0.0

        update_events = 0
        changed_synapse_events = 0
        mean_abs_delta_sum = 0.0

        for episode_number, order_index in enumerate(
            order,
            start=1,
        ):

            example_id = (
                logical_train_ids[
                    int(
                        order_index
                    )
                ]
            )

            example_samples = (
                train_grouped[
                    example_id
                ]
            )

            neural_seed = (
                TRAIN_NEURAL_SEEDS[
                    (
                        epoch
                        -
                        1
                        +
                        episode_number
                        -
                        1
                    )
                    %
                    len(
                        TRAIN_NEURAL_SEEDS
                    )
                ]
            )

            rows = run_episode(
                example_samples=(
                    example_samples
                ),
                neural_seed=(
                    neural_seed
                ),
                learn=True,
            )

            episode_correct = sum(
                int(
                    row[
                        "correct"
                    ]
                )

                for row
                in rows
            )

            total_tokens += len(
                rows
            )

            correct_tokens += (
                episode_correct
            )

            for row in rows:

                reward = (
                    CORRECT_REWARD
                    if row[
                        "correct"
                    ]
                    else
                    INCORRECT_REWARD
                )

                reward_sum += reward

                update = row[
                    "update"
                ]

                if update is not None:

                    update_events += 1

                    changed_synapse_events += int(
                        update[
                            "changed_synapses"
                        ]
                    )

                    mean_abs_delta_sum += float(
                        update[
                            "mean_abs_delta"
                        ]
                    )

            print(
                f"[{episode_number:02}/"
                f"{len(logical_train_ids):02}] "
                f"{example_id:18} "
                f"brain_seed={neural_seed} "
                f"token_acc="
                f"{episode_correct / len(rows) * 100:6.2f}%"
            )

        training_accuracy = (
            correct_tokens
            /
            total_tokens
        )

        # ====================================================
        # Validation after this epoch
        # ====================================================

        print()

        print(
            f"Evaluating validation after epoch "
            f"{epoch}..."
        )

        validation_metrics = (
            evaluate_validation()
        )

        checkpoint_path = (
            save_checkpoint(
                epoch
            )
        )

        weight_stats = (
            plasticity.statistics()
        )

        epoch_training = {
            "teacher_forcing_tokens": int(
                total_tokens
            ),

            "correct_tokens_before_update": int(
                correct_tokens
            ),

            "accuracy_before_update": float(
                training_accuracy
            ),

            "reward_sum": float(
                reward_sum
            ),

            "update_events": int(
                update_events
            ),

            "sum_changed_synapse_events": int(
                changed_synapse_events
            ),

            "mean_update_abs_delta": float(
                (
                    mean_abs_delta_sum
                    /
                    update_events
                )
                if update_events
                else
                0.0
            ),
        }

        history.append(
            {
                "epoch": int(
                    epoch
                ),

                "training": (
                    epoch_training
                ),

                "validation": (
                    validation_metrics
                ),

                "plastic_weights": (
                    weight_stats
                ),

                "checkpoint": (
                    normalized_relative(
                        checkpoint_path
                    )
                ),
            }
        )

        print()

        print(
            f"Epoch {epoch} training Top-1 "
            f"(before each online update): "
            f"{training_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Top-1:             "
            f"{validation_metrics['top1_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Macro F1:          "
            f"{validation_metrics['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Validation semantic accuracy: "
            f"{validation_metrics['semantic_accuracy'] * 100:.2f}%"
        )

        print(
            f"Changed synapses vs original: "
            f"{weight_stats['changed_from_original']:,}"
        )

        print(
            f"Mean weight magnitude ratio:  "
            f"{weight_stats['mean_relative_magnitude']:.6f}x"
        )

        print(
            f"Weight ratio min/max:         "
            f"{weight_stats['min_relative_magnitude']:.4f}x / "
            f"{weight_stats['max_relative_magnitude']:.4f}x"
        )

    # ========================================================
    # Final Step-79B summary
    # ========================================================

    final_validation = history[
        -1
    ][
        "validation"
    ]

    final_weights = history[
        -1
    ][
        "plastic_weights"
    ]

    validation_top1_change = (
        final_validation[
            "top1_accuracy"
        ]
        -
        before_validation[
            "top1_accuracy"
        ]
    )

    validation_macro_change = (
        final_validation[
            "macro_f1"
        ]
        -
        before_validation[
            "macro_f1"
        ]
    )

    validation_semantic_change = (
        final_validation[
            "semantic_accuracy"
        ]
        -
        before_validation[
            "semantic_accuracy"
        ]
    )

    # ========================================================
    # Save training manifest
    # ========================================================

    output = {
        "phase": 2,

        "step": "79B",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FIRST_REWARD_MODULATED_SYNAPTIC_TRAINING_COMPLETE"
        ),

        "scientific_scope": (
            "development pilot only; old final holdout "
            "not used"
        ),

        "fixed_reward_decoder": {
            "file": (
                normalized_relative(
                    FIXED_DECODER_FILE
                )
            ),

            "sha256": (
                sha256_file(
                    FIXED_DECODER_FILE
                )
            ),

            "training_data": (
                "generator_train frozen Step-62 features only"
            ),

            "configuration": (
                "Balanced Logistic PCA=100 C=0.1"
            ),
        },

        "plasticity": {
            "epochs": (
                TRAINING_EPOCHS
            ),

            "training_order_seed": (
                TRAINING_ORDER_SEED
            ),

            "training_neural_seeds": (
                TRAIN_NEURAL_SEEDS
            ),

            "validation_neural_seeds": (
                VALIDATION_NEURAL_SEEDS
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "correct_reward": (
                CORRECT_REWARD
            ),

            "incorrect_reward": (
                INCORRECT_REWARD
            ),

            "neuron_trace_decay": (
                NEURON_TRACE_DECAY
            ),

            "eligibility_decay": (
                ELIGIBILITY_DECAY
            ),

            "anti_causal_strength": (
                ANTI_CAUSAL_STRENGTH
            ),

            "minimum_relative_magnitude": (
                MIN_RELATIVE_MAGNITUDE
            ),

            "maximum_relative_magnitude": (
                MAX_RELATIVE_MAGNITUDE
            ),
        },

        "plastic_subset": {
            "file": (
                normalized_relative(
                    PLASTIC_SUBSET_FILE
                )
            ),

            "sha256": (
                sha256_file(
                    PLASTIC_SUBSET_FILE
                )
            ),

            "selected_synapses": int(
                len(
                    edge_offsets
                )
            ),
        },

        "validation_before": (
            before_validation
        ),

        "validation_after": (
            final_validation
        ),

        "validation_change": {
            "top1_pp": float(
                validation_top1_change
                *
                100
            ),

            "macro_f1_pp": float(
                validation_macro_change
                *
                100
            ),

            "semantic_accuracy_pp": float(
                validation_semantic_change
                *
                100
            ),
        },

        "final_weight_statistics": (
            final_weights
        ),

        "history": (
            history
        ),

        "final_checkpoint": (
            normalized_relative(
                FINAL_PLASTIC_WEIGHTS_FILE
            )
        ),

        "mapping_sha256": (
            sha256_file(
                MAPPING_FILE
            )
        ),

        "disk_connectome_modified": (
            False
        ),

        "rules": [
            (
                "Only recurrent weights in the selected "
                "Step-79A synapse subset were plastic."
            ),

            (
                "The reward decoder was fitted before "
                "plastic training and then kept fixed."
            ),

            (
                "generator_train alone drove synaptic updates."
            ),

            (
                "generator_validation was monitored but did "
                "not drive synaptic updates."
            ),

            (
                "The old final holdout was not opened or "
                "scored in Step 79B."
            ),

            (
                "This is an engineering reward-modulated "
                "STDP-like rule, not a claim of reproducing "
                "the biological mushroom-body dopamine rule."
            ),
        ],
    }

    TRAINING_METRICS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "===================================================="
    )

    print(
        "       STEP 79B - ACTUAL LEARNING RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Validation BEFORE learning:"
    )

    print(
        f"  Top-1:              "
        f"{before_validation['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{before_validation['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{before_validation['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        f"Validation AFTER {TRAINING_EPOCHS} epochs:"
    )

    print(
        f"  Top-1:              "
        f"{final_validation['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{final_validation['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{final_validation['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Change caused by recurrent synaptic learning:"
    )

    print(
        f"  Top-1:              "
        f"{validation_top1_change * 100:+.2f} pp"
    )

    print(
        f"  Macro F1:           "
        f"{validation_macro_change * 100:+.2f} pp"
    )

    print(
        f"  Semantic accuracy:  "
        f"{validation_semantic_change * 100:+.2f} pp"
    )

    print()

    print(
        f"Recurrent synapses changed from original: "
        f"{final_weights['changed_from_original']:,}"
    )

    print(
        f"Final mean magnitude ratio:             "
        f"{final_weights['mean_relative_magnitude']:.6f}x"
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{normalized_relative(FIXED_DECODER_FILE)}"
    )

    print(
        "  "
        f"{normalized_relative(TRAINING_METRICS_FILE)}"
    )

    for epoch in range(
        0,
        TRAINING_EPOCHS + 1,
    ):

        checkpoint = (
            PLASTICITY_DIR
            /
            f"phase2_epoch_{epoch:02}_"
            f"plastic_weights.npz"
        )

        print(
            "  "
            f"{normalized_relative(checkpoint)}"
        )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 79B COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "The fly's recurrent synapses have now "
        "actually been trained and the learned "
        "weights are saved."
    )

    print()

    print(
        "Next: inspect whether validation improved. "
        "Then Step 79C will freeze/tune the Phase-2 "
        "learning protocol using DEVELOPMENT data only."
    )


if __name__ == "__main__":
    main()
