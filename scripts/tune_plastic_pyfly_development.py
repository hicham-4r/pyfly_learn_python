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

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Phase 2 — Plastic PyFly
# Step 79C — DEVELOPMENT-ONLY plasticity tuning
#
# Step 79B proved that recurrent MaleCNS synapses can change,
# but its first precommitted rule harmed validation.
#
# This script:
#
#   1. NEVER opens the old final holdout.
#   2. Uses generator_train for synaptic updates.
#   3. Uses generator_validation for development selection.
#   4. Restores ORIGINAL Step-79A synapses before every
#      Stage-1 candidate.
#   5. Tries six conservative one-epoch candidates.
#   6. Continues ONLY the best one-epoch candidate to
#      epochs 2 and 3.
#   7. Selects the best development checkpoint among
#      baseline / epoch 1 / epoch 2 / epoch 3.
#
# If no plastic checkpoint beats the no-learning baseline,
# the script reports that plasticity has not yet helped.
#
# No final-holdout scoring occurs here.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"


# ============================================================
# DEVELOPMENT data only
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


# ============================================================
# Frozen encoder / Step 79A / Step 79B artifacts
# ============================================================

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)

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

FIXED_DECODER_FILE = (
    PLASTICITY_DIR
    / "phase2_train_only_fixed_decoder.joblib"
)

STEP79B_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_reward_plasticity_training.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_plasticity_tuning.json"
)

BEST_WEIGHTS_FILE = (
    PLASTICITY_DIR
    / "phase2_best_development_plastic_weights.npz"
)

FROZEN_PROTOCOL_FILE = (
    PLASTICITY_DIR
    / "phase2_frozen_learning_protocol.json"
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

EXPECTED_FEATURE_DIMENSIONS = 41_111

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Fixed eligibility mechanics
#
# We tune only:
#   learning_rate
#   incorrect_reward
#   anti_causal_strength
#
# Everything else remains fixed for Step 79C.
# ============================================================

NEURON_TRACE_DECAY = 0.85
ELIGIBILITY_DECAY = 0.90

CORRECT_REWARD = +1.0

MIN_RELATIVE_MAGNITUDE = 0.50
MAX_RELATIVE_MAGNITUDE = 1.50

ELIGIBILITY_EPS = 1e-8


# ============================================================
# PRECOMMITTED Stage-1 candidate grid
#
# Step 79B used:
#   lr=0.02, wrong=-1.0, anti=0.25
#
# It harmed validation, so all new candidates are deliberately
# more conservative.
# ============================================================

CANDIDATES = [
    {
        "name": "positive_only_lr_0.002",
        "learning_rate": 0.002,
        "incorrect_reward": 0.0,
        "anti_causal_strength": 0.0,
    },
    {
        "name": "positive_only_lr_0.005",
        "learning_rate": 0.005,
        "incorrect_reward": 0.0,
        "anti_causal_strength": 0.0,
    },
    {
        "name": "mild_negative_lr_0.002",
        "learning_rate": 0.002,
        "incorrect_reward": -0.10,
        "anti_causal_strength": 0.0,
    },
    {
        "name": "mild_negative_lr_0.005",
        "learning_rate": 0.005,
        "incorrect_reward": -0.10,
        "anti_causal_strength": 0.0,
    },
    {
        "name": "mild_stdp_lr_0.002",
        "learning_rate": 0.002,
        "incorrect_reward": -0.10,
        "anti_causal_strength": 0.10,
    },
    {
        "name": "mild_stdp_lr_0.005",
        "learning_rate": 0.005,
        "incorrect_reward": -0.10,
        "anti_causal_strength": 0.10,
    },
]


# ============================================================
# Training/evaluation reproducibility
# ============================================================

TRAINING_ORDER_SEED = 79_200

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

MAX_SELECTED_EPOCHS = 3


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


# ============================================================
# Load exact frozen mapping
# ============================================================

def load_frozen_dual_map(
    path: Path,
) -> DualChannelMap:

    data = np.load(
        path,
        allow_pickle=False,
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

    result = DualChannelMap(
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
        result.prompt_neurons_per_channel
        !=
        PROMPT_NEURONS_PER_CHANNEL
    ):

        raise ValueError(
            "Prompt mapping width changed."
        )

    if (
        result.feedback_neurons_per_token
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):

        raise ValueError(
            "Feedback mapping width changed."
        )

    if (
        result.max_prompt_positions
        !=
        MAX_PROMPT_POSITIONS
    ):

        raise ValueError(
            "Prompt-position maximum changed."
        )

    if (
        result.seed
        !=
        MAP_SEED
    ):

        raise ValueError(
            "Mapping seed changed."
        )

    return result


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
# Candidate ranking
#
# Primary: semantic accuracy
# Secondary: macro F1
# Tertiary: overall Top-1
# ============================================================

def selection_key(
    metrics: dict,
):

    return (
        metrics[
            "semantic_accuracy"
        ],
        metrics[
            "macro_f1"
        ],
        metrics[
            "top1_accuracy"
        ],
    )


# ============================================================
# Parameterized reward-plasticity engine
# ============================================================

class RewardPlasticity:

    def __init__(
        self,
        brain: FlyBrain,
        edge_offsets: np.ndarray,
        pre_neurons: np.ndarray,
        post_neurons: np.ndarray,
        original_weights: np.ndarray,
        learning_rate: float,
        incorrect_reward: float,
        anti_causal_strength: float,
    ) -> None:

        if brain.device != "cuda":

            raise RuntimeError(
                "Step 79C requires CUDA."
            )

        import cupy as cp

        self.cp = cp
        self.brain = brain

        self.learning_rate = float(
            learning_rate
        )

        self.incorrect_reward = float(
            incorrect_reward
        )

        self.anti_causal_strength = float(
            anti_causal_strength
        )

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
                self.anti_causal_strength
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

    def apply_reward(
        self,
        correct: bool,
    ) -> dict:

        reward = (
            CORRECT_REWARD
            if correct
            else
            self.incorrect_reward
        )

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
            or
            reward
            ==
            0.0
        ):

            self.edge_eligibility.fill(
                0
            )

            return {
                "reward": float(
                    reward
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
                self.learning_rate
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
        ] = new_weights

        self.edge_eligibility.fill(
            0
        )

        return {
            "reward": float(
                reward
            ),

            "changed_synapses": (
                changed_synapses
            ),

            "mean_abs_delta": (
                mean_abs_delta
            ),
        }

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

        changed = cp.abs(
            current
            -
            self.original_weights
        )

        ratio = (
            cp.abs(
                current
            )
            /
            self.original_magnitude
        )

        return {
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

            "mean_relative_magnitude": float(
                cp.mean(
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
        }


# ============================================================
# Main
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
        "       STEP 79C - DEVELOPMENT TUNING"
    )

    print(
        "===================================================="
    )

    print()

    required = [
        TRAIN_GENERATOR_FILE,
        VALIDATION_GENERATOR_FILE,
        MAPPING_FILE,
        SMOKE_TEST_MANIFEST_FILE,
        PLASTIC_SUBSET_FILE,
        FIXED_DECODER_FILE,
        STEP79B_METRICS_FILE,
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
            "Step 79A has not passed."
        )

    step79b = load_json(
        STEP79B_METRICS_FILE
    )

    print(
        "Step 79A infrastructure:       PASS"
    )

    print(
        "Step 79B first pilot:          COMPLETE"
    )

    print()

    print(
        "IMPORTANT: this script does NOT open "
        "the old final holdout."
    )

    print()

    print(
        f"Stage-1 candidates:           "
        f"{len(CANDIDATES)}"
    )

    for candidate in CANDIDATES:

        print(
            f"  {candidate['name']:28} "
            f"lr={candidate['learning_rate']:.4f} "
            f"wrong={candidate['incorrect_reward']:+.2f} "
            f"anti={candidate['anti_causal_strength']:.2f}"
        )

    # ========================================================
    # Data
    # ========================================================

    train_samples = load_generator_jsonl(
        TRAIN_GENERATOR_FILE
    )

    validation_samples = (
        load_generator_jsonl(
            VALIDATION_GENERATOR_FILE
        )
    )

    train_grouped = group_by_example(
        train_samples
    )

    validation_grouped = (
        group_by_example(
            validation_samples
        )
    )

    if (
        len(
            train_grouped
        )
        !=
        36
    ):

        raise ValueError(
            "Expected 36 training prompts."
        )

    if (
        len(
            validation_grouped
        )
        !=
        6
    ):

        raise ValueError(
            "Expected 6 validation prompts."
        )

    # ========================================================
    # Fixed decoder from Step 79B
    # ========================================================

    decoder = joblib.load(
        FIXED_DECODER_FILE
    )

    # ========================================================
    # Neural map + original plastic checkpoint
    # ========================================================

    dual_map = load_frozen_dual_map(
        MAPPING_FILE
    )

    checkpoint = np.load(
        PLASTIC_SUBSET_FILE,
        allow_pickle=False,
    )

    edge_offsets = checkpoint[
        "edge_offsets"
    ].astype(
        np.int64
    )

    pre_neurons = checkpoint[
        "pre_neurons"
    ].astype(
        np.int64
    )

    post_neurons = checkpoint[
        "post_neurons"
    ].astype(
        np.int64
    )

    original_weights = checkpoint[
        "original_weights"
    ].astype(
        np.float32
    )

    # ========================================================
    # Real MaleCNS
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
            "Unexpected neuron count."
        )

    if (
        brain._W.nnz
        !=
        EXPECTED_CONNECTIONS
    ):

        raise ValueError(
            "Unexpected recurrent edge count."
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
            "Input population changed."
        )

    if (
        len(
            readout_population
        )
        !=
        EXPECTED_READOUT_NEURONS
    ):

        raise ValueError(
            "Readout population changed."
        )

    # ========================================================
    # Verify edge offsets against fresh MaleCNS
    # ========================================================

    W_indptr = brain._W.indptr.get().astype(
        np.int64,
        copy=False,
    )

    W_indices = brain._W.indices.get().astype(
        np.int64,
        copy=False,
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
            "Plastic presynaptic map changed."
        )

    if not np.array_equal(
        current_post,
        post_neurons,
    ):

        raise ValueError(
            "Plastic postsynaptic map changed."
        )

    import cupy as cp

    gpu_offsets = cp.asarray(
        edge_offsets,
        dtype=cp.int64,
    )

    fresh_original = brain._W.data[
        gpu_offsets
    ].get()

    if not np.array_equal(
        fresh_original,
        original_weights,
    ):

        raise ValueError(
            "Fresh MaleCNS selected weights "
            "do not match Step 79A."
        )

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    # ========================================================
    # Exact restore helper
    # ========================================================

    def restore_original_weights():

        brain._W.data[
            gpu_offsets
        ] = cp.asarray(
            original_weights,
            dtype=cp.float32,
        )

        restored = brain._W.data[
            gpu_offsets
        ].get()

        if not np.array_equal(
            restored,
            original_weights,
        ):

            raise RuntimeError(
                "Original plastic weights "
                "failed to restore exactly."
            )

    # ========================================================
    # Brain / feature helpers
    # ========================================================

    def brain_step(
        plasticity,
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

        return state

    def stimulate_prompt(
        prompt_tokens,
        plasticity,
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

                states.append(
                    brain_step(
                        plasticity,
                        inject=[
                            (
                                neurons,
                                AMPLITUDE,
                            )
                        ],
                        learn=learn,
                    )
                )

            for _ in range(
                PROMPT_GAP_STEPS
            ):

                states.append(
                    brain_step(
                        plasticity,
                        learn=learn,
                    )
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

    def stimulate_feedback(
        token,
        plasticity,
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

            states.append(
                brain_step(
                    plasticity,
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ],
                    learn=learn,
                )
            )

        states = np.stack(
            states
        )

        feedback_mean = states.mean(
            axis=0
        )

        feedback_max = states.max(
            axis=0
        )

        feedback_final = states[
            -1
        ]

        delayed_state = feedback_final

        for _ in range(
            FEEDBACK_GAP_STEPS
        ):

            delayed_state = brain_step(
                plasticity,
                learn=learn,
            )

        return (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        )

    def run_episode(
        example_samples,
        neural_seed,
        plasticity,
        learn,
    ):

        brain.reset(
            seed=neural_seed
        )

        trace.reset()

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
            plasticity,
            learn,
        )

        rows = []

        for position, sample in enumerate(
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

            (
                feedback_mean,
                feedback_max,
                feedback_final,
                delayed_state,
            ) = stimulate_feedback(
                feedback_token,
                plasticity,
                learn,
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
                    "Feature dimension changed."
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

            update = None

            if learn:

                update = plasticity.apply_reward(
                    correct
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
                        update
                    ),
                }
            )

        return rows

    # ========================================================
    # Development validation evaluation
    # ========================================================

    def evaluate_validation(
        plasticity,
    ):

        targets = []
        predictions = []
        positions = []

        for example_id in sorted(
            validation_grouped
        ):

            samples = validation_grouped[
                example_id
            ]

            for neural_seed in (
                VALIDATION_NEURAL_SEEDS
            ):

                rows = run_episode(
                    samples,
                    neural_seed,
                    plasticity,
                    False,
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
    # One training epoch
    # ========================================================

    logical_train_ids = sorted(
        train_grouped.keys()
    )

    def train_one_epoch(
        plasticity,
        epoch_number,
        candidate_index,
    ):

        rng = np.random.default_rng(
            TRAINING_ORDER_SEED
            +
            candidate_index
            *
            100
            +
            epoch_number
        )

        order = rng.permutation(
            len(
                logical_train_ids
            )
        )

        total = 0
        correct = 0

        reward_sum = 0.0
        update_events = 0
        changed_synapse_events = 0

        for episode_number, order_index in enumerate(
            order,
            start=1,
        ):

            example_id = logical_train_ids[
                int(
                    order_index
                )
            ]

            samples = train_grouped[
                example_id
            ]

            neural_seed = (
                TRAIN_NEURAL_SEEDS[
                    (
                        episode_number
                        +
                        epoch_number
                        -
                        2
                    )
                    %
                    len(
                        TRAIN_NEURAL_SEEDS
                    )
                ]
            )

            rows = run_episode(
                samples,
                neural_seed,
                plasticity,
                True,
            )

            for row in rows:

                total += 1

                correct += int(
                    row[
                        "correct"
                    ]
                )

                update = row[
                    "update"
                ]

                if update is not None:

                    reward_sum += float(
                        update[
                            "reward"
                        ]
                    )

                    update_events += 1

                    changed_synapse_events += int(
                        update[
                            "changed_synapses"
                        ]
                    )

        return {
            "tokens": int(
                total
            ),

            "accuracy_before_update": float(
                correct
                /
                total
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
        }

    # ========================================================
    # No-learning baseline
    # ========================================================

    restore_original_weights()

    baseline_engine = RewardPlasticity(
        brain=brain,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_neurons=post_neurons,
        original_weights=original_weights,
        learning_rate=0.0,
        incorrect_reward=0.0,
        anti_causal_strength=0.0,
    )

    baseline_validation = (
        evaluate_validation(
            baseline_engine
        )
    )

    print()

    print(
        "===================================================="
    )

    print(
        "          NO-LEARNING VALIDATION BASELINE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Top-1:              "
        f"{baseline_validation['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:           "
        f"{baseline_validation['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Semantic accuracy:  "
        f"{baseline_validation['semantic_accuracy'] * 100:.2f}%"
    )

    # ========================================================
    # Stage 1 — six one-epoch candidates
    # ========================================================

    stage1_results = []

    print()

    print(
        "===================================================="
    )

    print(
        "       STAGE 1 - SIX CONSERVATIVE CANDIDATES"
    )

    print(
        "===================================================="
    )

    for candidate_index, candidate in enumerate(
        CANDIDATES,
        start=1,
    ):

        print()

        print(
            f"[{candidate_index}/"
            f"{len(CANDIDATES)}] "
            f"{candidate['name']}"
        )

        restore_original_weights()

        plasticity = RewardPlasticity(
            brain=brain,
            edge_offsets=edge_offsets,
            pre_neurons=pre_neurons,
            post_neurons=post_neurons,
            original_weights=original_weights,
            learning_rate=(
                candidate[
                    "learning_rate"
                ]
            ),
            incorrect_reward=(
                candidate[
                    "incorrect_reward"
                ]
            ),
            anti_causal_strength=(
                candidate[
                    "anti_causal_strength"
                ]
            ),
        )

        training = train_one_epoch(
            plasticity,
            epoch_number=1,
            candidate_index=(
                candidate_index
            ),
        )

        validation = (
            evaluate_validation(
                plasticity
            )
        )

        weight_stats = (
            plasticity.statistics()
        )

        candidate_weights = (
            plasticity.current_weights_cpu()
        )

        result = {
            "candidate_index": int(
                candidate_index
            ),

            "candidate": (
                candidate
            ),

            "epoch": 1,

            "training": (
                training
            ),

            "validation": (
                validation
            ),

            "weight_statistics": (
                weight_stats
            ),

            "_weights": (
                candidate_weights
            ),
        }

        stage1_results.append(
            result
        )

        print(
            f"  Validation Top-1:     "
            f"{validation['top1_accuracy'] * 100:.2f}%"
        )

        print(
            f"  Validation Macro F1:  "
            f"{validation['macro_f1'] * 100:.2f}%"
        )

        print(
            f"  Validation semantic:  "
            f"{validation['semantic_accuracy'] * 100:.2f}%"
        )

        print(
            f"  Changed synapses:     "
            f"{weight_stats['changed_from_original']:,}"
        )

        print(
            f"  Mean magnitude ratio: "
            f"{weight_stats['mean_relative_magnitude']:.6f}x"
        )

    # ========================================================
    # Choose best Stage-1 candidate by development validation
    # ========================================================

    winner_stage1 = max(
        stage1_results,
        key=lambda row:
            selection_key(
                row[
                    "validation"
                ]
            ),
    )

    selected_candidate = winner_stage1[
        "candidate"
    ]

    selected_index = winner_stage1[
        "candidate_index"
    ]

    print()

    print(
        "===================================================="
    )

    print(
        "          STAGE-1 DEVELOPMENT WINNER"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Candidate:           "
        f"{selected_candidate['name']}"
    )

    print(
        f"Semantic accuracy:   "
        f"{winner_stage1['validation']['semantic_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:            "
        f"{winner_stage1['validation']['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Top-1:               "
        f"{winner_stage1['validation']['top1_accuracy'] * 100:.2f}%"
    )

    # ========================================================
    # Restore winner's epoch-1 weights, then continue only it
    # to epochs 2 and 3.
    # ========================================================

    brain._W.data[
        gpu_offsets
    ] = cp.asarray(
        winner_stage1[
            "_weights"
        ],
        dtype=cp.float32,
    )

    selected_engine = RewardPlasticity(
        brain=brain,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_neurons=post_neurons,
        original_weights=original_weights,
        learning_rate=(
            selected_candidate[
                "learning_rate"
            ]
        ),
        incorrect_reward=(
            selected_candidate[
                "incorrect_reward"
            ]
        ),
        anti_causal_strength=(
            selected_candidate[
                "anti_causal_strength"
            ]
        ),
    )

    selected_history = [
        {
            "epoch": 1,

            "training": (
                winner_stage1[
                    "training"
                ]
            ),

            "validation": (
                winner_stage1[
                    "validation"
                ]
            ),

            "weight_statistics": (
                winner_stage1[
                    "weight_statistics"
                ]
            ),

            "_weights": (
                winner_stage1[
                    "_weights"
                ].copy()
            ),
        }
    ]

    for epoch in range(
        2,
        MAX_SELECTED_EPOCHS + 1,
    ):

        print()

        print(
            f"Continuing winner: epoch "
            f"{epoch}/"
            f"{MAX_SELECTED_EPOCHS}"
        )

        training = train_one_epoch(
            selected_engine,
            epoch_number=(
                epoch
            ),
            candidate_index=(
                selected_index
            ),
        )

        validation = (
            evaluate_validation(
                selected_engine
            )
        )

        weight_stats = (
            selected_engine.statistics()
        )

        weights = (
            selected_engine.current_weights_cpu()
        )

        selected_history.append(
            {
                "epoch": int(
                    epoch
                ),

                "training": (
                    training
                ),

                "validation": (
                    validation
                ),

                "weight_statistics": (
                    weight_stats
                ),

                "_weights": (
                    weights
                ),
            }
        )

        print(
            f"  Validation Top-1:     "
            f"{validation['top1_accuracy'] * 100:.2f}%"
        )

        print(
            f"  Validation Macro F1:  "
            f"{validation['macro_f1'] * 100:.2f}%"
        )

        print(
            f"  Validation semantic:  "
            f"{validation['semantic_accuracy'] * 100:.2f}%"
        )

        print(
            f"  Changed synapses:     "
            f"{weight_stats['changed_from_original']:,}"
        )

    # ========================================================
    # Select best plastic checkpoint among epochs 1-3
    # ========================================================

    best_plastic = max(
        selected_history,
        key=lambda row:
            selection_key(
                row[
                    "validation"
                ]
            ),
    )

    # Compare against no-learning baseline.
    baseline_key = selection_key(
        baseline_validation
    )

    plastic_key = selection_key(
        best_plastic[
            "validation"
        ]
    )

    plastic_beats_baseline = bool(
        plastic_key
        >
        baseline_key
    )

    # ========================================================
    # Save best plastic weights even if they do not beat
    # baseline, so the experiment is reproducible.
    # ========================================================

    np.savez(
        BEST_WEIGHTS_FILE,

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
            best_plastic[
                "_weights"
            ]
        ),

        candidate_name=np.asarray(
            selected_candidate[
                "name"
            ]
        ),

        selected_epoch=np.asarray(
            best_plastic[
                "epoch"
            ],
            dtype=np.int32,
        ),

        learning_rate=np.asarray(
            selected_candidate[
                "learning_rate"
            ],
            dtype=np.float32,
        ),

        incorrect_reward=np.asarray(
            selected_candidate[
                "incorrect_reward"
            ],
            dtype=np.float32,
        ),

        anti_causal_strength=np.asarray(
            selected_candidate[
                "anti_causal_strength"
            ],
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

        plastic_beats_baseline=np.asarray(
            plastic_beats_baseline
        ),
    )

    # ========================================================
    # JSON-safe copies
    # ========================================================

    def strip_weights(
        row,
    ):

        return {
            key: value

            for key, value
            in row.items()

            if key
            !=
            "_weights"
        }

    stage1_json = [
        strip_weights(
            row
        )

        for row
        in stage1_results
    ]

    selected_history_json = [
        strip_weights(
            row
        )

        for row
        in selected_history
    ]

    # ========================================================
    # Freeze Phase-2 development learning protocol
    # ========================================================

    frozen_protocol = {
        "phase": 2,

        "step": "79C",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "PHASE2_DEVELOPMENT_PROTOCOL_SELECTED"
        ),

        "selection_data": (
            "generator_validation only"
        ),

        "old_final_holdout_used": (
            False
        ),

        "selected_candidate": (
            selected_candidate
        ),

        "selected_epoch": int(
            best_plastic[
                "epoch"
            ]
        ),

        "plastic_beats_no_learning_baseline": (
            plastic_beats_baseline
        ),

        "baseline_validation": (
            baseline_validation
        ),

        "selected_validation": (
            best_plastic[
                "validation"
            ]
        ),

        "fixed_settings": {
            "correct_reward": (
                CORRECT_REWARD
            ),

            "neuron_trace_decay": (
                NEURON_TRACE_DECAY
            ),

            "eligibility_decay": (
                ELIGIBILITY_DECAY
            ),

            "minimum_relative_magnitude": (
                MIN_RELATIVE_MAGNITUDE
            ),

            "maximum_relative_magnitude": (
                MAX_RELATIVE_MAGNITUDE
            ),

            "training_order_seed": (
                TRAINING_ORDER_SEED
            ),

            "train_neural_seeds": (
                TRAIN_NEURAL_SEEDS
            ),

            "validation_neural_seeds": (
                VALIDATION_NEURAL_SEEDS
            ),
        },

        "artifacts": {
            "best_weights_file": (
                normalized_relative(
                    BEST_WEIGHTS_FILE
                )
            ),

            "best_weights_sha256": (
                sha256_file(
                    BEST_WEIGHTS_FILE
                )
            ),

            "plastic_subset_sha256": (
                sha256_file(
                    PLASTIC_SUBSET_FILE
                )
            ),

            "mapping_sha256": (
                sha256_file(
                    MAPPING_FILE
                )
            ),

            "fixed_decoder_sha256": (
                sha256_file(
                    FIXED_DECODER_FILE
                )
            ),
        },

        "next_rule": (
            "After development protocol is frozen, "
            "create a NEW Phase-2 holdout before "
            "claiming generalization."
        ),
    }

    FROZEN_PROTOCOL_FILE.write_text(
        json.dumps(
            frozen_protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Save tuning report
    # ========================================================

    output = {
        "phase": 2,

        "step": "79C",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "DEVELOPMENT_PLASTICITY_TUNING_COMPLETE"
        ),

        "old_final_holdout_used": (
            False
        ),

        "baseline_validation": (
            baseline_validation
        ),

        "stage1_candidates": (
            stage1_json
        ),

        "stage1_winner": {
            "candidate_index": (
                selected_index
            ),

            "candidate": (
                selected_candidate
            ),

            "validation": (
                winner_stage1[
                    "validation"
                ]
            ),
        },

        "selected_candidate_epoch_history": (
            selected_history_json
        ),

        "best_plastic_checkpoint": {
            "candidate": (
                selected_candidate
            ),

            "epoch": int(
                best_plastic[
                    "epoch"
                ]
            ),

            "validation": (
                best_plastic[
                    "validation"
                ]
            ),

            "weight_statistics": (
                best_plastic[
                    "weight_statistics"
                ]
            ),

            "beats_no_learning_baseline": (
                plastic_beats_baseline
            ),
        },

        "artifacts": {
            "best_weights_file": (
                normalized_relative(
                    BEST_WEIGHTS_FILE
                )
            ),

            "frozen_protocol_file": (
                normalized_relative(
                    FROZEN_PROTOCOL_FILE
                )
            ),
        },
    }

    OUTPUT_METRICS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final report
    # ========================================================

    best_validation = (
        best_plastic[
            "validation"
        ]
    )

    print()

    print(
        "===================================================="
    )

    print(
        "          STEP 79C DEVELOPMENT RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "No-learning baseline:"
    )

    print(
        f"  Top-1:              "
        f"{baseline_validation['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{baseline_validation['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{baseline_validation['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Best plastic development checkpoint:"
    )

    print(
        f"  Candidate:          "
        f"{selected_candidate['name']}"
    )

    print(
        f"  Epoch:              "
        f"{best_plastic['epoch']}"
    )

    print(
        f"  Top-1:              "
        f"{best_validation['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{best_validation['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{best_validation['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Plastic minus no-learning baseline:"
    )

    print(
        f"  Top-1:              "
        f"{(best_validation['top1_accuracy'] - baseline_validation['top1_accuracy']) * 100:+.2f} pp"
    )

    print(
        f"  Macro F1:           "
        f"{(best_validation['macro_f1'] - baseline_validation['macro_f1']) * 100:+.2f} pp"
    )

    print(
        f"  Semantic accuracy:  "
        f"{(best_validation['semantic_accuracy'] - baseline_validation['semantic_accuracy']) * 100:+.2f} pp"
    )

    print()

    print(
        f"Plasticity beats baseline by selection rule: "
        f"{plastic_beats_baseline}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{normalized_relative(OUTPUT_METRICS_FILE)}"
    )

    print(
        "  "
        f"{normalized_relative(BEST_WEIGHTS_FILE)}"
    )

    print(
        "  "
        f"{normalized_relative(FROZEN_PROTOCOL_FILE)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 79C COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    if plastic_beats_baseline:

        print(
            "A development plasticity rule has beaten "
            "the no-learning reservoir."
        )

        print()

        print(
            "Next: retrain this frozen rule from original "
            "weights on all Phase-2 development prompts, "
            "then create a NEW untouched Phase-2 holdout."
        )

    else:

        print(
            "None of the conservative plastic checkpoints "
            "beat the no-learning baseline."
        )

        print()

        print(
            "Next: change the LEARNING MECHANISM on "
            "development data; do not use any final holdout."
        )


if __name__ == "__main__":
    main()
