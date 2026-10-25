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
# Step 79D — Supervised three-factor recurrent plasticity
#
# Step 79B/79C used a scalar global reward.
# That changed real synapses, but did not improve the semantic
# validation metric.
#
# Step 79D gives the plastic synapses a more specific learning
# signal:
#
#     recent presynaptic activity
#              ×
#     decoder-derived postsynaptic error
#              ×
#     learning rate
#              ↓
#     recurrent MaleCNS synaptic update
#
# The postsynaptic error is computed by analytically
# backpropagating the fixed decoder's cross-entropy gradient
# through:
#
#     Logistic Regression
#             ↓
#     PCA
#             ↓
#     StandardScaler
#             ↓
#     raw 7-block MaleCNS feature
#
# We then collapse the seven feature-block gradients into one
# error value per indirect visual_projection readout neuron.
#
# IMPORTANT:
#   * This is an engineering supervised three-factor rule.
#   * It changes REAL recurrent MaleCNS weights.
#   * Original disk connectome files are never modified.
#   * generator_train drives weight updates.
#   * generator_validation is development monitoring only.
#   * The old final holdout is NOT opened.
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
# Frozen PyFly / Phase-2 artifacts
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

PLASTIC_SUBSET_FILE = (
    PLASTICITY_DIR
    / "plastic_synapse_subset.npz"
)

SMOKE_TEST_MANIFEST_FILE = (
    PLASTICITY_DIR
    / "plasticity_smoke_test.json"
)

FIXED_DECODER_FILE = (
    PLASTICITY_DIR
    / "phase2_train_only_fixed_decoder.joblib"
)

STEP79C_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_plasticity_tuning.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_supervised_three_factor_training.json"
)

BEST_WEIGHTS_FILE = (
    PLASTICITY_DIR
    / "phase2_supervised_best_weights.npz"
)

FROZEN_PROTOCOL_FILE = (
    PLASTICITY_DIR
    / "phase2_supervised_frozen_protocol.json"
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
EXPECTED_READOUT_NEURONS = 5_873
EXPECTED_DIRECT_INPUT_NEURONS = 3_328

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# PRECOMMITTED Step-79D learning settings
#
# This is a NEW learning mechanism, not a continuation of the
# failed reward-STDP rule.
# ============================================================

TRAINING_EPOCHS = 3

LEARNING_RATE = 0.003

# Recent presynaptic spikes define synaptic credit.
PRE_TRACE_DECAY = 0.90

# Keep synapses bounded relative to original MaleCNS values.
MIN_RELATIVE_MAGNITUDE = 0.50
MAX_RELATIVE_MAGNITUDE = 1.50

# The decoder uses seven readout blocks. These fixed weights
# collapse the raw decoder gradient to one error value per
# readout neuron.
#
# We emphasize final/delayed states slightly because they are
# closest to the recurrent state that drives the next token.
BLOCK_ERROR_WEIGHTS = np.asarray(
    [
        0.50,  # prompt_mean
        0.50,  # prompt_max
        1.00,  # prompt_final
        0.75,  # feedback_mean
        0.75,  # feedback_max
        1.00,  # feedback_final
        1.00,  # delayed_state
    ],
    dtype=np.float32,
)

TRAINING_ORDER_SEED = 79_300

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

EPSILON = 1e-8


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


def softmax(
    logits: np.ndarray,
) -> np.ndarray:

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    shifted = (
        logits
        -
        np.max(
            logits
        )
    )

    exp_values = np.exp(
        shifted
    )

    return (
        exp_values
        /
        np.sum(
            exp_values
        )
    )


# ============================================================
# Frozen mapping
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
            "Prompt map width changed."
        )

    if (
        dual_map.feedback_neurons_per_token
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):

        raise ValueError(
            "Feedback map width changed."
        )

    if (
        dual_map.max_prompt_positions
        !=
        MAX_PROMPT_POSITIONS
    ):

        raise ValueError(
            "Prompt position limit changed."
        )

    if (
        dual_map.seed
        !=
        MAP_SEED
    ):

        raise ValueError(
            "Frozen mapping seed changed."
        )

    return dual_map


# ============================================================
# Group teacher-forcing rows
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

    return grouped


# ============================================================
# Decoder gradient helper
#
# Return:
#   predicted token
#   target probability
#   cross-entropy
#   desired raw-feature direction = -dLoss/dFeature
# ============================================================

class FixedDecoderGradient:

    def __init__(
        self,
        pipeline,
    ) -> None:

        required = {
            "scale",
            "pca",
            "classifier",
        }

        if not required.issubset(
            set(
                pipeline.named_steps
            )
        ):

            raise ValueError(
                "Fixed decoder pipeline is missing "
                "scale / pca / classifier."
            )

        self.pipeline = pipeline

        self.scaler = (
            pipeline.named_steps[
                "scale"
            ]
        )

        self.pca = (
            pipeline.named_steps[
                "pca"
            ]
        )

        self.classifier = (
            pipeline.named_steps[
                "classifier"
            ]
        )

        self.classes = np.asarray(
            self.classifier.classes_
        ).astype(str)

        self.class_to_index = {
            token: index

            for index, token
            in enumerate(
                self.classes
            )
        }

        if (
            self.classifier.coef_.shape[
                0
            ]
            !=
            len(
                self.classes
            )
        ):

            raise ValueError(
                "Expected multinomial logistic "
                "coefficient matrix."
            )

        if (
            self.pca.components_.shape[
                1
            ]
            !=
            EXPECTED_FEATURE_DIMENSIONS
        ):

            raise ValueError(
                "Fixed decoder raw feature dimension changed."
            )

    def analyze(
        self,
        feature: np.ndarray,
        target_token: str,
    ):

        if target_token not in self.class_to_index:

            raise KeyError(
                f"Target token not in decoder classes: "
                f"{target_token}"
            )

        X = feature.reshape(
            1,
            -1,
        )

        scaled = self.scaler.transform(
            X
        )

        pca_feature = self.pca.transform(
            scaled
        )[
            0
        ]

        logits = (
            self.classifier.coef_
            @
            pca_feature
            +
            self.classifier.intercept_
        )

        probabilities = softmax(
            logits
        )

        target_index = (
            self.class_to_index[
                target_token
            ]
        )

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        predicted_token = str(
            self.classes[
                predicted_index
            ]
        )

        target_probability = float(
            probabilities[
                target_index
            ]
        )

        cross_entropy = float(
            -np.log(
                max(
                    target_probability,
                    1e-12,
                )
            )
        )

        # ----------------------------------------------------
        # Cross-entropy gradient w.r.t. logits
        # ----------------------------------------------------

        grad_logits = (
            probabilities.copy()
        )

        grad_logits[
            target_index
        ] -= 1.0

        # ----------------------------------------------------
        # dLoss / dPCA
        # ----------------------------------------------------

        grad_pca = (
            self.classifier.coef_.T
            @
            grad_logits
        )

        # ----------------------------------------------------
        # dLoss / dScaledRawFeature
        #
        # PCA transform derivative is components_.T.
        # PCA centering affects the forward value but not the
        # derivative.
        # ----------------------------------------------------

        grad_scaled = (
            self.pca.components_.T
            @
            grad_pca
        )

        # ----------------------------------------------------
        # StandardScaler derivative:
        #
        #   scaled = (raw - mean) / scale
        # ----------------------------------------------------

        grad_raw = (
            grad_scaled
            /
            self.scaler.scale_
        )

        # Desired feature direction that REDUCES loss.
        desired_raw = (
            -grad_raw
        ).astype(
            np.float32,
            copy=False,
        )

        return {
            "prediction": (
                predicted_token
            ),

            "correct": bool(
                predicted_token
                ==
                target_token
            ),

            "target_probability": (
                target_probability
            ),

            "cross_entropy": (
                cross_entropy
            ),

            "desired_raw": (
                desired_raw
            ),
        }


# ============================================================
# Three-factor recurrent plasticity engine
# ============================================================

class ThreeFactorPlasticity:

    def __init__(
        self,
        brain: FlyBrain,
        edge_offsets: np.ndarray,
        pre_neurons: np.ndarray,
        post_neurons: np.ndarray,
        post_readout_indices: np.ndarray,
        original_weights: np.ndarray,
    ) -> None:

        if brain.device != "cuda":

            raise RuntimeError(
                "Step 79D requires CUDA."
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

        self.post_readout_indices = cp.asarray(
            post_readout_indices,
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

        self.pre_trace = cp.zeros(
            brain.n,
            dtype=cp.float32,
        )

    def reset_episode(
        self,
    ) -> None:

        self.pre_trace.fill(
            0
        )

    def observe_spikes(
        self,
        fired: np.ndarray,
    ) -> None:

        cp = self.cp

        self.pre_trace *= np.float32(
            PRE_TRACE_DECAY
        )

        if len(
            fired
        ):

            fired_gpu = cp.asarray(
                fired,
                dtype=cp.int64,
            )

            self.pre_trace[
                fired_gpu
            ] += np.float32(
                1.0
            )

    def apply_supervised_error(
        self,
        readout_error_cpu: np.ndarray,
    ) -> dict:

        cp = self.cp

        if (
            readout_error_cpu.shape
            !=
            (
                EXPECTED_READOUT_NEURONS,
            )
        ):

            raise ValueError(
                "Readout error vector has wrong shape."
            )

        readout_error = cp.asarray(
            readout_error_cpu,
            dtype=cp.float32,
        )

        # ----------------------------------------------------
        # Presynaptic credit
        # ----------------------------------------------------

        pre_credit = self.pre_trace[
            self.pre_neurons
        ]

        pre_max = float(
            cp.max(
                pre_credit
            ).get()
        )

        if (
            not np.isfinite(
                pre_max
            )
            or
            pre_max
            <=
            EPSILON
        ):

            return {
                "changed_synapses": 0,
                "mean_abs_delta": 0.0,
                "max_abs_credit": 0.0,
            }

        pre_credit = (
            pre_credit
            /
            np.float32(
                pre_max
            )
        )

        post_error = readout_error[
            self.post_readout_indices
        ]

        # ----------------------------------------------------
        # Three-factor credit:
        #
        #   presynaptic recent activity
        #        ×
        #   desired postsynaptic change
        #
        # original_sign converts desired signed-output change
        # into a change in SYNAPTIC MAGNITUDE:
        #
        # positive error:
        #   excitatory -> strengthen
        #   inhibitory -> weaken
        #
        # negative error:
        #   excitatory -> weaken
        #   inhibitory -> strengthen
        # ----------------------------------------------------

        credit = (
            pre_credit
            *
            post_error
            *
            self.original_sign
        )

        max_abs_credit = float(
            cp.max(
                cp.abs(
                    credit
                )
            ).get()
        )

        if (
            not np.isfinite(
                max_abs_credit
            )
            or
            max_abs_credit
            <=
            EPSILON
        ):

            return {
                "changed_synapses": 0,
                "mean_abs_delta": 0.0,
                "max_abs_credit": (
                    max_abs_credit
                ),
            }

        # Readout error is already normalized to [-1, 1].
        # Clipping here prevents a single edge from receiving
        # more than one full learning-rate unit.
        credit = cp.clip(
            credit,
            -1.0,
            +1.0,
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
            )
            *
            credit
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

        return {
            "changed_synapses": (
                changed_synapses
            ),

            "mean_abs_delta": (
                mean_abs_delta
            ),

            "max_abs_credit": (
                max_abs_credit
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
        }


# ============================================================
# Main experiment
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
        "       STEP 79D - SUPERVISED THREE-FACTOR"
    )

    print(
        "===================================================="
    )

    print()

    required = [
        TRAIN_GENERATOR_FILE,
        VALIDATION_GENERATOR_FILE,
        MAPPING_FILE,
        PLASTIC_SUBSET_FILE,
        SMOKE_TEST_MANIFEST_FILE,
        FIXED_DECODER_FILE,
        STEP79C_METRICS_FILE,
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

    print(
        "Step 79A infrastructure:        PASS"
    )

    print(
        "Step 79C reward-STDP tuning:    COMPLETE / NO WIN"
    )

    print()

    print(
        "Old final holdout used:         NO"
    )

    print(
        f"Learning mechanism:             "
        f"decoder-error three-factor rule"
    )

    print(
        f"Learning rate:                  "
        f"{LEARNING_RATE}"
    )

    print(
        f"Training epochs:                "
        f"{TRAINING_EPOCHS}"
    )

    print(
        f"Pre-trace decay:                "
        f"{PRE_TRACE_DECAY}"
    )

    print(
        f"Weight magnitude bounds:        "
        f"{MIN_RELATIVE_MAGNITUDE:.2f}x .. "
        f"{MAX_RELATIVE_MAGNITUDE:.2f}x"
    )

    # ========================================================
    # Development data
    # ========================================================

    train_samples = load_generator_jsonl(
        TRAIN_GENERATOR_FILE
    )

    validation_samples = load_generator_jsonl(
        VALIDATION_GENERATOR_FILE
    )

    train_grouped = group_by_example(
        train_samples
    )

    validation_grouped = group_by_example(
        validation_samples
    )

    if len(
        train_grouped
    ) != 36:

        raise ValueError(
            "Expected 36 training prompts."
        )

    if len(
        validation_grouped
    ) != 6:

        raise ValueError(
            "Expected 6 validation prompts."
        )

    # ========================================================
    # Fixed train-only decoder and analytic gradient
    # ========================================================

    decoder = joblib.load(
        FIXED_DECODER_FILE
    )

    decoder_gradient = FixedDecoderGradient(
        decoder
    )

    print()

    print(
        "Fixed decoder:                  "
        "Balanced Logistic PCA=100 C=0.1"
    )

    print(
        "Decoder updated during training:NO"
    )

    # ========================================================
    # Mapping + Step-79A real synapses
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

    if brain.n != EXPECTED_NEURONS:

        raise ValueError(
            "Unexpected MaleCNS neuron count."
        )

    if brain._W.nnz != EXPECTED_CONNECTIONS:

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

    if len(
        input_neurons
    ) != EXPECTED_DIRECT_INPUT_NEURONS:

        raise ValueError(
            "Direct input population changed."
        )

    if len(
        readout_population
    ) != EXPECTED_READOUT_NEURONS:

        raise ValueError(
            "Readout population changed."
        )

    # ========================================================
    # Verify selected synapse coordinates
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
            "Selected presynaptic coordinates changed."
        )

    if not np.array_equal(
        current_post,
        post_neurons,
    ):

        raise ValueError(
            "Selected postsynaptic coordinates changed."
        )

    import cupy as cp

    gpu_offsets = cp.asarray(
        edge_offsets,
        dtype=cp.int64,
    )

    fresh_weights = brain._W.data[
        gpu_offsets
    ].get()

    if not np.array_equal(
        fresh_weights,
        original_weights,
    ):

        raise ValueError(
            "Fresh MaleCNS plastic weights "
            "do not match Step 79A baseline."
        )

    # ========================================================
    # Map global postsynaptic neuron IDs to readout positions
    # ========================================================

    readout_lookup = np.full(
        brain.n,
        -1,
        dtype=np.int64,
    )

    readout_lookup[
        readout_population
    ] = np.arange(
        len(
            readout_population
        ),
        dtype=np.int64,
    )

    post_readout_indices = readout_lookup[
        post_neurons
    ]

    if np.any(
        post_readout_indices
        <
        0
    ):

        raise ValueError(
            "Plastic subset contains postsynaptic "
            "neurons outside the indirect readout."
        )

    print(
        f"Plastic recurrent synapses:    "
        f"{len(edge_offsets):,}"
    )

    # ========================================================
    # Trace + plasticity
    # ========================================================

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    plasticity = ThreeFactorPlasticity(
        brain=brain,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_neurons=post_neurons,
        post_readout_indices=(
            post_readout_indices
        ),
        original_weights=(
            original_weights
        ),
    )

    # ========================================================
    # Convert raw decoder gradient to readout-neuron error
    # ========================================================

    def raw_gradient_to_readout_error(
        desired_raw: np.ndarray,
    ) -> np.ndarray:

        blocks = desired_raw.reshape(
            len(
                FEATURE_BLOCK_NAMES
            ),
            EXPECTED_READOUT_NEURONS,
        )

        error = np.sum(
            blocks
            *
            BLOCK_ERROR_WEIGHTS[
                :,
                None,
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        max_abs = float(
            np.max(
                np.abs(
                    error
                )
            )
        )

        if (
            not np.isfinite(
                max_abs
            )
            or
            max_abs
            <=
            EPSILON
        ):

            return np.zeros(
                EXPECTED_READOUT_NEURONS,
                dtype=np.float32,
            )

        return (
            error
            /
            np.float32(
                max_abs
            )
        ).astype(
            np.float32,
            copy=False,
        )

    # ========================================================
    # Brain step + stimulation
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

        return state

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

                states.append(
                    brain_step(
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

            if feature.shape != (
                EXPECTED_FEATURE_DIMENSIONS,
            ):

                raise RuntimeError(
                    "Unexpected feature shape."
                )

            analysis = decoder_gradient.analyze(
                feature,
                sample.target_token,
            )

            update = None

            if learn:

                readout_error = (
                    raw_gradient_to_readout_error(
                        analysis[
                            "desired_raw"
                        ]
                    )
                )

                update = (
                    plasticity.apply_supervised_error(
                        readout_error
                    )
                )

            rows.append(
                {
                    "target": (
                        sample.target_token
                    ),

                    "prediction": (
                        analysis[
                            "prediction"
                        ]
                    ),

                    "correct": (
                        analysis[
                            "correct"
                        ]
                    ),

                    "position": int(
                        sample.target_position
                    ),

                    "target_probability": (
                        analysis[
                            "target_probability"
                        ]
                    ),

                    "cross_entropy": (
                        analysis[
                            "cross_entropy"
                        ]
                    ),

                    "update": (
                        update
                    ),
                }
            )

        return rows

    # ========================================================
    # Validation
    # ========================================================

    def evaluate_validation():

        targets = []
        predictions = []
        positions = []
        losses = []

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

                    losses.append(
                        row[
                            "cross_entropy"
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

            "mean_cross_entropy": float(
                np.mean(
                    losses
                )
            ),

            "accuracy_by_position": (
                position_accuracy
            ),
        }

    # ========================================================
    # Save checkpoint
    # ========================================================

    def save_checkpoint(
        epoch: int,
    ) -> Path:

        path = (
            PLASTICITY_DIR
            /
            f"phase2_supervised_epoch_{epoch:02}_weights.npz"
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
                plasticity.current_weights_cpu()
            ),

            epoch=np.asarray(
                epoch,
                dtype=np.int32,
            ),

            learning_rate=np.asarray(
                LEARNING_RATE,
                dtype=np.float32,
            ),

            pre_trace_decay=np.asarray(
                PRE_TRACE_DECAY,
                dtype=np.float32,
            ),

            block_error_weights=(
                BLOCK_ERROR_WEIGHTS
            ),
        )

        return path

    # ========================================================
    # Baseline
    # ========================================================

    baseline_validation = (
        evaluate_validation()
    )

    baseline_checkpoint = (
        save_checkpoint(
            0
        )
    )

    history = [
        {
            "epoch": 0,

            "training": None,

            "validation": (
                baseline_validation
            ),

            "weight_statistics": (
                plasticity.statistics()
            ),

            "checkpoint": (
                normalized_relative(
                    baseline_checkpoint
                )
            ),
        }
    ]

    print()

    print(
        "===================================================="
    )

    print(
        "      VALIDATION BEFORE THREE-FACTOR LEARNING"
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

    print(
        f"Mean cross-entropy: "
        f"{baseline_validation['mean_cross_entropy']:.6f}"
    )

    # ========================================================
    # Train
    # ========================================================

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
            f"        THREE-FACTOR TRAINING EPOCH "
            f"{epoch}/{TRAINING_EPOCHS}"
        )

        print(
            "===================================================="
        )

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

        train_targets = []
        train_predictions = []

        losses = []

        update_events = 0
        changed_synapse_events = 0
        mean_abs_delta_sum = 0.0

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
                        epoch
                        +
                        episode_number
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
                True,
            )

            correct_count = 0

            for row in rows:

                train_targets.append(
                    row[
                        "target"
                    ]
                )

                train_predictions.append(
                    row[
                        "prediction"
                    ]
                )

                losses.append(
                    row[
                        "cross_entropy"
                    ]
                )

                correct_count += int(
                    row[
                        "correct"
                    ]
                )

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
                f"seed={neural_seed} "
                f"token_acc="
                f"{correct_count / len(rows) * 100:6.2f}%"
            )

        train_accuracy = float(
            accuracy_score(
                train_targets,
                train_predictions,
            )
        )

        train_macro_f1 = float(
            f1_score(
                train_targets,
                train_predictions,
                average="macro",
                zero_division=0,
            )
        )

        train_loss = float(
            np.mean(
                losses
            )
        )

        validation = (
            evaluate_validation()
        )

        weight_stats = (
            plasticity.statistics()
        )

        checkpoint_path = (
            save_checkpoint(
                epoch
            )
        )

        training_metrics = {
            "top1_accuracy_before_online_update": (
                train_accuracy
            ),

            "macro_f1_before_online_update": (
                train_macro_f1
            ),

            "mean_cross_entropy_before_online_update": (
                train_loss
            ),

            "update_events": int(
                update_events
            ),

            "sum_changed_synapse_events": int(
                changed_synapse_events
            ),

            "mean_abs_delta_per_update": float(
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
                    training_metrics
                ),

                "validation": (
                    validation
                ),

                "weight_statistics": (
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
            f"Epoch {epoch} training Top-1:         "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Epoch {epoch} training loss:          "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation Top-1:                    "
            f"{validation['top1_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Macro F1:                 "
            f"{validation['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Validation semantic accuracy:        "
            f"{validation['semantic_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation cross-entropy:            "
            f"{validation['mean_cross_entropy']:.6f}"
        )

        print(
            f"Synapses changed from original:      "
            f"{weight_stats['changed_from_original']:,}"
        )

        print(
            f"Mean magnitude ratio:                "
            f"{weight_stats['mean_relative_magnitude']:.6f}x"
        )

    # ========================================================
    # Select best DEVELOPMENT checkpoint
    #
    # Primary: semantic accuracy
    # Secondary: macro F1
    # Tertiary: Top-1
    # Final tie-break: lower cross entropy
    # ========================================================

    def selection_key(
        row,
    ):

        validation = row[
            "validation"
        ]

        return (
            validation[
                "semantic_accuracy"
            ],

            validation[
                "macro_f1"
            ],

            validation[
                "top1_accuracy"
            ],

            -validation[
                "mean_cross_entropy"
            ],
        )

    best_row = max(
        history,
        key=selection_key,
    )

    best_epoch = int(
        best_row[
            "epoch"
        ]
    )

    # Load exact best checkpoint weights.
    best_checkpoint_path = (
        PROJECT_ROOT
        /
        best_row[
            "checkpoint"
        ]
    )

    best_checkpoint = np.load(
        best_checkpoint_path,
        allow_pickle=False,
    )

    best_weights = best_checkpoint[
        "plastic_weights"
    ].astype(
        np.float32
    )

    plastic_beats_baseline = bool(
        best_epoch
        >
        0
        and
        selection_key(
            best_row
        )
        >
        selection_key(
            history[
                0
            ]
        )
    )

    # ========================================================
    # Save canonical best weights
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
            best_weights
        ),

        selected_epoch=np.asarray(
            best_epoch,
            dtype=np.int32,
        ),

        learning_rate=np.asarray(
            LEARNING_RATE,
            dtype=np.float32,
        ),

        pre_trace_decay=np.asarray(
            PRE_TRACE_DECAY,
            dtype=np.float32,
        ),

        block_error_weights=(
            BLOCK_ERROR_WEIGHTS
        ),

        plastic_beats_baseline=np.asarray(
            plastic_beats_baseline
        ),
    )

    # ========================================================
    # Freeze development protocol
    # ========================================================

    frozen_protocol = {
        "phase": 2,

        "step": "79D",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "SUPERVISED_THREE_FACTOR_DEVELOPMENT_COMPLETE"
        ),

        "learning_mechanism": (
            "decoder-gradient supervised three-factor "
            "recurrent plasticity"
        ),

        "old_final_holdout_used": (
            False
        ),

        "training_data": (
            "generator_train"
        ),

        "selection_data": (
            "generator_validation"
        ),

        "selected_epoch": (
            best_epoch
        ),

        "plastic_beats_no_learning_baseline": (
            plastic_beats_baseline
        ),

        "settings": {
            "learning_rate": (
                LEARNING_RATE
            ),

            "pre_trace_decay": (
                PRE_TRACE_DECAY
            ),

            "minimum_relative_magnitude": (
                MIN_RELATIVE_MAGNITUDE
            ),

            "maximum_relative_magnitude": (
                MAX_RELATIVE_MAGNITUDE
            ),

            "block_error_weights": (
                BLOCK_ERROR_WEIGHTS.tolist()
            ),

            "training_epochs": (
                TRAINING_EPOCHS
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

        "baseline_validation": (
            history[
                0
            ][
                "validation"
            ]
        ),

        "best_validation": (
            best_row[
                "validation"
            ]
        ),

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

            "fixed_decoder_sha256": (
                sha256_file(
                    FIXED_DECODER_FILE
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
        },

        "next_rule": (
            "If plasticity beats baseline, freeze this "
            "learning mechanism and create a new untouched "
            "Phase-2 benchmark before any final claim."
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
    # Save training report
    # ========================================================

    output = {
        "phase": 2,

        "step": "79D",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "SUPERVISED_THREE_FACTOR_TRAINING_COMPLETE"
        ),

        "old_final_holdout_used": (
            False
        ),

        "history": (
            history
        ),

        "best_epoch": (
            best_epoch
        ),

        "plastic_beats_no_learning_baseline": (
            plastic_beats_baseline
        ),

        "baseline_validation": (
            history[
                0
            ][
                "validation"
            ]
        ),

        "best_validation": (
            best_row[
                "validation"
            ]
        ),

        "best_weight_statistics": (
            best_row[
                "weight_statistics"
            ]
        ),

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

        "rules": [
            (
                "Only generator_train drove recurrent "
                "synaptic updates."
            ),

            (
                "generator_validation was used only for "
                "development checkpoint selection."
            ),

            (
                "The fixed decoder was not refit during "
                "synaptic learning."
            ),

            (
                "The old final holdout was never opened."
            ),

            (
                "This is an engineering supervised "
                "three-factor rule, not a claim of exact "
                "biological fly learning."
            ),
        ],
    }

    OUTPUT_METRICS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Report
    # ========================================================

    baseline = history[
        0
    ][
        "validation"
    ]

    best = best_row[
        "validation"
    ]

    print()

    print(
        "===================================================="
    )

    print(
        "          STEP 79D LEARNING RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "No-learning validation:"
    )

    print(
        f"  Top-1:              "
        f"{baseline['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{baseline['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{baseline['semantic_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Cross-entropy:      "
        f"{baseline['mean_cross_entropy']:.6f}"
    )

    print()

    print(
        f"Best learned checkpoint (epoch {best_epoch}):"
    )

    print(
        f"  Top-1:              "
        f"{best['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{best['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{best['semantic_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Cross-entropy:      "
        f"{best['mean_cross_entropy']:.6f}"
    )

    print()

    print(
        "Learned minus no-learning baseline:"
    )

    print(
        f"  Top-1:              "
        f"{(best['top1_accuracy'] - baseline['top1_accuracy']) * 100:+.2f} pp"
    )

    print(
        f"  Macro F1:           "
        f"{(best['macro_f1'] - baseline['macro_f1']) * 100:+.2f} pp"
    )

    print(
        f"  Semantic accuracy:  "
        f"{(best['semantic_accuracy'] - baseline['semantic_accuracy']) * 100:+.2f} pp"
    )

    print(
        f"  Cross-entropy:      "
        f"{best['mean_cross_entropy'] - baseline['mean_cross_entropy']:+.6f}"
    )

    print()

    print(
        f"Plastic checkpoint beats baseline: "
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
        "                 STEP 79D COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    if plastic_beats_baseline:

        print(
            "The recurrent MaleCNS synapses have now "
            "improved the development validation task."
        )

        print()

        print(
            "Next: freeze this mechanism, retrain cleanly "
            "from the original connectome, and create a NEW "
            "untouched Phase-2 holdout."
        )

    else:

        print(
            "This supervised recurrent plasticity rule "
            "still did not beat the no-learning baseline."
        )

        print()

        print(
            "Next: expand which recurrent synapses are "
            "plastic / move plasticity deeper into the "
            "MaleCNS pathway, still using development only."
        )


if __name__ == "__main__":
    main()
