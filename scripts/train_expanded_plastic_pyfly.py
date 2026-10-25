from __future__ import annotations

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
# Step 79E — Expanded supervised recurrent plasticity
#
# Why this exists:
#   Step 79D's best checkpoint was epoch 0. Its 100k
#   visual->readout plastic subset was too restrictive.
#
# New idea:
#   Make a much larger set of REAL recurrent MaleCNS
#   synapses plastic:
#
#       ANY presynaptic MaleCNS neuron
#                 ↓
#       indirect visual_projection readout neuron
#
#   up to 500,000 recurrent synapses.
#
# Learning signal:
#   fixed decoder cross-entropy gradient
#       -> raw 7-block feature gradient
#       -> per-readout-neuron error
#       × recent presynaptic activity
#       -> recurrent weight update
#
# IMPORTANT:
#   * The original disk connectome is never modified.
#   * generator_train drives updates.
#   * generator_validation is development selection only.
#   * The old final holdout is NEVER opened here.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda"

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_train.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_validation.jsonl"
)

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

FIXED_DECODER_FILE = (
    PLASTICITY_DIR
    / "phase2_train_only_fixed_decoder.joblib"
)

STEP79D_FILE = (
    PLASTICITY_DIR
    / "phase2_supervised_three_factor_training.json"
)

EXPANDED_SUBSET_FILE = (
    PLASTICITY_DIR
    / "phase2_expanded_plastic_synapse_subset.npz"
)

OUTPUT_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_expanded_supervised_training.json"
)

BEST_WEIGHTS_FILE = (
    PLASTICITY_DIR
    / "phase2_expanded_best_weights.npz"
)

FROZEN_PROTOCOL_FILE = (
    PLASTICITY_DIR
    / "phase2_expanded_frozen_protocol.json"
)


# ============================================================
# Frozen Step-62 stimulation
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
# Step-79E precommitted settings
# ============================================================

EXPANDED_SUBSET_SEED = 79_500
MAX_PLASTIC_SYNAPSES = 500_000

TRAINING_EPOCHS = 3
LEARNING_RATE = 0.001

PRE_TRACE_DECAY = 0.90

MIN_RELATIVE_MAGNITUDE = 0.50
MAX_RELATIVE_MAGNITUDE = 1.50

# Collapse the 7 raw decoder-gradient blocks into one desired
# error value per readout neuron.
BLOCK_ERROR_WEIGHTS = np.asarray(
    [
        0.25,  # prompt_mean
        0.25,  # prompt_max
        0.75,  # prompt_final
        0.50,  # feedback_mean
        0.50,  # feedback_max
        1.00,  # feedback_final
        1.00,  # delayed_state
    ],
    dtype=np.float32,
)

TRAINING_ORDER_SEED = 79_510

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

def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def relative(path: Path) -> str:
    return str(
        path.relative_to(PROJECT_ROOT)
    ).replace("\\", "/")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(
        logits,
        dtype=np.float64,
    )

    values = values - np.max(values)

    exp_values = np.exp(values)

    return exp_values / np.sum(exp_values)


def group_by_example(samples):
    grouped = defaultdict(list)

    for sample in samples:
        grouped[
            sample.example_id
        ].append(sample)

    for example_id in grouped:
        grouped[
            example_id
        ].sort(
            key=lambda row:
                row.target_position
        )

    return grouped


# ============================================================
# Frozen PyFly map
# ============================================================

def load_dual_map(
    path: Path,
) -> DualChannelMap:

    data = np.load(
        path,
        allow_pickle=False,
    )

    prompt_mapping = {}

    prompt_tokens = data[
        "prompt_tokens"
    ].astype(str)

    prompt_positions = data[
        "prompt_positions"
    ].astype(np.int64)

    prompt_neurons = data[
        "prompt_neurons"
    ].astype(np.int64)

    for index, (
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
                str(token),
                int(position),
            )
        ] = (
            prompt_neurons[
                index
            ].copy()
        )

    feedback_mapping = {}

    feedback_tokens = data[
        "feedback_tokens"
    ].astype(str)

    feedback_neurons = data[
        "feedback_neurons"
    ].astype(np.int64)

    for index, token in enumerate(
        feedback_tokens
    ):
        feedback_mapping[
            str(token)
        ] = (
            feedback_neurons[
                index
            ].copy()
        )

    result = DualChannelMap(
        prompt_mapping=prompt_mapping,
        feedback_mapping=feedback_mapping,
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

    if result.seed != MAP_SEED:
        raise ValueError(
            "Mapping seed changed."
        )

    return result


# ============================================================
# Select expanded recurrent plastic subset
#
# CUDA CSR orientation used here:
#   row    = postsynaptic neuron
#   column = presynaptic neuron
#
# Eligible:
#   every recurrent edge whose POSTSYNAPTIC neuron belongs to
#   the 5,873 indirect readout neurons.
#
# If >500k eligible, sample a fixed 500k.
# ============================================================

def select_expanded_subset(
    brain: FlyBrain,
    readout_population: np.ndarray,
) -> dict:

    W = brain._W

    indptr = W.indptr.get().astype(
        np.int64,
        copy=False,
    )

    indices = W.indices.get().astype(
        np.int64,
        copy=False,
    )

    weights = W.data.get().astype(
        np.float32,
        copy=False,
    )

    offsets_list = []
    pre_list = []
    post_list = []

    for post in readout_population:

        start = int(
            indptr[
                post
            ]
        )

        end = int(
            indptr[
                post + 1
            ]
        )

        if start == end:
            continue

        offsets = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        offsets_list.append(
            offsets
        )

        pre_list.append(
            indices[
                start:end
            ].astype(
                np.int64,
                copy=False,
            )
        )

        post_list.append(
            np.full(
                end - start,
                int(post),
                dtype=np.int64,
            )
        )

    if not offsets_list:
        raise RuntimeError(
            "No incoming recurrent readout "
            "synapses were found."
        )

    edge_offsets = np.concatenate(
        offsets_list
    )

    pre_neurons = np.concatenate(
        pre_list
    )

    post_neurons = np.concatenate(
        post_list
    )

    eligible_count = int(
        len(edge_offsets)
    )

    rng = np.random.default_rng(
        EXPANDED_SUBSET_SEED
    )

    if (
        eligible_count
        >
        MAX_PLASTIC_SYNAPSES
    ):
        chosen = rng.choice(
            eligible_count,
            size=MAX_PLASTIC_SYNAPSES,
            replace=False,
        )

        chosen.sort()

        edge_offsets = edge_offsets[
            chosen
        ]

        pre_neurons = pre_neurons[
            chosen
        ]

        post_neurons = post_neurons[
            chosen
        ]

    original_weights = weights[
        edge_offsets
    ].copy()

    if np.any(
        original_weights == 0
    ):
        raise RuntimeError(
            "Expanded plastic subset contains "
            "zero-weight edges."
        )

    return {
        "eligible_count": (
            eligible_count
        ),
        "edge_offsets": (
            edge_offsets
        ),
        "pre_neurons": (
            pre_neurons
        ),
        "post_neurons": (
            post_neurons
        ),
        "original_weights": (
            original_weights
        ),
    }


# ============================================================
# Analytic gradient through frozen decoder
# ============================================================

class FixedDecoderGradient:

    def __init__(self, pipeline) -> None:

        self.scaler = pipeline.named_steps[
            "scale"
        ]

        self.pca = pipeline.named_steps[
            "pca"
        ]

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
            in enumerate(self.classes)
        }

        if (
            self.pca.components_.shape[
                1
            ]
            !=
            EXPECTED_FEATURE_DIMENSIONS
        ):
            raise ValueError(
                "Decoder raw feature dimension "
                "does not match PyFly."
            )

    def analyze(
        self,
        feature: np.ndarray,
        target_token: str,
    ) -> dict:

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
            np.argmax(probabilities)
        )

        predicted = str(
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

        grad_logits = probabilities.copy()

        grad_logits[
            target_index
        ] -= 1.0

        grad_pca = (
            self.classifier.coef_.T
            @
            grad_logits
        )

        grad_scaled = (
            self.pca.components_.T
            @
            grad_pca
        )

        grad_raw = (
            grad_scaled
            /
            self.scaler.scale_
        )

        desired_raw = (
            -grad_raw
        ).astype(
            np.float32,
            copy=False,
        )

        return {
            "prediction": (
                predicted
            ),
            "correct": bool(
                predicted
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
# Expanded three-factor recurrent plasticity
# ============================================================

class ExpandedPlasticity:

    def __init__(
        self,
        brain: FlyBrain,
        edge_offsets: np.ndarray,
        pre_neurons: np.ndarray,
        post_readout_indices: np.ndarray,
        original_weights: np.ndarray,
    ) -> None:

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

    def reset_episode(self) -> None:
        self.pre_trace.fill(0)

    def observe_spikes(
        self,
        fired: np.ndarray,
    ) -> None:

        cp = self.cp

        self.pre_trace *= np.float32(
            PRE_TRACE_DECAY
        )

        if len(fired):

            fired_gpu = cp.asarray(
                fired,
                dtype=cp.int64,
            )

            self.pre_trace[
                fired_gpu
            ] += np.float32(1.0)

    def apply_error(
        self,
        readout_error_cpu: np.ndarray,
    ) -> dict:

        cp = self.cp

        readout_error = cp.asarray(
            readout_error_cpu,
            dtype=cp.float32,
        )

        pre_credit = self.pre_trace[
            self.pre_neurons
        ]

        pre_max = float(
            cp.max(
                pre_credit
            ).get()
        )

        if (
            not np.isfinite(pre_max)
            or
            pre_max <= EPSILON
        ):
            return {
                "changed_synapses": 0,
                "mean_abs_delta": 0.0,
            }

        pre_credit = (
            pre_credit
            /
            np.float32(pre_max)
        )

        post_error = readout_error[
            self.post_readout_indices
        ]

        # Convert desired signed postsynaptic change into a
        # desired change of synaptic MAGNITUDE.
        credit = (
            pre_credit
            *
            post_error
            *
            self.original_sign
        )

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

        new_magnitude = (
            current_magnitude
            *
            (
                np.float32(1.0)
                +
                np.float32(
                    LEARNING_RATE
                )
                *
                credit
            )
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
                cp.abs(delta)
                >
                np.float32(1e-10)
            ).get()
        )

        mean_abs_delta = float(
            cp.mean(
                cp.abs(delta)
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
        }

    def current_weights_cpu(
        self,
    ) -> np.ndarray:

        return self.brain._W.data[
            self.edge_offsets
        ].get()

    def statistics(self) -> dict:

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
            cp.abs(current)
            /
            self.original_magnitude
        )

        return {
            "changed_from_original": int(
                cp.count_nonzero(
                    changed
                    >
                    np.float32(1e-8)
                ).get()
            ),
            "mean_abs_change_from_original": float(
                cp.mean(changed).get()
            ),
            "mean_relative_magnitude": float(
                cp.mean(ratio).get()
            ),
            "min_relative_magnitude": float(
                cp.min(ratio).get()
            ),
            "max_relative_magnitude": float(
                cp.max(ratio).get()
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
        "       STEP 79E - EXPANDED PLASTICITY"
    )
    print(
        "===================================================="
    )

    required = [
        TRAIN_FILE,
        VALIDATION_FILE,
        MAPPING_FILE,
        FIXED_DECODER_FILE,
        STEP79D_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    step79d = load_json(
        STEP79D_FILE
    )

    print()

    print(
        "Step 79D result:              "
        f"plastic_beats_baseline="
        f"{step79d.get('plastic_beats_no_learning_baseline')}"
    )

    print(
        "Old final holdout used:       NO"
    )

    print(
        f"Maximum plastic synapses:     "
        f"{MAX_PLASTIC_SYNAPSES:,}"
    )

    print(
        f"Learning rate:                "
        f"{LEARNING_RATE}"
    )

    print(
        f"Training epochs:              "
        f"{TRAINING_EPOCHS}"
    )

    # ========================================================
    # Data
    # ========================================================

    train_samples = load_generator_jsonl(
        TRAIN_FILE
    )

    validation_samples = (
        load_generator_jsonl(
            VALIDATION_FILE
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

    if len(train_grouped) != 36:
        raise ValueError(
            "Expected 36 training prompts."
        )

    if len(validation_grouped) != 6:
        raise ValueError(
            "Expected 6 validation prompts."
        )

    # ========================================================
    # Frozen decoder
    # ========================================================

    decoder = joblib.load(
        FIXED_DECODER_FILE
    )

    decoder_gradient = (
        FixedDecoderGradient(
            decoder
        )
    )

    # ========================================================
    # Mapping + MaleCNS
    # ========================================================

    dual_map = load_dual_map(
        MAPPING_FILE
    )

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
            "Unexpected neuron count."
        )

    if brain._W.nnz != EXPECTED_CONNECTIONS:
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
    # Expanded plastic subset
    # ========================================================

    print()

    print(
        "Selecting expanded incoming "
        "recurrent readout synapses..."
    )

    subset = select_expanded_subset(
        brain,
        readout_population,
    )

    edge_offsets = subset[
        "edge_offsets"
    ]

    pre_neurons = subset[
        "pre_neurons"
    ]

    post_neurons = subset[
        "post_neurons"
    ]

    original_weights = subset[
        "original_weights"
    ]

    print(
        f"Eligible incoming readout edges: "
        f"{subset['eligible_count']:,}"
    )

    print(
        f"Selected plastic synapses:       "
        f"{len(edge_offsets):,}"
    )

    print(
        f"Excitatory selected:             "
        f"{np.sum(original_weights > 0):,}"
    )

    print(
        f"Inhibitory selected:             "
        f"{np.sum(original_weights < 0):,}"
    )

    np.savez(
        EXPANDED_SUBSET_FILE,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_neurons=post_neurons,
        original_weights=original_weights,
        subset_seed=np.asarray(
            EXPANDED_SUBSET_SEED,
            dtype=np.int64,
        ),
    )

    # ========================================================
    # Map postsynaptic global IDs to readout positions
    # ========================================================

    readout_lookup = np.full(
        brain.n,
        -1,
        dtype=np.int64,
    )

    readout_lookup[
        readout_population
    ] = np.arange(
        len(readout_population),
        dtype=np.int64,
    )

    post_readout_indices = (
        readout_lookup[
            post_neurons
        ]
    )

    if np.any(
        post_readout_indices
        <
        0
    ):
        raise ValueError(
            "Expanded subset has a post neuron "
            "outside the readout."
        )

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    plasticity = ExpandedPlasticity(
        brain=brain,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_readout_indices=(
            post_readout_indices
        ),
        original_weights=(
            original_weights
        ),
    )

    # ========================================================
    # Decoder gradient -> one error per readout neuron
    # ========================================================

    def readout_error_from_raw(
        desired_raw: np.ndarray,
    ) -> np.ndarray:

        blocks = desired_raw.reshape(
            len(FEATURE_BLOCK_NAMES),
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
                np.abs(error)
            )
        )

        if (
            not np.isfinite(max_abs)
            or
            max_abs <= EPSILON
        ):
            return np.zeros(
                EXPECTED_READOUT_NEURONS,
                dtype=np.float32,
            )

        return (
            error
            /
            np.float32(max_abs)
        ).astype(
            np.float32,
            copy=False,
        )

    # ========================================================
    # Simulation helpers
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

        for position, token in enumerate(
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

        states = np.stack(states)

        return (
            states.mean(axis=0),
            states.max(axis=0),
            states[-1],
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

        states = np.stack(states)

        feedback_mean = (
            states.mean(axis=0)
        )

        feedback_max = (
            states.max(axis=0)
        )

        feedback_final = (
            states[-1]
        )

        delayed_state = (
            feedback_final
        )

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
            trace.observe(fired)

        if learn:
            plasticity.reset_episode()

        prompt_tokens = (
            example_samples[
                0
            ].prompt_tokens
        )

        prompt_summary = (
            stimulate_prompt(
                prompt_tokens,
                learn,
            )
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

            feedback_summary = (
                stimulate_feedback(
                    feedback_token,
                    learn,
                )
            )

            feature = np.concatenate(
                [
                    *prompt_summary,
                    *feedback_summary,
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            analysis = (
                decoder_gradient.analyze(
                    feature,
                    sample.target_token,
                )
            )

            update = None

            if learn:
                error = (
                    readout_error_from_raw(
                        analysis[
                            "desired_raw"
                        ]
                    )
                )

                update = (
                    plasticity.apply_error(
                        error
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

            samples = (
                validation_grouped[
                    example_id
                ]
            )

            for seed in (
                VALIDATION_NEURAL_SEEDS
            ):
                rows = run_episode(
                    samples,
                    seed,
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

        targets = np.asarray(targets)
        predictions = np.asarray(predictions)

        positions = np.asarray(
            positions,
            dtype=np.int32,
        )

        semantic_mask = np.isin(
            positions,
            SEMANTIC_POSITIONS,
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
                np.mean(losses)
            ),
        }

    # ========================================================
    # Checkpoint
    # ========================================================

    def save_checkpoint(
        epoch: int,
    ) -> Path:

        path = (
            PLASTICITY_DIR
            /
            f"phase2_expanded_epoch_{epoch:02}_weights.npz"
        )

        np.savez(
            path,
            edge_offsets=edge_offsets,
            pre_neurons=pre_neurons,
            post_neurons=post_neurons,
            original_weights=original_weights,
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
        )

        return path

    # ========================================================
    # Baseline
    # ========================================================

    baseline = evaluate_validation()

    baseline_checkpoint = (
        save_checkpoint(0)
    )

    history = [
        {
            "epoch": 0,
            "training": None,
            "validation": baseline,
            "weight_statistics": (
                plasticity.statistics()
            ),
            "checkpoint": (
                relative(
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
        "         EXPANDED-PLASTICITY BASELINE"
    )
    print(
        "===================================================="
    )

    print(
        f"Top-1:              "
        f"{baseline['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:           "
        f"{baseline['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Semantic accuracy:  "
        f"{baseline['semantic_accuracy'] * 100:.2f}%"
    )

    # ========================================================
    # Training
    # ========================================================

    logical_ids = sorted(
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
            f"      EXPANDED TRAINING EPOCH "
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
            len(logical_ids)
        )

        targets = []
        predictions = []
        losses = []

        update_events = 0
        changed_events = 0

        for number, order_index in enumerate(
            order,
            start=1,
        ):

            example_id = logical_ids[
                int(order_index)
            ]

            samples = train_grouped[
                example_id
            ]

            seed = (
                TRAIN_NEURAL_SEEDS[
                    (
                        epoch
                        +
                        number
                        -
                        2
                    )
                    %
                    len(TRAIN_NEURAL_SEEDS)
                ]
            )

            rows = run_episode(
                samples,
                seed,
                True,
            )

            episode_correct = 0

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

                losses.append(
                    row[
                        "cross_entropy"
                    ]
                )

                episode_correct += int(
                    row[
                        "correct"
                    ]
                )

                if row[
                    "update"
                ] is not None:
                    update_events += 1
                    changed_events += int(
                        row[
                            "update"
                        ][
                            "changed_synapses"
                        ]
                    )

            print(
                f"[{number:02}/"
                f"{len(logical_ids):02}] "
                f"{example_id:18} "
                f"seed={seed} "
                f"token_acc="
                f"{episode_correct / len(rows) * 100:6.2f}%"
            )

        train_metrics = {
            "top1_accuracy_before_online_update": float(
                accuracy_score(
                    targets,
                    predictions,
                )
            ),
            "macro_f1_before_online_update": float(
                f1_score(
                    targets,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "mean_cross_entropy_before_online_update": float(
                np.mean(losses)
            ),
            "update_events": int(
                update_events
            ),
            "sum_changed_synapse_events": int(
                changed_events
            ),
        }

        validation = (
            evaluate_validation()
        )

        weight_stats = (
            plasticity.statistics()
        )

        checkpoint_path = (
            save_checkpoint(epoch)
        )

        history.append(
            {
                "epoch": int(epoch),
                "training": train_metrics,
                "validation": validation,
                "weight_statistics": (
                    weight_stats
                ),
                "checkpoint": (
                    relative(
                        checkpoint_path
                    )
                ),
            }
        )

        print()

        print(
            f"Validation Top-1:             "
            f"{validation['top1_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Macro F1:          "
            f"{validation['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Validation semantic accuracy: "
            f"{validation['semantic_accuracy'] * 100:.2f}%"
        )

        print(
            f"Changed synapses:             "
            f"{weight_stats['changed_from_original']:,}"
        )

        print(
            f"Mean magnitude ratio:         "
            f"{weight_stats['mean_relative_magnitude']:.6f}x"
        )

    # ========================================================
    # Best development checkpoint
    # ========================================================

    def selection_key(row):

        v = row[
            "validation"
        ]

        return (
            v[
                "semantic_accuracy"
            ],
            v[
                "macro_f1"
            ],
            v[
                "top1_accuracy"
            ],
            -v[
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

    beats_baseline = bool(
        best_epoch > 0
        and
        selection_key(best_row)
        >
        selection_key(history[0])
    )

    np.savez(
        BEST_WEIGHTS_FILE,
        edge_offsets=edge_offsets,
        pre_neurons=pre_neurons,
        post_neurons=post_neurons,
        original_weights=original_weights,
        plastic_weights=best_weights,
        selected_epoch=np.asarray(
            best_epoch,
            dtype=np.int32,
        ),
        learning_rate=np.asarray(
            LEARNING_RATE,
            dtype=np.float32,
        ),
        beats_baseline=np.asarray(
            beats_baseline
        ),
    )

    protocol = {
        "phase": 2,
        "step": "79E",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": (
            "EXPANDED_PLASTICITY_DEVELOPMENT_COMPLETE"
        ),
        "old_final_holdout_used": False,
        "selected_epoch": (
            best_epoch
        ),
        "plastic_beats_no_learning_baseline": (
            beats_baseline
        ),
        "settings": {
            "maximum_plastic_synapses": (
                MAX_PLASTIC_SYNAPSES
            ),
            "subset_seed": (
                EXPANDED_SUBSET_SEED
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "pre_trace_decay": (
                PRE_TRACE_DECAY
            ),
            "training_epochs": (
                TRAINING_EPOCHS
            ),
            "block_error_weights": (
                BLOCK_ERROR_WEIGHTS.tolist()
            ),
        },
        "baseline_validation": (
            history[0][
                "validation"
            ]
        ),
        "best_validation": (
            best_row[
                "validation"
            ]
        ),
        "best_weights_file": (
            relative(
                BEST_WEIGHTS_FILE
            )
        ),
    }

    FROZEN_PROTOCOL_FILE.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    output = {
        "phase": 2,
        "step": "79E",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "old_final_holdout_used": False,
        "eligible_plastic_synapses": (
            subset[
                "eligible_count"
            ]
        ),
        "selected_plastic_synapses": int(
            len(edge_offsets)
        ),
        "history": history,
        "best_epoch": (
            best_epoch
        ),
        "plastic_beats_no_learning_baseline": (
            beats_baseline
        ),
        "baseline_validation": (
            history[0][
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

    baseline_v = history[
        0
    ][
        "validation"
    ]

    best_v = best_row[
        "validation"
    ]

    print()

    print(
        "===================================================="
    )
    print(
        "          STEP 79E LEARNING RESULT"
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
        f"{baseline_v['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{baseline_v['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{baseline_v['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        f"Best learned checkpoint "
        f"(epoch {best_epoch}):"
    )

    print(
        f"  Top-1:              "
        f"{best_v['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:           "
        f"{best_v['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:  "
        f"{best_v['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Learned minus baseline:"
    )

    print(
        f"  Top-1:              "
        f"{(best_v['top1_accuracy'] - baseline_v['top1_accuracy']) * 100:+.2f} pp"
    )

    print(
        f"  Macro F1:           "
        f"{(best_v['macro_f1'] - baseline_v['macro_f1']) * 100:+.2f} pp"
    )

    print(
        f"  Semantic accuracy:  "
        f"{(best_v['semantic_accuracy'] - baseline_v['semantic_accuracy']) * 100:+.2f} pp"
    )

    print()

    print(
        f"Plastic checkpoint beats baseline: "
        f"{beats_baseline}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{relative(EXPANDED_SUBSET_FILE)}"
    )

    print(
        "  "
        f"{relative(OUTPUT_METRICS_FILE)}"
    )

    print(
        "  "
        f"{relative(BEST_WEIGHTS_FILE)}"
    )

    print(
        "  "
        f"{relative(FROZEN_PROTOCOL_FILE)}"
    )

    print()

    print(
        "===================================================="
    )
    print(
        "                 STEP 79E COMPLETE"
    )
    print(
        "===================================================="
    )

    print()

    if beats_baseline:
        print(
            "Expanded recurrent plasticity improved "
            "unseen development validation."
        )

        print(
            "Next: freeze/retrain this rule cleanly and "
            "create a NEW untouched Phase-2 benchmark."
        )

    else:
        print(
            "Expanded recurrent plasticity still did not "
            "beat the no-learning reservoir."
        )

        print(
            "Next: move plasticity to an upstream hidden "
            "pathway / learn a dedicated fly output circuit."
        )


if __name__ == "__main__":
    main()
