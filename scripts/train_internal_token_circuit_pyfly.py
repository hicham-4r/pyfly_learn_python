from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Phase 2 — Plastic PyFly
# Step 79F — Dedicated internal token-output circuit
#
# Major change from Steps 79B-79E:
#
#   NO external decoder is used for prediction or learning.
#
# Instead, real MaleCNS neurons are assigned to the 9 Python
# token classes. Their neural activity IS the token output.
#
#                 MaleCNS
#                    ↓
#       9 real token-neuron populations
#                    ↓
#          highest activity = token
#
# Learning changes real recurrent synapses arriving at these
# token-output neurons.
#
# Supervised three-factor rule:
#
#   recent presynaptic activity
#            ×
#   token-population classification error
#            ×
#   synapse sign
#            ↓
#   recurrent synaptic magnitude update
#
# This is an engineering learning circuit, not a claim that
# the fly brain naturally represents Python tokens this way.
#
# Data rule:
#   generator_train      -> synaptic updates
#   generator_validation -> development evaluation only
#   old final holdout    -> NEVER opened
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda"


# ============================================================
# Development files only
# ============================================================

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


# ============================================================
# Previous Phase-2 result (must exist)
# ============================================================

PLASTICITY_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "plasticity"
)

STEP79E_FILE = (
    PLASTICITY_DIR
    / "phase2_expanded_supervised_training.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_CIRCUIT_FILE = (
    PLASTICITY_DIR
    / "phase2_internal_token_circuit.npz"
)

OUTPUT_METRICS_FILE = (
    PLASTICITY_DIR
    / "phase2_internal_token_training.json"
)

BEST_WEIGHTS_FILE = (
    PLASTICITY_DIR
    / "phase2_internal_token_best_weights.npz"
)

FROZEN_PROTOCOL_FILE = (
    PLASTICITY_DIR
    / "phase2_internal_token_frozen_protocol.json"
)


# ============================================================
# Frozen PyFly stimulation
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

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Step-79F precommitted output-circuit settings
# ============================================================

OUTPUT_SELECTION_SEED = 79_600

OUTPUT_NEURONS_PER_TOKEN = 32

# Number of top-indegree indirect visual neurons placed in the
# candidate pool before deterministic token-group assignment.
OUTPUT_CANDIDATE_POOL = 1_024

TRAINING_EPOCHS = 5

LEARNING_RATE = 0.005

PRE_TRACE_DECAY = 0.90

MIN_RELATIVE_MAGNITUDE = 0.25
MAX_RELATIVE_MAGNITUDE = 2.00

# Convert raw group activity to softmax probabilities.
SCORE_TEMPERATURE = 0.10

# Small penalty prevents unbounded competition updates.
ERROR_CLIP = 1.0

TRAINING_ORDER_SEED = 79_610

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

def relative(
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

        if positions != list(
            range(
                len(
                    positions
                )
            )
        ):

            raise ValueError(
                "Non-contiguous target positions "
                f"for {example_id}: {positions}"
            )

    return grouped


def softmax(
    values: np.ndarray,
    temperature: float,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    scaled = (
        values
        /
        max(
            float(
                temperature
            ),
            1e-8,
        )
    )

    scaled -= np.max(
        scaled
    )

    exp_values = np.exp(
        scaled
    )

    return (
        exp_values
        /
        np.sum(
            exp_values
        )
    )


# ============================================================
# Frozen dual-channel input map
# ============================================================

def load_dual_map(
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

    prompt_mapping = {}

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
                str(
                    token
                ),
                int(
                    position
                ),
            )
        ] = (
            prompt_neurons[
                index
            ].copy()
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

    feedback_mapping = {}

    for index, token in enumerate(
        feedback_tokens
    ):

        feedback_mapping[
            str(
                token
            )
        ] = (
            feedback_neurons[
                index
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
            "Prompt-position limit changed."
        )

    if result.seed != MAP_SEED:

        raise ValueError(
            "Mapping seed changed."
        )

    return result


# ============================================================
# Build internal token-output circuit
#
# We select high-indegree neurons from the indirect visual
# population because they have many recurrent inputs available
# for learning. Then deterministically shuffle/partition them
# into equal token groups.
#
# Labels are used ONLY to determine how many output groups are
# required; validation performance is not used in selection.
# ============================================================

def build_token_output_circuit(
    brain: FlyBrain,
    readout_population: np.ndarray,
    token_classes: list[str],
) -> dict:

    W_indptr = (
        brain._W.indptr.get().astype(
            np.int64,
            copy=False,
        )
    )

    indegree = (
        W_indptr[
            readout_population
            +
            1
        ]
        -
        W_indptr[
            readout_population
        ]
    )

    candidate_count = min(
        OUTPUT_CANDIDATE_POOL,
        len(
            readout_population
        ),
    )

    top_order = np.argsort(
        indegree
    )[
        -candidate_count:
    ]

    candidate_neurons = (
        readout_population[
            top_order
        ].copy()
    )

    needed = (
        len(
            token_classes
        )
        *
        OUTPUT_NEURONS_PER_TOKEN
    )

    if len(
        candidate_neurons
    ) < needed:

        raise ValueError(
            "Not enough candidate output neurons."
        )

    rng = np.random.default_rng(
        OUTPUT_SELECTION_SEED
    )

    candidate_neurons = rng.permutation(
        candidate_neurons
    )

    selected = candidate_neurons[
        :needed
    ]

    groups = selected.reshape(
        len(
            token_classes
        ),
        OUTPUT_NEURONS_PER_TOKEN,
    )

    # --------------------------------------------------------
    # Gather every recurrent edge arriving at output neurons.
    # --------------------------------------------------------

    W_indices = (
        brain._W.indices.get().astype(
            np.int64,
            copy=False,
        )
    )

    W_weights = (
        brain._W.data.get().astype(
            np.float32,
            copy=False,
        )
    )

    edge_offsets = []
    pre_neurons = []
    post_neurons = []
    edge_group_indices = []

    for group_index, neurons in enumerate(
        groups
    ):

        for post in neurons:

            start = int(
                W_indptr[
                    post
                ]
            )

            end = int(
                W_indptr[
                    post
                    +
                    1
                ]
            )

            if start == end:
                continue

            offsets = np.arange(
                start,
                end,
                dtype=np.int64,
            )

            edge_offsets.append(
                offsets
            )

            pre_neurons.append(
                W_indices[
                    start:end
                ].astype(
                    np.int64,
                    copy=False,
                )
            )

            post_neurons.append(
                np.full(
                    end - start,
                    int(
                        post
                    ),
                    dtype=np.int64,
                )
            )

            edge_group_indices.append(
                np.full(
                    end - start,
                    int(
                        group_index
                    ),
                    dtype=np.int32,
                )
            )

    edge_offsets = np.concatenate(
        edge_offsets
    )

    pre_neurons = np.concatenate(
        pre_neurons
    )

    post_neurons = np.concatenate(
        post_neurons
    )

    edge_group_indices = np.concatenate(
        edge_group_indices
    )

    original_weights = (
        W_weights[
            edge_offsets
        ].copy()
    )

    if np.any(
        original_weights
        ==
        0
    ):

        raise RuntimeError(
            "Token-output circuit contains "
            "zero-weight recurrent edges."
        )

    return {
        "token_classes": (
            token_classes
        ),

        "output_groups": (
            groups.astype(
                np.int64
            )
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

        "edge_group_indices": (
            edge_group_indices
        ),

        "original_weights": (
            original_weights
        ),

        "candidate_pool_size": int(
            candidate_count
        ),
    }


# ============================================================
# Local recurrent output-circuit plasticity
# ============================================================

class TokenCircuitPlasticity:

    def __init__(
        self,
        brain: FlyBrain,
        circuit: dict,
    ) -> None:

        import cupy as cp

        self.cp = cp
        self.brain = brain

        self.edge_offsets = cp.asarray(
            circuit[
                "edge_offsets"
            ],
            dtype=cp.int64,
        )

        self.pre_neurons = cp.asarray(
            circuit[
                "pre_neurons"
            ],
            dtype=cp.int64,
        )

        self.edge_group_indices = cp.asarray(
            circuit[
                "edge_group_indices"
            ],
            dtype=cp.int32,
        )

        self.original_weights = cp.asarray(
            circuit[
                "original_weights"
            ],
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

    def apply_class_error(
        self,
        group_error_cpu: np.ndarray,
    ) -> dict:

        cp = self.cp

        group_error = cp.asarray(
            group_error_cpu,
            dtype=cp.float32,
        )

        pre_credit = (
            self.pre_trace[
                self.pre_neurons
            ]
        )

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
            }

        pre_credit = (
            pre_credit
            /
            np.float32(
                pre_max
            )
        )

        error_for_edge = (
            group_error[
                self.edge_group_indices
            ]
        )

        # ----------------------------------------------------
        # Convert desired output-drive change to synaptic
        # magnitude change while preserving synapse sign.
        # ----------------------------------------------------

        credit = (
            pre_credit
            *
            error_for_edge
            *
            self.original_sign
        )

        credit = cp.clip(
            credit,
            -ERROR_CLIP,
            +ERROR_CLIP,
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
        "       STEP 79F - INTERNAL TOKEN CIRCUIT"
    )

    print(
        "===================================================="
    )

    print()

    required = [
        TRAIN_FILE,
        VALIDATION_FILE,
        MAPPING_FILE,
        STEP79E_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: "
                f"{path}"
            )

    step79e = load_json(
        STEP79E_FILE
    )

    print(
        "Step 79E complete:             YES"
    )

    print(
        f"Step 79E plastic win:         "
        f"{step79e.get('plastic_beats_no_learning_baseline')}"
    )

    print(
        "External decoder used:        NO"
    )

    print(
        "Old final holdout used:       NO"
    )

    print(
        f"Output neurons per token:     "
        f"{OUTPUT_NEURONS_PER_TOKEN}"
    )

    print(
        f"Training epochs:              "
        f"{TRAINING_EPOCHS}"
    )

    print(
        f"Learning rate:                "
        f"{LEARNING_RATE}"
    )

    # ========================================================
    # Development data
    # ========================================================

    train_samples = (
        load_generator_jsonl(
            TRAIN_FILE
        )
    )

    validation_samples = (
        load_generator_jsonl(
            VALIDATION_FILE
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

    token_classes = sorted(
        {
            sample.target_token

            for sample
            in train_samples
        }
    )

    validation_tokens = {
        sample.target_token

        for sample
        in validation_samples
    }

    if not validation_tokens.issubset(
        set(
            token_classes
        )
    ):

        raise ValueError(
            "Validation contains target tokens "
            "not present in training."
        )

    print()

    print(
        f"Token classes ({len(token_classes)}):"
    )

    print(
        "  "
        +
        ", ".join(
            token_classes
        )
    )

    # ========================================================
    # Frozen input map + MaleCNS
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
            "Unexpected MaleCNS neuron count."
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
            "Indirect readout population changed."
        )

    # ========================================================
    # Create the dedicated neural token circuit
    # ========================================================

    circuit = build_token_output_circuit(
        brain,
        readout_population,
        token_classes,
    )

    output_groups = circuit[
        "output_groups"
    ]

    total_output_neurons = int(
        output_groups.size
    )

    print()

    print(
        f"Token-output neurons:         "
        f"{total_output_neurons}"
    )

    print(
        f"Plastic recurrent synapses:   "
        f"{len(circuit['edge_offsets']):,}"
    )

    print(
        f"Excitatory plastic synapses:  "
        f"{np.sum(circuit['original_weights'] > 0):,}"
    )

    print(
        f"Inhibitory plastic synapses:  "
        f"{np.sum(circuit['original_weights'] < 0):,}"
    )

    np.savez(
        OUTPUT_CIRCUIT_FILE,

        token_classes=np.asarray(
            token_classes
        ),

        output_groups=(
            output_groups
        ),

        edge_offsets=(
            circuit[
                "edge_offsets"
            ]
        ),

        pre_neurons=(
            circuit[
                "pre_neurons"
            ]
        ),

        post_neurons=(
            circuit[
                "post_neurons"
            ]
        ),

        edge_group_indices=(
            circuit[
                "edge_group_indices"
            ]
        ),

        original_weights=(
            circuit[
                "original_weights"
            ]
        ),

        output_selection_seed=np.asarray(
            OUTPUT_SELECTION_SEED,
            dtype=np.int64,
        ),
    )

    # ========================================================
    # Trace ONLY the token-output neurons
    # ========================================================

    flat_output_neurons = (
        output_groups.reshape(
            -1
        )
    )

    trace = Trace(
        brain,
        idx=flat_output_neurons,
        tau=TRACE_TAU,
    )

    plasticity = TokenCircuitPlasticity(
        brain,
        circuit,
    )

    token_to_index = {
        token: index

        for index, token
        in enumerate(
            token_classes
        )
    }

    # ========================================================
    # Score output groups
    # ========================================================

    def neural_group_scores(
        state: np.ndarray,
    ) -> np.ndarray:

        grouped = state.reshape(
            len(
                token_classes
            ),
            OUTPUT_NEURONS_PER_TOKEN,
        )

        return grouped.mean(
            axis=1
        )

    # ========================================================
    # One simulator step
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

    # ========================================================
    # Prompt stimulation
    # ========================================================

    def stimulate_prompt(
        prompt_tokens,
        learn,
    ):

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

                brain_step(
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ],
                    learn=learn,
                )

            for _ in range(
                PROMPT_GAP_STEPS
            ):

                brain_step(
                    learn=learn,
                )

    # ========================================================
    # Feedback stimulation + output measurement
    # ========================================================

    def stimulate_feedback(
        token,
        learn,
    ) -> np.ndarray:

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

        for _ in range(
            FEEDBACK_GAP_STEPS
        ):

            states.append(
                brain_step(
                    learn=learn,
                )
            )

        states = np.stack(
            states
        )

        # Combine mean and final neural state.
        return (
            0.50
            *
            states.mean(
                axis=0
            )
            +
            0.50
            *
            states[
                -1
            ]
        ).astype(
            np.float32,
            copy=False,
        )

    # ========================================================
    # One teacher-forced episode
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

        stimulate_prompt(
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
                    position
                    -
                    1
                ].target_token
            )

            output_state = (
                stimulate_feedback(
                    feedback_token,
                    learn,
                )
            )

            scores = neural_group_scores(
                output_state
            )

            probabilities = softmax(
                scores,
                SCORE_TEMPERATURE,
            )

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            predicted_token = (
                token_classes[
                    predicted_index
                ]
            )

            target_index = (
                token_to_index[
                    sample.target_token
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

            correct = (
                predicted_token
                ==
                sample.target_token
            )

            update = None

            if learn:

                # ------------------------------------------------
                # Desired change in token-population drive:
                #
                #   one_hot(target) - softmax(current activity)
                # ------------------------------------------------

                group_error = (
                    -probabilities
                ).astype(
                    np.float32,
                    copy=False,
                )

                group_error[
                    target_index
                ] += np.float32(
                    1.0
                )

                update = (
                    plasticity.apply_class_error(
                        group_error
                    )
                )

            rows.append(
                {
                    "target": (
                        sample.target_token
                    ),

                    "prediction": (
                        predicted_token
                    ),

                    "correct": bool(
                        correct
                    ),

                    "position": int(
                        sample.target_position
                    ),

                    "target_probability": (
                        target_probability
                    ),

                    "cross_entropy": (
                        cross_entropy
                    ),

                    "scores": (
                        scores.tolist()
                    ),

                    "update": (
                        update
                    ),
                }
            )

        return rows

    # ========================================================
    # Evaluate validation
    # ========================================================

    def evaluate_validation():

        targets = []
        predictions = []
        positions = []
        losses = []

        per_seed = defaultdict(
            lambda: {
                "targets": [],
                "predictions": [],
            }
        )

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

                    per_seed[
                        seed
                    ][
                        "targets"
                    ].append(
                        row[
                            "target"
                        ]
                    )

                    per_seed[
                        seed
                    ][
                        "predictions"
                    ].append(
                        row[
                            "prediction"
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

        seed_accuracy = {}

        for seed in sorted(
            per_seed
        ):

            seed_accuracy[
                str(
                    seed
                )
            ] = float(
                accuracy_score(
                    per_seed[
                        seed
                    ][
                        "targets"
                    ],
                    per_seed[
                        seed
                    ][
                        "predictions"
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

            "per_seed_top1": (
                seed_accuracy
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
            f"phase2_internal_token_epoch_"
            f"{epoch:02}_weights.npz"
        )

        np.savez(
            path,

            token_classes=np.asarray(
                token_classes
            ),

            output_groups=(
                output_groups
            ),

            edge_offsets=(
                circuit[
                    "edge_offsets"
                ]
            ),

            pre_neurons=(
                circuit[
                    "pre_neurons"
                ]
            ),

            post_neurons=(
                circuit[
                    "post_neurons"
                ]
            ),

            edge_group_indices=(
                circuit[
                    "edge_group_indices"
                ]
            ),

            original_weights=(
                circuit[
                    "original_weights"
                ]
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
        )

        return path

    # ========================================================
    # Baseline
    # ========================================================

    baseline = evaluate_validation()

    baseline_checkpoint = save_checkpoint(
        0
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
        "       INTERNAL TOKEN CIRCUIT BASELINE"
    )

    print(
        "===================================================="
    )

    print()

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

    print(
        f"Cross-entropy:      "
        f"{baseline['mean_cross_entropy']:.6f}"
    )

    # ========================================================
    # Training
    # ========================================================

    logical_train_ids = sorted(
        train_grouped.keys()
    )

    for epoch in range(
        1,
        TRAINING_EPOCHS
        +
        1,
    ):

        print()

        print(
            "===================================================="
        )

        print(
            f"        INTERNAL TOKEN TRAINING EPOCH "
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
        train_losses = []

        update_events = 0
        changed_synapse_events = 0

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

            samples = (
                train_grouped[
                    example_id
                ]
            )

            seed = (
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
                seed,
                True,
            )

            episode_correct = 0

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

                train_losses.append(
                    row[
                        "cross_entropy"
                    ]
                )

                episode_correct += int(
                    row[
                        "correct"
                    ]
                )

                if (
                    row[
                        "update"
                    ]
                    is not None
                ):

                    update_events += 1

                    changed_synapse_events += int(
                        row[
                            "update"
                        ][
                            "changed_synapses"
                        ]
                    )

            print(
                f"[{episode_number:02}/"
                f"{len(logical_train_ids):02}] "
                f"{example_id:18} "
                f"seed={seed} "
                f"token_acc="
                f"{episode_correct / len(rows) * 100:6.2f}%"
            )

        train_metrics = {
            "top1_accuracy_before_online_update": float(
                accuracy_score(
                    train_targets,
                    train_predictions,
                )
            ),

            "macro_f1_before_online_update": float(
                f1_score(
                    train_targets,
                    train_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

            "mean_cross_entropy_before_online_update": float(
                np.mean(
                    train_losses
                )
            ),

            "update_events": int(
                update_events
            ),

            "sum_changed_synapse_events": int(
                changed_synapse_events
            ),
        }

        validation = evaluate_validation()

        weight_stats = (
            plasticity.statistics()
        )

        checkpoint = save_checkpoint(
            epoch
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),

                "training": (
                    train_metrics
                ),

                "validation": (
                    validation
                ),

                "weight_statistics": (
                    weight_stats
                ),

                "checkpoint": (
                    relative(
                        checkpoint
                    )
                ),
            }
        )

        print()

        print(
            f"Training Top-1:              "
            f"{train_metrics['top1_accuracy_before_online_update'] * 100:.2f}%"
        )

        print(
            f"Validation Top-1:            "
            f"{validation['top1_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Macro F1:         "
            f"{validation['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Validation semantic:         "
            f"{validation['semantic_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation cross-entropy:    "
            f"{validation['mean_cross_entropy']:.6f}"
        )

        print(
            f"Synapses changed:            "
            f"{weight_stats['changed_from_original']:,}"
        )

        print(
            f"Mean magnitude ratio:        "
            f"{weight_stats['mean_relative_magnitude']:.6f}x"
        )

    # ========================================================
    # Select best development epoch
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

    best_weights = (
        best_checkpoint[
            "plastic_weights"
        ].astype(
            np.float32
        )
    )

    beats_baseline = bool(
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

    np.savez(
        BEST_WEIGHTS_FILE,

        token_classes=np.asarray(
            token_classes
        ),

        output_groups=(
            output_groups
        ),

        edge_offsets=(
            circuit[
                "edge_offsets"
            ]
        ),

        pre_neurons=(
            circuit[
                "pre_neurons"
            ]
        ),

        post_neurons=(
            circuit[
                "post_neurons"
            ]
        ),

        edge_group_indices=(
            circuit[
                "edge_group_indices"
            ]
        ),

        original_weights=(
            circuit[
                "original_weights"
            ]
        ),

        plastic_weights=(
            best_weights
        ),

        selected_epoch=np.asarray(
            best_epoch,
            dtype=np.int32,
        ),

        beats_baseline=np.asarray(
            beats_baseline
        ),
    )

    # ========================================================
    # Save report/protocol
    # ========================================================

    protocol = {
        "phase": 2,

        "step": "79F",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "INTERNAL_TOKEN_CIRCUIT_DEVELOPMENT_COMPLETE"
        ),

        "old_final_holdout_used": (
            False
        ),

        "external_decoder_used": (
            False
        ),

        "token_classes": (
            token_classes
        ),

        "output_neurons_per_token": (
            OUTPUT_NEURONS_PER_TOKEN
        ),

        "selected_epoch": (
            best_epoch
        ),

        "plastic_beats_no_learning_baseline": (
            beats_baseline
        ),

        "settings": {
            "output_selection_seed": (
                OUTPUT_SELECTION_SEED
            ),

            "output_candidate_pool": (
                OUTPUT_CANDIDATE_POOL
            ),

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

            "score_temperature": (
                SCORE_TEMPERATURE
            ),

            "training_epochs": (
                TRAINING_EPOCHS
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
            "circuit_file": (
                relative(
                    OUTPUT_CIRCUIT_FILE
                )
            ),

            "best_weights_file": (
                relative(
                    BEST_WEIGHTS_FILE
                )
            ),
        },
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

        "step": "79F",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "old_final_holdout_used": (
            False
        ),

        "external_decoder_used": (
            False
        ),

        "token_classes": (
            token_classes
        ),

        "output_neurons": (
            total_output_neurons
        ),

        "plastic_synapses": int(
            len(
                circuit[
                    "edge_offsets"
                ]
            )
        ),

        "history": (
            history
        ),

        "best_epoch": (
            best_epoch
        ),

        "plastic_beats_no_learning_baseline": (
            beats_baseline
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
        "          STEP 79F LEARNING RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Internal token circuit BEFORE learning:"
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
        "Learned minus internal-circuit baseline:"
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
        f"Plastic token circuit beats baseline: "
        f"{beats_baseline}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{relative(OUTPUT_CIRCUIT_FILE)}"
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
        "                 STEP 79F COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    if beats_baseline:

        print(
            "SUCCESS: real MaleCNS recurrent synapses "
            "improved a token output produced directly "
            "by MaleCNS neurons."
        )

        print()

        print(
            "Next: freeze this internal output-learning "
            "mechanism and build a NEW untouched "
            "Phase-2 benchmark."
        )

    else:

        print(
            "The direct internal token circuit still did "
            "not improve enough."
        )

        print()

        print(
            "Next: add teacher-clamped target-neuron "
            "activity / homeostatic competition to the "
            "internal circuit, still development-only."
        )


if __name__ == "__main__":
    main()
