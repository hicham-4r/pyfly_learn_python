from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import DualChannelMap
from pyfly.tokenizer import code_to_tokens, tokens_to_code


# ============================================================
# Phase 3 — Trainable PyFly
# Step 83 — ONE-TIME FINAL EVALUATION
#
# Evaluates the FROZEN Step-80/81 Phase-3 pipeline on the
# fresh Step-82 holdout.
#
# Two final modes:
#
#   A) Teacher-forced:
#      ground-truth previous token is fed back to MaleCNS.
#
#   B) True closed-loop:
#      PyFly's own predicted token is fed back to MaleCNS.
#
# ABSOLUTELY NO:
#   * fitting
#   * gradient updates
#   * architecture changes
#   * hyperparameter tuning
#   * beam search
#   * top-k rescue
#
# Greedy Top-1 is used for closed-loop generation.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_NAME = "cuda"


# ============================================================
# Frozen Step-82 final protocol + fresh holdout
# ============================================================

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_final_evaluation_protocol.json"
)

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phase3_fresh_holdout.jsonl"
)


# ============================================================
# Frozen Step-80/81 pipeline
# ============================================================

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "phase3_pyfly_adapter.pt"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "phase3_pyfly_adapter_scaler.npz"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)

STEP80_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_pyfly_adapter_training.json"
)

STEP81_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_closed_loop_development_test.json"
)


# ============================================================
# Final outputs
# ============================================================

FINAL_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_final_holdout_metrics.json"
)

TEACHER_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_final_holdout_teacher_forced_predictions.jsonl"
)

CLOSED_LOOP_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_final_holdout_closed_loop_predictions.jsonl"
)


# ============================================================
# Frozen MaleCNS stimulation configuration
# ============================================================

PROMPT_TOKEN_STEPS = 8
PROMPT_GAP_STEPS = 2

FEEDBACK_TOKEN_STEPS = 8
FEEDBACK_GAP_STEPS = 1

WARMUP_STEPS = 10

AMPLITUDE = 0.8
TRACE_TAU = 0.25

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873
EXPECTED_FEATURE_DIM = 41_111

EXPECTED_LOGICAL_PROMPTS = 24
EXPECTED_INTENTS = 6
EXPECTED_PROMPTS_PER_INTENT = 4

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Exact Step-80 model architecture
# ============================================================

class PyFlyNeuralAdapter(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_position: int,
        hidden_dim: int,
        second_hidden_dim: int,
        position_embed_dim: int,
        dropout: float,
        input_dropout: float,
    ) -> None:

        super().__init__()

        self.input_norm = nn.LayerNorm(
            input_dim
        )

        self.input_dropout = nn.Dropout(
            input_dropout
        )

        self.neural_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.position_embedding = nn.Embedding(
            max_position + 1,
            position_embed_dim,
        )

        merged_dim = (
            hidden_dim
            +
            position_embed_dim
        )

        self.adapter = nn.Sequential(
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                merged_dim,
                second_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.LayerNorm(
                second_hidden_dim
            ),
        )

        self.classifier = nn.Linear(
            second_hidden_dim,
            num_classes,
        )

    def forward(
        self,
        X: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:

        X = self.input_norm(
            X
        )

        X = self.input_dropout(
            X
        )

        neural_hidden = (
            self.neural_projection(
                X
            )
        )

        position_hidden = (
            self.position_embedding(
                positions
            )
        )

        merged = torch.cat(
            [
                neural_hidden,
                position_hidden,
            ],
            dim=1,
        )

        hidden = self.adapter(
            merged
        )

        return self.classifier(
            hidden
        )


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


def safe_mean(
    values,
) -> float:

    values = list(
        values
    )

    if not values:

        return 0.0

    return float(
        np.mean(
            values
        )
    )


def macro_f1_from_labels(
    true_labels: list[str],
    predicted_labels: list[str],
    classes: list[str],
) -> float:

    true_labels = np.asarray(
        true_labels,
        dtype=str,
    )

    predicted_labels = np.asarray(
        predicted_labels,
        dtype=str,
    )

    f1_values = []

    for token in classes:

        true_positive = int(
            np.sum(
                (
                    true_labels
                    ==
                    token
                )
                &
                (
                    predicted_labels
                    ==
                    token
                )
            )
        )

        false_positive = int(
            np.sum(
                (
                    true_labels
                    !=
                    token
                )
                &
                (
                    predicted_labels
                    ==
                    token
                )
            )
        )

        false_negative = int(
            np.sum(
                (
                    true_labels
                    ==
                    token
                )
                &
                (
                    predicted_labels
                    !=
                    token
                )
            )
        )

        denominator = (
            2
            *
            true_positive
            +
            false_positive
            +
            false_negative
        )

        if denominator == 0:

            f1 = 0.0

        else:

            f1 = (
                2
                *
                true_positive
                /
                denominator
            )

        f1_values.append(
            float(
                f1
            )
        )

    return safe_mean(
        f1_values
    )


def sequence_token_accuracy(
    target: list[str],
    predicted: list[str],
) -> float:

    denominator = max(
        len(
            target
        ),
        len(
            predicted
        ),
    )

    if denominator == 0:

        return 1.0

    correct = 0

    for index in range(
        denominator
    ):

        true_token = (
            target[
                index
            ]
            if index
            <
            len(
                target
            )
            else None
        )

        predicted_token = (
            predicted[
                index
            ]
            if index
            <
            len(
                predicted
            )
            else None
        )

        if true_token == predicted_token:

            correct += 1

    return float(
        correct
        /
        denominator
    )


def first_error_position(
    target: list[str],
    predicted: list[str],
):

    length = max(
        len(
            target
        ),
        len(
            predicted
        ),
    )

    for index in range(
        length
    ):

        true_token = (
            target[
                index
            ]
            if index
            <
            len(
                target
            )
            else None
        )

        predicted_token = (
            predicted[
                index
            ]
            if index
            <
            len(
                predicted
            )
            else None
        )

        if true_token != predicted_token:

            return int(
                index
            )

    return None


def decode_tokens(
    tokens: list[str],
):

    try:

        code = tokens_to_code(
            tokens
        )

        template_valid = True

    except Exception:

        return (
            None,
            False,
            False,
        )

    try:

        ast.parse(
            code
        )

        syntax_valid = True

    except SyntaxError:

        syntax_valid = False

    return (
        code,
        template_valid,
        syntax_valid,
    )


# ============================================================
# Verify frozen Step-82 protocol
# ============================================================

def verify_frozen_protocol() -> dict:

    required = [
        PROTOCOL_FILE,
        FINAL_HOLDOUT_FILE,
        MODEL_FILE,
        SCALER_FILE,
        MAPPING_FILE,
        STEP80_METRICS_FILE,
        STEP81_METRICS_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required frozen file missing: "
                f"{path}"
            )

    protocol = load_json(
        PROTOCOL_FILE
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_PHASE3_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Step-82 protocol is not marked frozen."
        )

    if (
        protocol.get(
            "scoring_performed_in_step82"
        )
        is not False
    ):

        raise ValueError(
            "Step-82 protocol does not confirm "
            "that the holdout remained unscored."
        )

    frozen = protocol.get(
        "frozen_artifacts",
        {}
    )

    expected = {
        "model": MODEL_FILE,
        "scaler": SCALER_FILE,
        "mapping": MAPPING_FILE,
        "step80_metrics": STEP80_METRICS_FILE,
        "step81_metrics": STEP81_METRICS_FILE,
        "fresh_holdout": FINAL_HOLDOUT_FILE,
    }

    for key, path in expected.items():

        info = frozen.get(
            key
        )

        if not isinstance(
            info,
            dict,
        ):

            raise ValueError(
                f"Frozen protocol missing artifact "
                f"entry {key!r}."
            )

        frozen_hash = str(
            info.get(
                "sha256",
                "",
            )
        )

        actual_hash = sha256_file(
            path
        )

        if (
            not frozen_hash
            or
            frozen_hash
            !=
            actual_hash
        ):

            raise ValueError(
                f"Frozen artifact hash mismatch "
                f"for {key}: {path}"
            )

    return protocol


# ============================================================
# Load exact frozen neural input map
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

    return DualChannelMap(
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


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        "       PHASE 3 - TRAINABLE PYFLY"
    )

    print(
        "       STEP 83 - ONE-TIME FINAL EVALUATION"
    )

    print(
        "===================================================="
    )

    print()

    protocol = verify_frozen_protocol()

    print(
        "Step-82 frozen protocol integrity: PASS"
    )

    print(
        "Training performed in Step 83:      NO"
    )

    print(
        "Hyperparameter tuning in Step 83:   NO"
    )

    print(
        "Closed-loop decoding:               greedy Top-1"
    )

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available to PyTorch."
        )

    device = torch.device(
        DEVICE_NAME
    )

    # ========================================================
    # Load frozen Step-80 adapter
    # ========================================================

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    token_classes = [
        str(
            token
        )

        for token
        in checkpoint[
            "token_classes"
        ]
    ]

    token_to_id = {
        token: index

        for index, token
        in enumerate(
            token_classes
        )
    }

    input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    if (
        input_dim
        !=
        EXPECTED_FEATURE_DIM
    ):

        raise ValueError(
            f"Expected model input dimension "
            f"{EXPECTED_FEATURE_DIM}, "
            f"got {input_dim}."
        )

    model = PyFlyNeuralAdapter(
        input_dim=(
            input_dim
        ),

        num_classes=len(
            token_classes
        ),

        max_position=int(
            checkpoint[
                "max_position"
            ]
        ),

        hidden_dim=int(
            checkpoint[
                "hidden_dim"
            ]
        ),

        second_hidden_dim=int(
            checkpoint[
                "second_hidden_dim"
            ]
        ),

        position_embed_dim=int(
            checkpoint[
                "position_embed_dim"
            ]
        ),

        dropout=float(
            checkpoint[
                "dropout"
            ]
        ),

        input_dropout=float(
            checkpoint[
                "input_dropout"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    max_position = int(
        checkpoint[
            "max_position"
        ]
    )

    max_generation_steps = (
        max_position
        +
        1
    )

    print()

    print(
        f"GPU:                                "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Frozen Step-80 best epoch:          "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"Target classes:                     "
        f"{len(token_classes)}"
    )

    print(
        f"Maximum output positions:           "
        f"{max_generation_steps}"
    )

    # ========================================================
    # Load frozen Step-80 scaler
    # ========================================================

    scaler = np.load(
        SCALER_FILE,
        allow_pickle=False,
    )

    feature_mean = scaler[
        "mean"
    ].astype(
        np.float32
    )

    feature_std = scaler[
        "std"
    ].astype(
        np.float32
    )

    if (
        feature_mean.shape
        !=
        (
            EXPECTED_FEATURE_DIM,
        )
        or
        feature_std.shape
        !=
        (
            EXPECTED_FEATURE_DIM,
        )
    ):

        raise ValueError(
            "Frozen Step-80 scaler dimensions changed."
        )

    # ========================================================
    # Load fresh Step-82 holdout
    # ========================================================

    examples = load_dataset(
        FINAL_HOLDOUT_FILE
    )

    if (
        len(
            examples
        )
        !=
        EXPECTED_LOGICAL_PROMPTS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_LOGICAL_PROMPTS} fresh "
            f"holdout prompts, got "
            f"{len(examples)}."
        )

    intent_counts = Counter(
        str(
            example.intent
        )

        for example
        in examples
    )

    if (
        len(
            intent_counts
        )
        !=
        EXPECTED_INTENTS
    ):

        raise ValueError(
            f"Expected {EXPECTED_INTENTS} intents, "
            f"got {dict(intent_counts)}."
        )

    for intent, count in (
        intent_counts.items()
    ):

        if (
            count
            !=
            EXPECTED_PROMPTS_PER_INTENT
        ):

            raise ValueError(
                f"Intent {intent!r} has {count} "
                f"holdout prompts; expected "
                f"{EXPECTED_PROMPTS_PER_INTENT}."
            )

    print()

    print(
        f"Fresh logical prompts:              "
        f"{len(examples)}"
    )

    print(
        f"Intents:                            "
        f"{len(intent_counts)}"
    )

    print(
        f"Neural seeds per prompt:            "
        f"{len(NEURAL_SEEDS)}"
    )

    # ========================================================
    # Validate target sequences + neural mappings before score
    # ========================================================

    target_by_example = {}

    for example in examples:

        target_tokens = list(
            code_to_tokens(
                example.code
            )
        )

        if (
            len(
                target_tokens
            )
            >
            max_generation_steps
        ):

            raise ValueError(
                f"{example.id}: target has "
                f"{len(target_tokens)} positions, "
                f"but frozen model supports only "
                f"{max_generation_steps}."
            )

        unknown_targets = [
            token

            for token
            in target_tokens

            if token
            not in
            token_to_id
        ]

        if unknown_targets:

            raise ValueError(
                f"{example.id}: target contains tokens "
                f"unknown to frozen model: "
                f"{unknown_targets}"
            )

        target_by_example[
            example.id
        ] = (
            target_tokens
        )

    # ========================================================
    # Frozen neural mapping + real MaleCNS
    # ========================================================

    dual_map = load_dual_map(
        MAPPING_FILE
    )

    required_feedback_tokens = (
        set(
            token_classes
        )
        |
        {
            "BOS"
        }
    )

    missing_feedback = sorted(
        required_feedback_tokens
        -
        set(
            dual_map.feedback_mapping.keys()
        )
    )

    if missing_feedback:

        raise ValueError(
            "Frozen neural feedback map is missing "
            f"tokens: {missing_feedback}"
        )

    # Verify every fresh prompt channel exists BEFORE scoring.
    for example in examples:

        for position, token in enumerate(
            example.prompt_tokens
        ):

            try:

                dual_map.prompt_neurons_for(
                    token,
                    position,
                )

            except Exception as exc:

                raise ValueError(
                    f"{example.id}: frozen prompt map "
                    f"does not support token={token!r} "
                    f"at position={position}."
                ) from exc

    print()

    print(
        "Loading real MaleCNS on CUDA..."
    )

    brain = FlyBrain(
        device="cuda",
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
            "Frozen direct-input population changed."
        )

    if (
        len(
            readout_population
        )
        !=
        EXPECTED_READOUT_NEURONS
    ):

        raise ValueError(
            "Frozen indirect-readout population changed."
        )

    if (
        len(
            readout_population
        )
        *
        7
        !=
        EXPECTED_FEATURE_DIM
    ):

        raise ValueError(
            "Frozen feature dimension changed."
        )

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    print(
        f"Direct input neurons:               "
        f"{len(input_neurons):,}"
    )

    print(
        f"Indirect readout neurons:           "
        f"{len(readout_population):,}"
    )

    # ========================================================
    # Frozen live MaleCNS stimulation
    # ========================================================

    def warmup():

        for _ in range(
            WARMUP_STEPS
        ):

            fired = brain.step()

            trace.observe(
                fired
            )

    def stimulate_prompt(
        prompt_tokens,
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

                fired = brain.step(
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ]
                )

                states.append(
                    np.asarray(
                        trace.observe(
                            fired
                        )
                    ).copy()
                )

            for _ in range(
                PROMPT_GAP_STEPS
            ):

                fired = brain.step()

                states.append(
                    np.asarray(
                        trace.observe(
                            fired
                        )
                    ).copy()
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
        token: str,
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

            fired = brain.step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ]
            )

            states.append(
                np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()
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

            fired = brain.step()

            delayed_state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()

        return (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        )

    # ========================================================
    # Frozen adapter prediction
    # ========================================================

    @torch.no_grad()
    def predict(
        feature: np.ndarray,
        position: int,
    ):

        standardized = (
            (
                feature
                -
                feature_mean
            )
            /
            feature_std
        ).astype(
            np.float32,
            copy=False,
        )

        X = torch.from_numpy(
            standardized
        ).reshape(
            1,
            -1,
        ).to(
            device
        )

        position_tensor = torch.tensor(
            [
                position
            ],
            dtype=torch.long,
            device=device,
        )

        logits = model(
            X,
            position_tensor,
        )[
            0
        ]

        probabilities = torch.softmax(
            logits,
            dim=0,
        )

        predicted_id = int(
            torch.argmax(
                probabilities
            ).item()
        )

        top_k = min(
            3,
            len(
                token_classes
            ),
        )

        top_probabilities, top_ids = (
            torch.topk(
                probabilities,
                k=top_k,
            )
        )

        top_ids_list = [
            int(
                value
            )

            for value
            in top_ids.cpu().tolist()
        ]

        top_predictions = [
            {
                "token": (
                    token_classes[
                        token_id
                    ]
                ),

                "probability": float(
                    probability
                ),
            }

            for probability, token_id
            in zip(
                top_probabilities.cpu().tolist(),
                top_ids_list,
            )
        ]

        return {
            "predicted_id": (
                predicted_id
            ),

            "predicted_token": (
                token_classes[
                    predicted_id
                ]
            ),

            "top_ids": (
                top_ids_list
            ),

            "top3": (
                top_predictions
            ),
        }

    # ========================================================
    # A) FINAL TEACHER-FORCED EVALUATION
    # ========================================================

    teacher_rows = []

    total_teacher_episodes = (
        len(
            examples
        )
        *
        len(
            NEURAL_SEEDS
        )
    )

    teacher_episode_index = 0

    print()

    print(
        "===================================================="
    )

    print(
        "       FINAL A - TEACHER-FORCED EVALUATION"
    )

    print(
        "===================================================="
    )

    print()

    for example in sorted(
        examples,
        key=lambda item:
            item.id,
    ):

        target_tokens = (
            target_by_example[
                example.id
            ]
        )

        for neural_seed in (
            NEURAL_SEEDS
        ):

            teacher_episode_index += 1

            brain.reset(
                seed=neural_seed
            )

            trace.reset()

            warmup()

            prompt_summary = (
                stimulate_prompt(
                    list(
                        example.prompt_tokens
                    )
                )
            )

            correct_count = 0

            for position, target_token in enumerate(
                target_tokens
            ):

                feedback_token = (
                    "BOS"
                    if position == 0
                    else
                    target_tokens[
                        position - 1
                    ]
                )

                feedback_summary = (
                    stimulate_feedback(
                        feedback_token
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

                if (
                    feature.shape
                    !=
                    (
                        EXPECTED_FEATURE_DIM,
                    )
                ):

                    raise RuntimeError(
                        "Unexpected teacher-forced "
                        f"feature shape: {feature.shape}"
                    )

                prediction = predict(
                    feature,
                    position,
                )

                predicted_token = (
                    prediction[
                        "predicted_token"
                    ]
                )

                correct = (
                    predicted_token
                    ==
                    target_token
                )

                correct_count += int(
                    correct
                )

                target_id = (
                    token_to_id[
                        target_token
                    ]
                )

                teacher_rows.append(
                    {
                        "example_id": (
                            example.id
                        ),

                        "intent": str(
                            example.intent
                        ),

                        "neural_seed": int(
                            neural_seed
                        ),

                        "position": int(
                            position
                        ),

                        "feedback_token": (
                            feedback_token
                        ),

                        "target_token": (
                            target_token
                        ),

                        "target_id": int(
                            target_id
                        ),

                        "prediction": (
                            predicted_token
                        ),

                        "predicted_id": int(
                            prediction[
                                "predicted_id"
                            ]
                        ),

                        "top_ids": (
                            prediction[
                                "top_ids"
                            ]
                        ),

                        "top3": (
                            prediction[
                                "top3"
                            ]
                        ),

                        "correct": bool(
                            correct
                        ),
                    }
                )

            print(
                f"[{teacher_episode_index:03}/"
                f"{total_teacher_episodes:03}] "
                f"{example.id:24} "
                f"seed={neural_seed} "
                f"Top-1="
                f"{correct_count / len(target_tokens) * 100:6.2f}%"
            )

    # --------------------------------------------------------
    # Teacher-forced aggregate metrics
    # --------------------------------------------------------

    teacher_true_tokens = [
        row[
            "target_token"
        ]

        for row
        in teacher_rows
    ]

    teacher_pred_tokens = [
        row[
            "prediction"
        ]

        for row
        in teacher_rows
    ]

    teacher_positions = np.asarray(
        [
            row[
                "position"
            ]

            for row
            in teacher_rows
        ],
        dtype=np.int32,
    )

    teacher_correct = np.asarray(
        [
            row[
                "correct"
            ]

            for row
            in teacher_rows
        ],
        dtype=bool,
    )

    teacher_top2 = np.asarray(
        [
            row[
                "target_id"
            ]
            in
            row[
                "top_ids"
            ][
                :2
            ]

            for row
            in teacher_rows
        ],
        dtype=bool,
    )

    teacher_top3 = np.asarray(
        [
            row[
                "target_id"
            ]
            in
            row[
                "top_ids"
            ][
                :3
            ]

            for row
            in teacher_rows
        ],
        dtype=bool,
    )

    semantic_mask = np.isin(
        teacher_positions,
        SEMANTIC_POSITIONS,
    )

    teacher_metrics = {
        "rows": int(
            len(
                teacher_rows
            )
        ),

        "top1_accuracy": float(
            np.mean(
                teacher_correct
            )
        ),

        "top2_accuracy": float(
            np.mean(
                teacher_top2
            )
        ),

        "top3_accuracy": float(
            np.mean(
                teacher_top3
            )
        ),

        "macro_f1": (
            macro_f1_from_labels(
                teacher_true_tokens,
                teacher_pred_tokens,
                token_classes,
            )
        ),

        "semantic_accuracy": float(
            np.mean(
                teacher_correct[
                    semantic_mask
                ]
            )
        ),

        "semantic_macro_f1": (
            macro_f1_from_labels(
                np.asarray(
                    teacher_true_tokens
                )[
                    semantic_mask
                ].tolist(),

                np.asarray(
                    teacher_pred_tokens
                )[
                    semantic_mask
                ].tolist(),

                token_classes,
            )
        ),
    }

    teacher_by_position = {}

    for position in sorted(
        np.unique(
            teacher_positions
        )
    ):

        mask = (
            teacher_positions
            ==
            position
        )

        teacher_by_position[
            str(
                int(
                    position
                )
            )
        ] = {
            "rows": int(
                np.sum(
                    mask
                )
            ),

            "top1_accuracy": float(
                np.mean(
                    teacher_correct[
                        mask
                    ]
                )
            ),
        }

    teacher_metrics[
        "accuracy_by_position"
    ] = (
        teacher_by_position
    )

    teacher_by_seed = {}

    for seed in NEURAL_SEEDS:

        rows = [
            row

            for row
            in teacher_rows

            if row[
                "neural_seed"
            ]
            ==
            seed
        ]

        teacher_by_seed[
            str(
                seed
            )
        ] = {
            "rows": int(
                len(
                    rows
                )
            ),

            "top1_accuracy": safe_mean(
                row[
                    "correct"
                ]

                for row
                in rows
            ),
        }

    teacher_metrics[
        "per_seed"
    ] = teacher_by_seed

    teacher_by_intent = {}

    for intent in sorted(
        intent_counts
    ):

        rows = [
            row

            for row
            in teacher_rows

            if row[
                "intent"
            ]
            ==
            intent
        ]

        teacher_by_intent[
            intent
        ] = {
            "rows": int(
                len(
                    rows
                )
            ),

            "top1_accuracy": safe_mean(
                row[
                    "correct"
                ]

                for row
                in rows
            ),
        }

    teacher_metrics[
        "per_intent"
    ] = teacher_by_intent

    # ========================================================
    # B) FINAL TRUE CLOSED-LOOP EVALUATION
    # ========================================================

    closed_rows = []

    total_closed_episodes = (
        len(
            examples
        )
        *
        len(
            NEURAL_SEEDS
        )
    )

    closed_episode_index = 0

    print()

    print(
        "===================================================="
    )

    print(
        "       FINAL B - TRUE CLOSED-LOOP GENERATION"
    )

    print(
        "===================================================="
    )

    print()

    for example in sorted(
        examples,
        key=lambda item:
            item.id,
    ):

        target_tokens = (
            target_by_example[
                example.id
            ]
        )

        prompt_tokens = list(
            example.prompt_tokens
        )

        for neural_seed in (
            NEURAL_SEEDS
        ):

            closed_episode_index += 1

            brain.reset(
                seed=neural_seed
            )

            trace.reset()

            warmup()

            prompt_summary = (
                stimulate_prompt(
                    prompt_tokens
                )
            )

            generated_tokens = []
            generation_trace = []

            feedback_token = "BOS"

            for position in range(
                max_generation_steps
            ):

                feedback_summary = (
                    stimulate_feedback(
                        feedback_token
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

                if (
                    feature.shape
                    !=
                    (
                        EXPECTED_FEATURE_DIM,
                    )
                ):

                    raise RuntimeError(
                        "Unexpected closed-loop "
                        f"feature shape: {feature.shape}"
                    )

                prediction = predict(
                    feature,
                    position,
                )

                predicted_token = (
                    prediction[
                        "predicted_token"
                    ]
                )

                generated_tokens.append(
                    predicted_token
                )

                generation_trace.append(
                    {
                        "position": int(
                            position
                        ),

                        "feedback_token": (
                            feedback_token
                        ),

                        "prediction": (
                            predicted_token
                        ),

                        "top3": (
                            prediction[
                                "top3"
                            ]
                        ),
                    }
                )

                if (
                    predicted_token
                    ==
                    "EOS"
                ):

                    break

                feedback_token = (
                    predicted_token
                )

            token_accuracy = (
                sequence_token_accuracy(
                    target_tokens,
                    generated_tokens,
                )
            )

            exact_program = bool(
                generated_tokens
                ==
                target_tokens
            )

            (
                generated_code,
                valid_program,
                syntax_valid,
            ) = decode_tokens(
                generated_tokens
            )

            eos_completed = bool(
                generated_tokens
                and
                generated_tokens[
                    -1
                ]
                ==
                "EOS"
            )

            first_error = (
                first_error_position(
                    target_tokens,
                    generated_tokens,
                )
            )

            print(
                f"[{closed_episode_index:03}/"
                f"{total_closed_episodes:03}] "
                f"{example.id:24} "
                f"seed={neural_seed} "
                f"token="
                f"{token_accuracy * 100:6.2f}% "
                f"exact="
                f"{str(exact_program):5} "
                f"valid="
                f"{str(valid_program):5}"
            )

            closed_rows.append(
                {
                    "example_id": (
                        example.id
                    ),

                    "intent": str(
                        example.intent
                    ),

                    "neural_seed": int(
                        neural_seed
                    ),

                    "prompt_tokens": (
                        prompt_tokens
                    ),

                    "target_tokens": (
                        target_tokens
                    ),

                    "target_code": (
                        example.code
                    ),

                    "generated_tokens": (
                        generated_tokens
                    ),

                    "generated_code": (
                        generated_code
                    ),

                    "token_accuracy": float(
                        token_accuracy
                    ),

                    "exact_program": (
                        exact_program
                    ),

                    "valid_program": bool(
                        valid_program
                    ),

                    "syntax_valid": bool(
                        syntax_valid
                    ),

                    "eos_completed": (
                        eos_completed
                    ),

                    "first_error_position": (
                        first_error
                    ),

                    "generation_trace": (
                        generation_trace
                    ),
                }
            )

    closed_metrics = {
        "episodes": int(
            len(
                closed_rows
            )
        ),

        "mean_token_accuracy": safe_mean(
            row[
                "token_accuracy"
            ]

            for row
            in closed_rows
        ),

        "exact_program_accuracy": safe_mean(
            row[
                "exact_program"
            ]

            for row
            in closed_rows
        ),

        "valid_program_rate": safe_mean(
            row[
                "valid_program"
            ]

            for row
            in closed_rows
        ),

        "syntax_valid_rate": safe_mean(
            row[
                "syntax_valid"
            ]

            for row
            in closed_rows
        ),

        "eos_completion_rate": safe_mean(
            row[
                "eos_completed"
            ]

            for row
            in closed_rows
        ),
    }

    # --------------------------------------------------------
    # Closed-loop per-seed
    # --------------------------------------------------------

    closed_per_seed = {}

    for seed in NEURAL_SEEDS:

        rows = [
            row

            for row
            in closed_rows

            if row[
                "neural_seed"
            ]
            ==
            seed
        ]

        closed_per_seed[
            str(
                seed
            )
        ] = {
            "episodes": int(
                len(
                    rows
                )
            ),

            "mean_token_accuracy": safe_mean(
                row[
                    "token_accuracy"
                ]

                for row
                in rows
            ),

            "exact_program_accuracy": safe_mean(
                row[
                    "exact_program"
                ]

                for row
                in rows
            ),

            "valid_program_rate": safe_mean(
                row[
                    "valid_program"
                ]

                for row
                in rows
            ),
        }

    closed_metrics[
        "per_seed"
    ] = closed_per_seed

    # --------------------------------------------------------
    # Closed-loop per-intent
    # --------------------------------------------------------

    closed_per_intent = {}

    for intent in sorted(
        intent_counts
    ):

        rows = [
            row

            for row
            in closed_rows

            if row[
                "intent"
            ]
            ==
            intent
        ]

        closed_per_intent[
            intent
        ] = {
            "episodes": int(
                len(
                    rows
                )
            ),

            "mean_token_accuracy": safe_mean(
                row[
                    "token_accuracy"
                ]

                for row
                in rows
            ),

            "exact_program_accuracy": safe_mean(
                row[
                    "exact_program"
                ]

                for row
                in rows
            ),

            "valid_program_rate": safe_mean(
                row[
                    "valid_program"
                ]

                for row
                in rows
            ),
        }

    closed_metrics[
        "per_intent"
    ] = closed_per_intent

    # --------------------------------------------------------
    # First error position + common sequences
    # --------------------------------------------------------

    first_errors = Counter()

    common_sequences = Counter()

    for row in closed_rows:

        if (
            row[
                "first_error_position"
            ]
            is None
        ):

            first_errors[
                "exact"
            ] += 1

        else:

            first_errors[
                f"P{row['first_error_position']}"
            ] += 1

        common_sequences[
            tuple(
                row[
                    "generated_tokens"
                ]
            )
        ] += 1

    closed_metrics[
        "first_error_position"
    ] = dict(
        first_errors
    )

    closed_metrics[
        "most_common_sequences"
    ] = [
        {
            "tokens": list(
                sequence
            ),

            "count": int(
                count
            ),
        }

        for sequence, count
        in common_sequences.most_common(
            10
        )
    ]

    # ========================================================
    # Save final prediction files
    # ========================================================

    FINAL_METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TEACHER_PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in teacher_rows:

            file.write(
                json.dumps(
                    row
                )
            )

            file.write(
                "\n"
            )

    with CLOSED_LOOP_PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in closed_rows:

            file.write(
                json.dumps(
                    row
                )
            )

            file.write(
                "\n"
            )

    # ========================================================
    # Development results are included ONLY for context
    # ========================================================

    step80 = load_json(
        STEP80_METRICS_FILE
    )

    step81 = load_json(
        STEP81_METRICS_FILE
    )

    development_context = {
        "teacher_forced_development_test": (
            step80.get(
                "development_test"
            )
        ),

        "closed_loop_development_test": (
            step81.get(
                "aggregate"
            )
        ),
    }

    # ========================================================
    # Save one immutable final report
    # ========================================================

    final_report = {
        "phase": 3,

        "step": "83",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "PHASE3_FINAL_EVALUATION_COMPLETE"
        ),

        "protocol_status": (
            protocol.get(
                "status"
            )
        ),

        "training_performed": (
            False
        ),

        "hyperparameter_tuning_performed": (
            False
        ),

        "holdout": {
            "file": (
                relative(
                    FINAL_HOLDOUT_FILE
                )
            ),

            "sha256": (
                sha256_file(
                    FINAL_HOLDOUT_FILE
                )
            ),

            "logical_prompts": (
                len(
                    examples
                )
            ),

            "intents": (
                dict(
                    intent_counts
                )
            ),

            "neural_seeds": (
                NEURAL_SEEDS
            ),
        },

        "teacher_forced_final": (
            teacher_metrics
        ),

        "closed_loop_final": (
            closed_metrics
        ),

        "development_context_not_final": (
            development_context
        ),

        "frozen_artifacts": {
            "model_sha256": (
                sha256_file(
                    MODEL_FILE
                )
            ),

            "scaler_sha256": (
                sha256_file(
                    SCALER_FILE
                )
            ),

            "mapping_sha256": (
                sha256_file(
                    MAPPING_FILE
                )
            ),
        },

        "artifacts": {
            "teacher_forced_predictions": (
                relative(
                    TEACHER_PREDICTIONS_FILE
                )
            ),

            "closed_loop_predictions": (
                relative(
                    CLOSED_LOOP_PREDICTIONS_FILE
                )
            ),
        },

        "scientific_rule": (
            "These are the one-time Phase-3 final holdout "
            "results for the frozen Step-80/81 pipeline. "
            "Do not retroactively tune the reported model "
            "using these final results."
        ),
    }

    FINAL_METRICS_FILE.write_text(
        json.dumps(
            final_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final terminal report
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "          PHASE 3 FINAL HOLDOUT RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "FINAL TEACHER-FORCED:"
    )

    print(
        f"  Rows:                "
        f"{teacher_metrics['rows']:,}"
    )

    print(
        f"  Top-1:               "
        f"{teacher_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Top-2:               "
        f"{teacher_metrics['top2_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Top-3:               "
        f"{teacher_metrics['top3_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:            "
        f"{teacher_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy:   "
        f"{teacher_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "FINAL TRUE CLOSED-LOOP:"
    )

    print(
        f"  Episodes:            "
        f"{closed_metrics['episodes']}"
    )

    print(
        f"  Mean token accuracy: "
        f"{closed_metrics['mean_token_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Exact programs:      "
        f"{closed_metrics['exact_program_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Valid programs:      "
        f"{closed_metrics['valid_program_rate'] * 100:.2f}%"
    )

    print(
        f"  Syntax-valid:        "
        f"{closed_metrics['syntax_valid_rate'] * 100:.2f}%"
    )

    print(
        f"  EOS completion:      "
        f"{closed_metrics['eos_completion_rate'] * 100:.2f}%"
    )

    print()

    print(
        "Closed-loop per neural seed:"
    )

    for seed in NEURAL_SEEDS:

        row = closed_per_seed[
            str(
                seed
            )
        ]

        print(
            f"  seed={seed}: "
            f"token="
            f"{row['mean_token_accuracy'] * 100:6.2f}% "
            f"exact="
            f"{row['exact_program_accuracy'] * 100:6.2f}% "
            f"valid="
            f"{row['valid_program_rate'] * 100:6.2f}%"
        )

    print()

    print(
        "First error positions:"
    )

    for key, count in (
        first_errors.most_common()
    ):

        print(
            f"  {key:6} "
            f"{count:3}/"
            f"{len(closed_rows)}"
        )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{relative(FINAL_METRICS_FILE)}"
    )

    print(
        "  "
        f"{relative(TEACHER_PREDICTIONS_FILE)}"
    )

    print(
        "  "
        f"{relative(CLOSED_LOOP_PREDICTIONS_FILE)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 83 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Phase 3 final evaluation is complete."
    )

    print()

    print(
        "Do NOT tune the reported Step-80 model using "
        "these final holdout results."
    )


if __name__ == "__main__":
    main()
