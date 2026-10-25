from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl
from pyfly.tokenizer import tokens_to_code


# ============================================================
# Phase 3 — Trainable PyFly
# Step 81 — TRUE CLOSED-LOOP GENERATION
#
# Uses the frozen Step-80 PyTorch adapter.
#
# No fitting.
# No gradient updates.
# No hyperparameter tuning.
#
# Loop:
#
#   prompt
#      ↓
#   MaleCNS
#      ↓
#   Step-80 adapter
#      ↓
#   predicted token
#      ↓
#   feed predicted token BACK into MaleCNS
#      ↓
#   next prediction
#
# Dataset:
#   generator_test ONLY
#
# This is a development closed-loop diagnostic.
# A fresh Phase-3 holdout should be created after this.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_NAME = "cuda"


# ============================================================
# Development-test prompts
# ============================================================

DEV_TEST_GENERATOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_test.jsonl"
)


# ============================================================
# Frozen Step-80 model
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

STEP80_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_pyfly_adapter_training.json"
)


# ============================================================
# Frozen PyFly mapping
# ============================================================

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_closed_loop_development_test.json"
)

OUTPUT_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_closed_loop_development_test_predictions.jsonl"
)


# ============================================================
# Frozen Step-62 MaleCNS stimulation settings
# ============================================================

PROMPT_TOKEN_STEPS = 8
PROMPT_GAP_STEPS = 2

FEEDBACK_TOKEN_STEPS = 8
FEEDBACK_GAP_STEPS = 1

WARMUP_STEPS = 10

AMPLITUDE = 0.8
TRACE_TAU = 0.25

EXPECTED_FEATURE_DIM = 41_111
EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

NEURAL_SEEDS = [
    0,
    1,
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
            key=lambda row:
                row.target_position
        )

    return grouped


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
# Load frozen dual-channel neural map
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
        "       STEP 81 - TRUE CLOSED-LOOP GENERATION"
    )

    print(
        "===================================================="
    )

    print()

    required = [
        DEV_TEST_GENERATOR_FILE,
        MODEL_FILE,
        SCALER_FILE,
        STEP80_METRICS_FILE,
        MAPPING_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: "
                f"{path}"
            )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available to PyTorch."
        )

    device = torch.device(
        DEVICE_NAME
    )

    step80 = load_json(
        STEP80_METRICS_FILE
    )

    print(
        f"Step-80 best epoch:        "
        f"{step80.get('best_epoch')}"
    )

    print(
        "Model training in Step 81: NO"
    )

    print(
        "Dataset:                   generator_test"
    )

    print(
        "Mode:                      true closed loop"
    )

    # ========================================================
    # Load Step-80 checkpoint
    # ========================================================

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    token_classes = list(
        checkpoint[
            "token_classes"
        ]
    )

    token_to_id = {
        token: index

        for index, token
        in enumerate(
            token_classes
        )
    }

    model = PyFlyNeuralAdapter(
        input_dim=int(
            checkpoint[
                "input_dim"
            ]
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
        f"GPU:                       "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Token classes:             "
        f"{len(token_classes)}"
    )

    print(
        f"Maximum generation steps:  "
        f"{max_generation_steps}"
    )

    # ========================================================
    # Load scaler
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
            "Step-80 scaler feature dimensions changed."
        )

    # ========================================================
    # Development-test logical prompts
    # ========================================================

    samples = load_generator_jsonl(
        DEV_TEST_GENERATOR_FILE
    )

    grouped = group_by_example(
        samples
    )

    print()

    print(
        f"Logical development-test prompts: "
        f"{len(grouped)}"
    )

    # ========================================================
    # Frozen PyFly map + real MaleCNS
    # ========================================================

    dual_map = load_dual_map(
        MAPPING_FILE
    )

    print()

    print(
        "Loading real MaleCNS on CUDA..."
    )

    brain = FlyBrain(
        device="cuda",
        seed=0,
    )

    if brain.n != EXPECTED_NEURONS:

        raise ValueError(
            "Unexpected MaleCNS neuron count."
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

    if (
        len(
            input_neurons
        )
        !=
        EXPECTED_DIRECT_INPUT_NEURONS
    ):

        raise ValueError(
            "Direct-input population changed."
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
            "Feature dimensions changed."
        )

    print(
        f"Direct input neurons:       "
        f"{len(input_neurons):,}"
    )

    print(
        f"Indirect readout neurons:   "
        f"{len(readout_population):,}"
    )

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    # ========================================================
    # Live MaleCNS stimulation
    # ========================================================

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
        token,
    ):

        if (
            token
            not in
            dual_map.feedback_mapping
        ):

            raise KeyError(
                f"No neural feedback mapping "
                f"for predicted token {token!r}"
            )

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
    # Step-80 adapter prediction
    # ========================================================

    @torch.no_grad()
    def predict_token(
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
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[
            0
        ]

        predicted_id = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_token = (
            token_classes[
                predicted_id
            ]
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

        top_predictions = [
            {
                "token": (
                    token_classes[
                        int(
                            token_id
                        )
                    ]
                ),
                "probability": float(
                    probability
                ),
            }

            for probability, token_id
            in zip(
                top_probabilities.cpu().tolist(),
                top_ids.cpu().tolist(),
            )
        ]

        return (
            predicted_token,
            top_predictions,
        )

    # ========================================================
    # Generate one program closed loop
    # ========================================================

    def generate(
        prompt_tokens,
        neural_seed: int,
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

        prompt_summary = stimulate_prompt(
            prompt_tokens
        )

        feedback_token = "BOS"

        generated_tokens = []
        generation_trace = []

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
                    f"Unexpected live feature shape: "
                    f"{feature.shape}"
                )

            (
                predicted_token,
                top_predictions,
            ) = predict_token(
                feature,
                position,
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
                        top_predictions
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

        return (
            generated_tokens,
            generation_trace,
        )

    # ========================================================
    # Run all 6 prompts × 5 neural seeds
    # ========================================================

    total_episodes = (
        len(
            grouped
        )
        *
        len(
            NEURAL_SEEDS
        )
    )

    episode_index = 0

    results = []

    print()

    print(
        "===================================================="
    )

    print(
        "       STEP 81 CLOSED-LOOP EPISODES"
    )

    print(
        "===================================================="
    )

    print()

    for example_id in sorted(
        grouped
    ):

        example_samples = (
            grouped[
                example_id
            ]
        )

        prompt_tokens = list(
            example_samples[
                0
            ].prompt_tokens
        )

        target_tokens = [
            sample.target_token

            for sample
            in example_samples
        ]

        intent = getattr(
            example_samples[
                0
            ],
            "intent",
            "unknown",
        )

        for neural_seed in (
            NEURAL_SEEDS
        ):

            episode_index += 1

            (
                generated_tokens,
                generation_trace,
            ) = generate(
                prompt_tokens,
                neural_seed,
            )

            token_accuracy = (
                sequence_token_accuracy(
                    target_tokens,
                    generated_tokens,
                )
            )

            exact_program = (
                generated_tokens
                ==
                target_tokens
            )

            (
                generated_code,
                template_valid,
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
                f"[{episode_index:02}/"
                f"{total_episodes:02}] "
                f"{example_id:18} "
                f"seed={neural_seed} "
                f"token="
                f"{token_accuracy * 100:6.2f}% "
                f"exact="
                f"{str(exact_program):5} "
                f"EOS="
                f"{str(eos_completed):5}"
            )

            results.append(
                {
                    "example_id": (
                        example_id
                    ),

                    "intent": (
                        intent
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

                    "generated_tokens": (
                        generated_tokens
                    ),

                    "generated_code": (
                        generated_code
                    ),

                    "token_accuracy": float(
                        token_accuracy
                    ),

                    "exact_program": bool(
                        exact_program
                    ),

                    "valid_program": bool(
                        template_valid
                    ),

                    "syntax_valid": bool(
                        syntax_valid
                    ),

                    "eos_completed": bool(
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

    # ========================================================
    # Aggregate metrics
    # ========================================================

    aggregate = {
        "episodes": int(
            len(
                results
            )
        ),

        "mean_token_accuracy": float(
            np.mean(
                [
                    row[
                        "token_accuracy"
                    ]

                    for row in results
                ]
            )
        ),

        "exact_program_accuracy": float(
            np.mean(
                [
                    row[
                        "exact_program"
                    ]

                    for row in results
                ]
            )
        ),

        "valid_program_rate": float(
            np.mean(
                [
                    row[
                        "valid_program"
                    ]

                    for row in results
                ]
            )
        ),

        "syntax_valid_rate": float(
            np.mean(
                [
                    row[
                        "syntax_valid"
                    ]

                    for row in results
                ]
            )
        ),

        "eos_completion_rate": float(
            np.mean(
                [
                    row[
                        "eos_completed"
                    ]

                    for row in results
                ]
            )
        ),
    }

    # ========================================================
    # Per-seed metrics
    # ========================================================

    per_seed = {}

    for seed in NEURAL_SEEDS:

        rows = [
            row

            for row in results

            if row[
                "neural_seed"
            ]
            ==
            seed
        ]

        per_seed[
            str(
                seed
            )
        ] = {
            "mean_token_accuracy": float(
                np.mean(
                    [
                        row[
                            "token_accuracy"
                        ]

                        for row in rows
                    ]
                )
            ),

            "exact_program_accuracy": float(
                np.mean(
                    [
                        row[
                            "exact_program"
                        ]

                        for row in rows
                    ]
                )
            ),

            "valid_program_rate": float(
                np.mean(
                    [
                        row[
                            "valid_program"
                        ]

                        for row in rows
                    ]
                )
            ),
        }

    # ========================================================
    # First errors + common sequences
    # ========================================================

    first_errors = Counter()

    common_sequences = Counter()

    for row in results:

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

    # ========================================================
    # Save prediction JSONL
    # ========================================================

    OUTPUT_PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in results:

            file.write(
                json.dumps(
                    row
                )
            )

            file.write(
                "\n"
            )

    # ========================================================
    # Save metrics
    # ========================================================

    output = {
        "phase": 3,

        "step": "81",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "DEVELOPMENT_CLOSED_LOOP_COMPLETE"
        ),

        "dataset": (
            "generator_test"
        ),

        "model_training_performed": (
            False
        ),

        "aggregate": (
            aggregate
        ),

        "per_seed": (
            per_seed
        ),

        "first_error_position": (
            dict(
                first_errors
            )
        ),

        "most_common_sequences": [
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
        ],

        "step80_development_test_teacher_forced": (
            step80.get(
                "development_test"
            )
        ),

        "artifacts": {
            "predictions": (
                relative(
                    OUTPUT_PREDICTIONS_FILE
                )
            ),

            "model": (
                relative(
                    MODEL_FILE
                )
            ),

            "scaler": (
                relative(
                    SCALER_FILE
                )
            ),
        },

        "next": (
            "Create a fresh untouched Phase-3 holdout "
            "after the closed-loop pipeline is frozen."
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

    print()

    print(
        "===================================================="
    )

    print(
        "          STEP 81 CLOSED-LOOP RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Generation episodes:      "
        f"{aggregate['episodes']}"
    )

    print(
        f"Mean token accuracy:      "
        f"{aggregate['mean_token_accuracy'] * 100:.2f}%"
    )

    print(
        f"Exact-program accuracy:   "
        f"{aggregate['exact_program_accuracy'] * 100:.2f}%"
    )

    print(
        f"Valid-program rate:       "
        f"{aggregate['valid_program_rate'] * 100:.2f}%"
    )

    print(
        f"Syntax-valid rate:        "
        f"{aggregate['syntax_valid_rate'] * 100:.2f}%"
    )

    print(
        f"EOS completion rate:      "
        f"{aggregate['eos_completion_rate'] * 100:.2f}%"
    )

    print()

    print(
        "Per neural seed:"
    )

    for seed in NEURAL_SEEDS:

        row = per_seed[
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
            f"{len(results)}"
        )

    print()

    print(
        "Most common generated sequences:"
    )

    for sequence, count in (
        common_sequences.most_common(
            10
        )
    ):

        print(
            f"  {count:3}x "
            f"{list(sequence)}"
        )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{relative(OUTPUT_METRICS_FILE)}"
    )

    print(
        "  "
        f"{relative(OUTPUT_PREDICTIONS_FILE)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 81 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Next: Step 82 - freeze this generation "
        "pipeline and create a NEW untouched "
        "Phase-3 holdout for final evaluation."
    )


if __name__ == "__main__":
    main()
