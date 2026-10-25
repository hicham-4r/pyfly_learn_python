from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import DualChannelMap
from pyfly.tokenizer import code_to_tokens, tokens_to_code


# ============================================================
# Step 77E
# FINAL CLOSED-LOOP GENERATION
#
# Use the frozen Step-62 architecture and the Step-77B model
# on the untouched Step-77A holdout.
#
# IMPORTANT:
#   * No fitting.
#   * No architecture search.
#   * No hyperparameter search.
#   * No beam search / top-k rescue.
#   * Greedy top-1 predictions are fed back into MaleCNS.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

REFIT_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_decoder_refit_manifest.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "generator_final_frozen_decoder.joblib"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_closed_loop_metrics.json"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_closed_loop_predictions.jsonl"
)


# ============================================================
# Frozen Step-62 configuration
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

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

MAX_GENERATION_STEPS = 10

EXPECTED_LOGICAL_PROMPTS = 24
EXPECTED_EPISODES = 120

EXPECTED_DIRECT_INPUT_NEURONS = 3328
EXPECTED_READOUT_NEURONS = 5873
EXPECTED_FEATURE_DIMENSIONS = 41111

EXPECTED_FAMILY = "Balanced Logistic"
EXPECTED_CONFIGURATION = (
    "Balanced Logistic PCA=100 C=0.1"
)

EXPECTED_FEATURE_BLOCKS = [
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
]


# ============================================================
# Development closed-loop record
#
# CONTEXT ONLY. This was repeatedly inspected during
# development and is NOT a final benchmark.
# ============================================================

DEVELOPMENT_CLOSED_LOOP = {
    "mean_token_accuracy": 0.6390,
    "exact_program_accuracy": 0.1667,
    "valid_program_rate": 0.3333,
    "eos_completion_rate": 1.0,
}


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_frozen_hash(
    protocol: dict,
    normalized_suffix: str,
) -> str:
    frozen = protocol.get(
        "frozen_file_hashes",
        {},
    )

    normalized_suffix = (
        normalized_suffix
        .replace("\\", "/")
    )

    for stored_path, info in frozen.items():
        normalized_path = (
            stored_path
            .replace("\\", "/")
        )

        if normalized_path.endswith(
            normalized_suffix
        ):
            value = info.get("sha256")

            if value:
                return str(value)

    raise ValueError(
        "Frozen protocol is missing SHA-256 for "
        f"{normalized_suffix}"
    )


# ============================================================
# Verify frozen protocol + model
# ============================================================

def verify_integrity() -> tuple[dict, dict]:

    required_files = [
        FINAL_HOLDOUT_FILE,
        PROTOCOL_FILE,
        REFIT_MANIFEST_FILE,
        MODEL_FILE,
        MAPPING_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required frozen file: {path}"
            )

    protocol = load_json(
        PROTOCOL_FILE
    )

    refit_manifest = load_json(
        REFIT_MANIFEST_FILE
    )

    # --------------------------------------------------------
    # Protocol must still be frozen
    # --------------------------------------------------------

    if (
        protocol.get("status")
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):
        raise ValueError(
            "Step-77A protocol is not marked frozen."
        )

    architecture = protocol.get(
        "architecture",
        {},
    )

    decoder_selection = protocol.get(
        "decoder_selection",
        {},
    )

    if (
        architecture.get("recurrent_weights")
        !=
        "frozen"
    ):
        raise ValueError(
            "MaleCNS recurrent weights are no longer "
            "marked frozen in the protocol."
        )

    if (
        architecture.get("feature_blocks")
        !=
        EXPECTED_FEATURE_BLOCKS
    ):
        raise ValueError(
            "Frozen Step-62 feature layout changed."
        )

    if (
        int(
            architecture.get(
                "max_prompt_positions",
                -1,
            )
        )
        !=
        MAX_PROMPT_POSITIONS
    ):
        raise ValueError(
            "Frozen max prompt positions changed."
        )

    if (
        int(
            architecture.get(
                "prompt_neurons_per_channel",
                -1,
            )
        )
        !=
        PROMPT_NEURONS_PER_CHANNEL
    ):
        raise ValueError(
            "Frozen prompt channel width changed."
        )

    if (
        int(
            architecture.get(
                "feedback_neurons_per_token",
                -1,
            )
        )
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):
        raise ValueError(
            "Frozen feedback channel width changed."
        )

    if (
        int(
            architecture.get(
                "prompt_token_steps",
                -1,
            )
        )
        !=
        PROMPT_TOKEN_STEPS
    ):
        raise ValueError(
            "Frozen prompt token timing changed."
        )

    if (
        int(
            architecture.get(
                "prompt_gap_steps",
                -1,
            )
        )
        !=
        PROMPT_GAP_STEPS
    ):
        raise ValueError(
            "Frozen prompt gap timing changed."
        )

    if (
        int(
            architecture.get(
                "feedback_token_steps",
                -1,
            )
        )
        !=
        FEEDBACK_TOKEN_STEPS
    ):
        raise ValueError(
            "Frozen feedback timing changed."
        )

    if (
        int(
            architecture.get(
                "feedback_gap_steps",
                -1,
            )
        )
        !=
        FEEDBACK_GAP_STEPS
    ):
        raise ValueError(
            "Frozen feedback gap changed."
        )

    if (
        int(
            architecture.get(
                "warmup_steps",
                -1,
            )
        )
        !=
        WARMUP_STEPS
    ):
        raise ValueError(
            "Frozen warm-up changed."
        )

    if not np.isclose(
        float(
            architecture.get(
                "amplitude",
                np.nan,
            )
        ),
        AMPLITUDE,
    ):
        raise ValueError(
            "Frozen injection amplitude changed."
        )

    if not np.isclose(
        float(
            architecture.get(
                "trace_tau",
                np.nan,
            )
        ),
        TRACE_TAU,
    ):
        raise ValueError(
            "Frozen trace tau changed."
        )

    if (
        int(
            architecture.get(
                "mapping_seed",
                -1,
            )
        )
        !=
        MAP_SEED
    ):
        raise ValueError(
            "Frozen mapping seed changed."
        )

    protocol_seeds = [
        int(value)
        for value in architecture.get(
            "neural_seeds",
            [],
        )
    ]

    if protocol_seeds != NEURAL_SEEDS:
        raise ValueError(
            "Frozen neural seeds changed: "
            f"{protocol_seeds}"
        )

    if (
        decoder_selection.get(
            "selected_family"
        )
        !=
        EXPECTED_FAMILY
    ):
        raise ValueError(
            "Frozen decoder family changed."
        )

    if (
        decoder_selection.get(
            "selected_configuration"
        )
        !=
        EXPECTED_CONFIGURATION
    ):
        raise ValueError(
            "Frozen decoder configuration changed."
        )

    # --------------------------------------------------------
    # Frozen holdout + mapping hashes
    # --------------------------------------------------------

    expected_holdout_hash = (
        find_frozen_hash(
            protocol,
            "data/raw/final_holdout.jsonl",
        )
    )

    expected_mapping_hash = (
        find_frozen_hash(
            protocol,
            "data/mappings/"
            "generator_dual_channel_map.npz",
        )
    )

    current_holdout_hash = (
        sha256_file(
            FINAL_HOLDOUT_FILE
        )
    )

    current_mapping_hash = (
        sha256_file(
            MAPPING_FILE
        )
    )

    if (
        current_holdout_hash
        !=
        expected_holdout_hash
    ):
        raise ValueError(
            "Final holdout hash changed."
        )

    if (
        current_mapping_hash
        !=
        expected_mapping_hash
    ):
        raise ValueError(
            "Frozen neural mapping hash changed."
        )

    # --------------------------------------------------------
    # Step-77B model integrity
    # --------------------------------------------------------

    if (
        refit_manifest.get(
            "status"
        )
        !=
        "FROZEN_DECODER_REFIT_COMPLETE"
    ):
        raise ValueError(
            "Step-77B refit manifest is not complete."
        )

    if (
        refit_manifest.get(
            "selected_family"
        )
        !=
        EXPECTED_FAMILY
    ):
        raise ValueError(
            "Step-77B decoder family mismatch."
        )

    if (
        refit_manifest.get(
            "selected_configuration"
        )
        !=
        EXPECTED_CONFIGURATION
    ):
        raise ValueError(
            "Step-77B decoder configuration mismatch."
        )

    expected_model_hash = str(
        refit_manifest.get(
            "model_sha256",
            "",
        )
    )

    current_model_hash = (
        sha256_file(
            MODEL_FILE
        )
    )

    if (
        not expected_model_hash
        or
        current_model_hash
        !=
        expected_model_hash
    ):
        raise ValueError(
            "Frozen Step-77B model hash changed."
        )

    expected_protocol_hash = str(
        refit_manifest.get(
            "protocol_sha256",
            "",
        )
    )

    if (
        expected_protocol_hash
        and
        sha256_file(
            PROTOCOL_FILE
        )
        !=
        expected_protocol_hash
    ):
        raise ValueError(
            "Step-77A protocol hash changed after "
            "the frozen decoder was refit."
        )

    return (
        protocol,
        refit_manifest,
    )


# ============================================================
# Load the SAVED Step-62 neural map
#
# Never recreate this mapping from the final holdout.
# ============================================================

def load_frozen_dual_map(
    path: Path,
) -> DualChannelMap:

    data = np.load(
        path,
        allow_pickle=True,
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
            "Frozen neural map is missing arrays: "
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
        ].astype(np.int64)
    )

    prompt_neurons = (
        data[
            "prompt_neurons"
        ].astype(np.int64)
    )

    feedback_tokens = (
        data[
            "feedback_tokens"
        ].astype(str)
    )

    feedback_neurons = (
        data[
            "feedback_neurons"
        ].astype(np.int64)
    )

    prompt_mapping = {}

    for index in range(
        len(prompt_tokens)
    ):
        key = (
            str(
                prompt_tokens[index]
            ),
            int(
                prompt_positions[index]
            ),
        )

        prompt_mapping[
            key
        ] = (
            prompt_neurons[
                index
            ].copy()
        )

    feedback_mapping = {}

    for index in range(
        len(feedback_tokens)
    ):
        feedback_mapping[
            str(
                feedback_tokens[index]
            )
        ] = (
            feedback_neurons[
                index
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
            "Saved prompt mapping width changed."
        )

    if (
        dual_map.feedback_neurons_per_token
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):
        raise ValueError(
            "Saved feedback mapping width changed."
        )

    if (
        dual_map.max_prompt_positions
        !=
        MAX_PROMPT_POSITIONS
    ):
        raise ValueError(
            "Saved max prompt positions changed."
        )

    if (
        dual_map.seed
        !=
        MAP_SEED
    ):
        raise ValueError(
            "Saved neural mapping seed changed."
        )

    return dual_map


# ============================================================
# Token sequence accuracy
# ============================================================

def sequence_token_accuracy(
    target: list[str],
    predicted: list[str],
) -> float:

    denominator = max(
        len(target),
        len(predicted),
    )

    if denominator == 0:
        return 1.0

    correct = 0

    for index in range(
        denominator
    ):
        target_token = (
            target[index]
            if index < len(target)
            else None
        )

        predicted_token = (
            predicted[index]
            if index < len(predicted)
            else None
        )

        if (
            target_token
            ==
            predicted_token
        ):
            correct += 1

    return float(
        correct
        /
        denominator
    )


# ============================================================
# Decode generated token sequence
# ============================================================

def decode_generated_tokens(
    tokens: list[str],
) -> tuple[
    bool,
    bool,
    str | None,
]:

    try:
        code = tokens_to_code(
            tokens
        )

    except ValueError:
        return (
            False,
            False,
            None,
        )

    try:
        ast.parse(
            code
        )
        syntax_valid = True

    except SyntaxError:
        syntax_valid = False

    return (
        True,
        syntax_valid,
        code,
    )


# ============================================================
# First-error position
# ============================================================

def first_error_position(
    target: list[str],
    predicted: list[str],
) -> int | None:

    limit = max(
        len(target),
        len(predicted),
    )

    for index in range(
        limit
    ):
        target_token = (
            target[index]
            if index < len(target)
            else None
        )

        predicted_token = (
            predicted[index]
            if index < len(predicted)
            else None
        )

        if (
            target_token
            !=
            predicted_token
        ):
            return int(index)

    return None


# ============================================================
# Header + frozen-state verification
# ============================================================

print()

print(
    "===================================================="
)

print(
    " PyFly STEP 77E - FINAL CLOSED-LOOP GENERATION"
)

print(
    "===================================================="
)

print()

(
    protocol,
    refit_manifest,
) = verify_integrity()

print(
    "Frozen protocol integrity: PASS"
)

print(
    "Frozen decoder integrity:  PASS"
)

print(
    "Final holdout integrity:   PASS"
)

print(
    "Frozen mapping integrity:  PASS"
)

print()


# ============================================================
# Load final logical prompts
# ============================================================

examples = load_dataset(
    FINAL_HOLDOUT_FILE
)

if (
    len(examples)
    !=
    EXPECTED_LOGICAL_PROMPTS
):
    raise ValueError(
        f"Expected {EXPECTED_LOGICAL_PROMPTS} "
        "final prompts, got "
        f"{len(examples)}"
    )

example_ids = [
    example.id
    for example in examples
]

if (
    len(
        set(example_ids)
    )
    !=
    EXPECTED_LOGICAL_PROMPTS
):
    raise ValueError(
        "Duplicate final holdout IDs."
    )

print(
    f"Final logical prompts:      "
    f"{len(examples)}"
)


# ============================================================
# Load frozen decoder
# ============================================================

model = joblib.load(
    MODEL_FILE
)

required_steps = {
    "scale",
    "pca",
    "classifier",
}

if not required_steps.issubset(
    set(
        model.named_steps
    )
):
    raise ValueError(
        "Frozen decoder pipeline does not "
        "contain scale / pca / classifier."
    )

pca = (
    model.named_steps[
        "pca"
    ]
)

classifier = (
    model.named_steps[
        "classifier"
    ]
)

if (
    int(
        pca.n_components
    )
    !=
    100
):
    raise ValueError(
        "Frozen decoder PCA changed."
    )

if not np.isclose(
    float(
        classifier.C
    ),
    0.1,
):
    raise ValueError(
        "Frozen logistic C changed."
    )

if (
    classifier.class_weight
    !=
    "balanced"
):
    raise ValueError(
        "Frozen classifier is not balanced."
    )

model_classes = np.asarray(
    classifier.classes_
).astype(str)

print(
    f"Frozen decoder:             "
    f"{EXPECTED_CONFIGURATION}"
)

print(
    f"Decoder classes:            "
    f"{len(model_classes)}"
)


# ============================================================
# Load frozen neural map
# ============================================================

dual_map = load_frozen_dual_map(
    MAPPING_FILE
)

# ------------------------------------------------------------
# Validate every final prompt channel BEFORE simulation
# ------------------------------------------------------------

for example in examples:

    if (
        len(
            example.prompt_tokens
        )
        >
        MAX_PROMPT_POSITIONS
    ):
        raise ValueError(
            f"Prompt too long: {example.id}"
        )

    for position, token in enumerate(
        example.prompt_tokens
    ):
        dual_map.prompt_neurons_for(
            token,
            position,
        )

# ------------------------------------------------------------
# Validate every feedback token that can be generated
# ------------------------------------------------------------

required_feedback_tokens = {
    "BOS",
    *model_classes.tolist(),
}

missing_feedback = sorted(
    token
    for token in required_feedback_tokens
    if token
    not in
    dual_map.feedback_mapping
)

if missing_feedback:
    raise ValueError(
        "Frozen mapping lacks feedback token(s): "
        f"{missing_feedback}"
    )


# ============================================================
# Load FlyBrain
# ============================================================

print()

print(
    "Loading FlyBrain on GPU..."
)

brain = FlyBrain(
    device=DEVICE,
    seed=0,
)

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
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
    len(input_neurons)
    !=
    EXPECTED_DIRECT_INPUT_NEURONS
):
    raise ValueError(
        "Direct-input population changed: "
        f"{len(input_neurons)}"
    )

if (
    len(readout_population)
    !=
    EXPECTED_READOUT_NEURONS
):
    raise ValueError(
        "Indirect readout population changed: "
        f"{len(readout_population)}"
    )

if len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) != 0:
    raise ValueError(
        "Direct-input neurons leaked into readout."
    )

expected_dimensions = (
    len(readout_population)
    *
    len(
        EXPECTED_FEATURE_BLOCKS
    )
)

if (
    expected_dimensions
    !=
    EXPECTED_FEATURE_DIMENSIONS
):
    raise ValueError(
        "Frozen feature dimensions changed: "
        f"{expected_dimensions}"
    )

print(
    "FlyBrain loaded."
)

print(
    f"Direct input neurons:       "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout neurons:   "
    f"{len(readout_population):,}"
)

print(
    f"Feature dimensions:         "
    f"{expected_dimensions:,}"
)

print()


# ============================================================
# Trace
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Prompt stimulation
# ============================================================

def stimulate_prompt(
    tokens: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    states = []

    for position, token in enumerate(
        tokens
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

            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()

            states.append(
                state
            )

        for _ in range(
            PROMPT_GAP_STEPS
        ):
            fired = brain.step()

            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()

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


# ============================================================
# Non-positional feedback stimulation
# ============================================================

def stimulate_feedback(
    token: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

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

        state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()

        states.append(
            state
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


# ============================================================
# Step-62 7-block feature
# ============================================================

def build_feature(
    prompt_summary,
    feedback_summary,
) -> np.ndarray:

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
        feature.shape[0]
        !=
        EXPECTED_FEATURE_DIMENSIONS
    ):
        raise RuntimeError(
            "Closed-loop feature dimension mismatch: "
            f"{feature.shape[0]}"
        )

    if not np.isfinite(
        feature
    ).all():
        raise RuntimeError(
            "Non-finite closed-loop feature."
        )

    return feature


# ============================================================
# Ranked decoder scores
# ============================================================

def predict_ranked_tokens(
    feature: np.ndarray,
) -> list[tuple[str, float]]:

    X = feature.reshape(
        1,
        -1,
    )

    scores = np.asarray(
        model.decision_function(
            X
        )
    )

    if scores.ndim == 1:
        scores = scores.reshape(
            1,
            -1,
        )

    row = scores[
        0
    ]

    ranked_indices = (
        np.argsort(
            row
        )[
            ::-1
        ]
    )

    return [
        (
            str(
                model_classes[
                    index
                ]
            ),
            float(
                row[
                    index
                ]
            ),
        )

        for index in ranked_indices
    ]


# ============================================================
# Generate one closed-loop program
# ============================================================

def generate_program(
    prompt_tokens: list[str],
    seed: int,
) -> tuple[
    list[str],
    list[dict],
]:

    brain.reset(
        seed=seed
    )

    trace.reset()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):
        fired = brain.step()

        trace.observe(
            fired
        )

    # --------------------------------------------------------
    # Prompt is encoded ONCE
    # --------------------------------------------------------

    prompt_summary = (
        stimulate_prompt(
            prompt_tokens
        )
    )

    feedback_token = "BOS"

    generated_tokens = []
    generation_trace = []

    # --------------------------------------------------------
    # Greedy closed-loop generation
    # --------------------------------------------------------

    for step in range(
        MAX_GENERATION_STEPS
    ):

        feedback_summary = (
            stimulate_feedback(
                feedback_token
            )
        )

        feature = build_feature(
            prompt_summary,
            feedback_summary,
        )

        ranked = (
            predict_ranked_tokens(
                feature
            )
        )

        predicted_token = (
            ranked[
                0
            ][
                0
            ]
        )

        generated_tokens.append(
            predicted_token
        )

        generation_trace.append(
            {
                "step": int(
                    step
                ),
                "feedback": str(
                    feedback_token
                ),
                "prediction": str(
                    predicted_token
                ),
                "top3": [
                    {
                        "token": str(
                            token
                        ),
                        "score": float(
                            score
                        ),
                    }
                    for token, score
                    in ranked[
                        :3
                    ]
                ],
            }
        )

        if (
            predicted_token
            ==
            "EOS"
        ):
            break

        # ----------------------------------------------------
        # CLOSED LOOP:
        # model's own prediction becomes next feedback token
        # ----------------------------------------------------

        feedback_token = (
            predicted_token
        )

    return (
        generated_tokens,
        generation_trace,
    )


# ============================================================
# Run final closed-loop evaluation
# ============================================================

results = []

total_episodes = (
    len(examples)
    *
    len(
        NEURAL_SEEDS
    )
)

if (
    total_episodes
    !=
    EXPECTED_EPISODES
):
    raise ValueError(
        f"Expected {EXPECTED_EPISODES} "
        f"episodes, got {total_episodes}"
    )

episode_number = 0

print(
    "===================================================="
)

print(
    "          FINAL CLOSED-LOOP EPISODES"
)

print(
    "===================================================="
)

print()


for example in examples:

    target_tokens = (
        code_to_tokens(
            example.code
        )
    )

    for seed in NEURAL_SEEDS:

        episode_number += 1

        (
            generated_tokens,
            generation_trace,
        ) = generate_program(
            prompt_tokens=(
                example.prompt_tokens
            ),
            seed=seed,
        )

        exact_program = (
            generated_tokens
            ==
            target_tokens
        )

        token_accuracy = (
            sequence_token_accuracy(
                target_tokens,
                generated_tokens,
            )
        )

        (
            template_valid,
            syntax_valid,
            generated_code,
        ) = decode_generated_tokens(
            generated_tokens
        )

        produced_eos = bool(
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

        mark = (
            "✓"
            if exact_program
            else "✗"
        )

        print(
            "----------------------------------------------------"
        )

        print(
            f"[{episode_number:03}/"
            f"{total_episodes:03}] "
            f"{mark} "
            f"{example.id} "
            f"seed={seed}"
        )

        print(
            f"Prompt:        "
            f"{' '.join(example.prompt_tokens)}"
        )

        print(
            f"Target code:   "
            f"{example.code}"
        )

        print(
            f"Generated:     "
            f"{generated_tokens}"
        )

        print(
            f"Generated code:"
            f" "
            f"{generated_code if generated_code is not None else '<INVALID>'}"
        )

        print(
            f"Token accuracy:"
            f" "
            f"{token_accuracy * 100:.2f}%"
        )

        print(
            f"First error:   "
            f"{first_error if first_error is not None else 'none'}"
        )

        results.append(
            {
                "example_id": (
                    example.id
                ),
                "intent": (
                    example.intent
                ),
                "seed": int(
                    seed
                ),
                "prompt_tokens": (
                    list(
                        example.prompt_tokens
                    )
                ),
                "target_code": (
                    example.code
                ),
                "target_tokens": (
                    list(
                        target_tokens
                    )
                ),
                "generated_tokens": (
                    list(
                        generated_tokens
                    )
                ),
                "generated_code": (
                    generated_code
                ),
                "template_valid_program": bool(
                    template_valid
                ),
                "syntax_valid_program": bool(
                    syntax_valid
                ),
                "exact_program": bool(
                    exact_program
                ),
                "produced_eos": bool(
                    produced_eos
                ),
                "token_accuracy": float(
                    token_accuracy
                ),
                "first_error_position": (
                    first_error
                ),
                "generation_trace": (
                    generation_trace
                ),
            }
        )


# ============================================================
# Aggregate metrics
# ============================================================

mean_token_accuracy = float(
    np.mean(
        [
            row[
                "token_accuracy"
            ]
            for row in results
        ]
    )
)

exact_program_accuracy = float(
    np.mean(
        [
            row[
                "exact_program"
            ]
            for row in results
        ]
    )
)

valid_program_rate = float(
    np.mean(
        [
            row[
                "template_valid_program"
            ]
            for row in results
        ]
    )
)

syntax_valid_rate = float(
    np.mean(
        [
            row[
                "syntax_valid_program"
            ]
            for row in results
        ]
    )
)

eos_completion_rate = float(
    np.mean(
        [
            row[
                "produced_eos"
            ]
            for row in results
        ]
    )
)


# ============================================================
# Per-seed metrics + mean +/- SD
# ============================================================

per_seed = {}

for seed in NEURAL_SEEDS:

    rows = [
        row
        for row in results
        if row[
            "seed"
        ]
        ==
        seed
    ]

    per_seed[
        str(seed)
    ] = {
        "episodes": int(
            len(rows)
        ),
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
                        "template_valid_program"
                    ]
                    for row in rows
                ]
            )
        ),
        "syntax_valid_rate": float(
            np.mean(
                [
                    row[
                        "syntax_valid_program"
                    ]
                    for row in rows
                ]
            )
        ),
        "eos_completion_rate": float(
            np.mean(
                [
                    row[
                        "produced_eos"
                    ]
                    for row in rows
                ]
            )
        ),
    }


def seed_mean_sd(
    metric_name: str,
) -> tuple[float, float]:

    values = np.asarray(
        [
            per_seed[
                str(seed)
            ][
                metric_name
            ]
            for seed in NEURAL_SEEDS
        ],
        dtype=float,
    )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.std(
                values,
                ddof=0,
            )
        ),
    )


seed_summary = {}

for metric_name in [
    "mean_token_accuracy",
    "exact_program_accuracy",
    "valid_program_rate",
    "syntax_valid_rate",
    "eos_completion_rate",
]:

    mean_value, sd_value = (
        seed_mean_sd(
            metric_name
        )
    )

    seed_summary[
        metric_name
    ] = {
        "mean": mean_value,
        "sd": sd_value,
    }


# ============================================================
# Per-intent metrics
# ============================================================

per_intent = {}

intent_names = sorted(
    {
        row[
            "intent"
        ]
        for row in results
    }
)

for intent in intent_names:

    rows = [
        row
        for row in results
        if row[
            "intent"
        ]
        ==
        intent
    ]

    per_intent[
        intent
    ] = {
        "episodes": int(
            len(rows)
        ),
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
                        "template_valid_program"
                    ]
                    for row in rows
                ]
            )
        ),
        "syntax_valid_rate": float(
            np.mean(
                [
                    row[
                        "syntax_valid_program"
                    ]
                    for row in rows
                ]
            )
        ),
        "eos_completion_rate": float(
            np.mean(
                [
                    row[
                        "produced_eos"
                    ]
                    for row in rows
                ]
            )
        ),
    }


# ============================================================
# Per-prompt metrics
# ============================================================

per_prompt = {}

for example in examples:

    rows = [
        row
        for row in results
        if row[
            "example_id"
        ]
        ==
        example.id
    ]

    per_prompt[
        example.id
    ] = {
        "intent": (
            example.intent
        ),
        "episodes": int(
            len(rows)
        ),
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
                        "template_valid_program"
                    ]
                    for row in rows
                ]
            )
        ),
    }


# ============================================================
# First-error analysis
# ============================================================

first_error_counter = Counter()

for row in results:

    first_error = row[
        "first_error_position"
    ]

    if first_error is None:
        first_error_counter[
            "exact"
        ] += 1

    else:
        first_error_counter[
            f"P{first_error}"
        ] += 1


# ============================================================
# Common generated sequences
# ============================================================

sequence_counter = Counter(
    tuple(
        row[
            "generated_tokens"
        ]
    )
    for row in results
)


# ============================================================
# Final report
# ============================================================

print()

print(
    "===================================================="
)

print(
    "       FINAL HOLDOUT - CLOSED-LOOP RESULTS"
)

print(
    "===================================================="
)

print()

print(
    f"Generation episodes:        "
    f"{total_episodes}"
)

print(
    f"Mean token accuracy:        "
    f"{mean_token_accuracy * 100:.2f}%"
)

print(
    f"Exact-program accuracy:     "
    f"{exact_program_accuracy * 100:.2f}%"
)

print(
    f"Valid-program rate:         "
    f"{valid_program_rate * 100:.2f}%"
)

print(
    f"Syntax-valid rate:          "
    f"{syntax_valid_rate * 100:.2f}%"
)

print(
    f"EOS completion rate:        "
    f"{eos_completion_rate * 100:.2f}%"
)

print()

print(
    "Mean +/- SD across neural seeds:"
)

print(
    f"  Token accuracy: "
    f"{seed_summary['mean_token_accuracy']['mean'] * 100:.2f}% "
    f"+/- "
    f"{seed_summary['mean_token_accuracy']['sd'] * 100:.2f} pp"
)

print(
    f"  Exact programs: "
    f"{seed_summary['exact_program_accuracy']['mean'] * 100:.2f}% "
    f"+/- "
    f"{seed_summary['exact_program_accuracy']['sd'] * 100:.2f} pp"
)

print(
    f"  Valid programs: "
    f"{seed_summary['valid_program_rate']['mean'] * 100:.2f}% "
    f"+/- "
    f"{seed_summary['valid_program_rate']['sd'] * 100:.2f} pp"
)

print(
    f"  EOS completion: "
    f"{seed_summary['eos_completion_rate']['mean'] * 100:.2f}% "
    f"+/- "
    f"{seed_summary['eos_completion_rate']['sd'] * 100:.2f} pp"
)


# ============================================================
# Per-seed report
# ============================================================

print()

print(
    "Per-seed closed-loop results:"
)

for seed in NEURAL_SEEDS:

    row = per_seed[
        str(seed)
    ]

    print(
        f"  seed={seed}: "
        f"token="
        f"{row['mean_token_accuracy'] * 100:6.2f}%  "
        f"exact="
        f"{row['exact_program_accuracy'] * 100:6.2f}%  "
        f"valid="
        f"{row['valid_program_rate'] * 100:6.2f}%  "
        f"EOS="
        f"{row['eos_completion_rate'] * 100:6.2f}%"
    )


# ============================================================
# Per-intent report
# ============================================================

print()

print(
    "Closed-loop accuracy by intent:"
)

for intent in intent_names:

    row = per_intent[
        intent
    ]

    print(
        f"  {intent:10} "
        f"token="
        f"{row['mean_token_accuracy'] * 100:6.2f}%  "
        f"exact="
        f"{row['exact_program_accuracy'] * 100:6.2f}%  "
        f"valid="
        f"{row['valid_program_rate'] * 100:6.2f}%"
    )


# ============================================================
# First-error report
# ============================================================

print()

print(
    "First error position:"
)

for key, count in sorted(
    first_error_counter.items(),
    key=lambda item: (
        999
        if item[0] == "exact"
        else int(
            item[0][1:]
        )
    ),
):

    print(
        f"  {key:6} "
        f"{count:3}/{total_episodes} "
        f"= "
        f"{count / total_episodes * 100:6.2f}%"
    )


# ============================================================
# Development context comparison
# ============================================================

print()

print(
    "Development closed-loop diagnostic "
    "(context only, NOT final):"
)

print(
    f"  Token accuracy: "
    f"{DEVELOPMENT_CLOSED_LOOP['mean_token_accuracy'] * 100:.2f}%"
)

print(
    f"  Exact programs: "
    f"{DEVELOPMENT_CLOSED_LOOP['exact_program_accuracy'] * 100:.2f}%"
)

print(
    f"  Valid programs: "
    f"{DEVELOPMENT_CLOSED_LOOP['valid_program_rate'] * 100:.2f}%"
)

print(
    f"  EOS completion: "
    f"{DEVELOPMENT_CLOSED_LOOP['eos_completion_rate'] * 100:.2f}%"
)

print()

print(
    "Final minus development diagnostic:"
)

print(
    f"  Token accuracy: "
    f"{(mean_token_accuracy - DEVELOPMENT_CLOSED_LOOP['mean_token_accuracy']) * 100:+.2f} pp"
)

print(
    f"  Exact programs: "
    f"{(exact_program_accuracy - DEVELOPMENT_CLOSED_LOOP['exact_program_accuracy']) * 100:+.2f} pp"
)

print(
    f"  Valid programs: "
    f"{(valid_program_rate - DEVELOPMENT_CLOSED_LOOP['valid_program_rate']) * 100:+.2f} pp"
)

print(
    f"  EOS completion: "
    f"{(eos_completion_rate - DEVELOPMENT_CLOSED_LOOP['eos_completion_rate']) * 100:+.2f} pp"
)


# ============================================================
# Most common generated sequences
# ============================================================

print()

print(
    "Most common generated sequences:"
)

for sequence, count in (
    sequence_counter.most_common(
        10
    )
):

    print(
        f"  {count:3}x "
        f"{list(sequence)}"
    )


# ============================================================
# Save JSONL predictions
# ============================================================

PREDICTIONS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with PREDICTIONS_FILE.open(
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


# ============================================================
# Save final metrics
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output = {
    "step": "77E",
    "created_at_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),
    "status": (
        "FINAL_CLOSED_LOOP_EVALUATION_COMPLETE"
    ),
    "frozen_configuration": (
        EXPECTED_CONFIGURATION
    ),
    "device": DEVICE,
    "logical_prompts": int(
        len(examples)
    ),
    "neural_seeds": (
        NEURAL_SEEDS
    ),
    "generation_episodes": int(
        total_episodes
    ),
    "max_generation_steps": int(
        MAX_GENERATION_STEPS
    ),
    "aggregate": {
        "mean_token_accuracy": (
            mean_token_accuracy
        ),
        "exact_program_accuracy": (
            exact_program_accuracy
        ),
        "valid_program_rate": (
            valid_program_rate
        ),
        "syntax_valid_rate": (
            syntax_valid_rate
        ),
        "eos_completion_rate": (
            eos_completion_rate
        ),
    },
    "mean_sd_across_neural_seeds": (
        seed_summary
    ),
    "per_seed": (
        per_seed
    ),
    "per_intent": (
        per_intent
    ),
    "per_prompt": (
        per_prompt
    ),
    "first_error_position": (
        dict(
            first_error_counter
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
        in sequence_counter.most_common(
            20
        )
    ],
    "development_context_only": (
        DEVELOPMENT_CLOSED_LOOP
    ),
    "integrity": {
        "protocol_sha256": (
            sha256_file(
                PROTOCOL_FILE
            )
        ),
        "model_sha256": (
            sha256_file(
                MODEL_FILE
            )
        ),
        "final_holdout_sha256": (
            sha256_file(
                FINAL_HOLDOUT_FILE
            )
        ),
        "mapping_sha256": (
            sha256_file(
                MAPPING_FILE
            )
        ),
        "direct_input_neurons": int(
            len(
                input_neurons
            )
        ),
        "indirect_readout_neurons": int(
            len(
                readout_population
            )
        ),
        "feature_dimensions": int(
            EXPECTED_FEATURE_DIMENSIONS
        ),
    },
    "rules": [
        (
            "No fitting was performed during "
            "Step 77E."
        ),
        (
            "No architecture or hyperparameter "
            "selection used final-holdout results."
        ),
        (
            "Generation used greedy top-1 closed-loop "
            "feedback with the frozen Step-62 model."
        ),
        (
            "The saved Step-62 neural mapping was loaded "
            "rather than recreated from the final holdout."
        ),
    ],
}

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


print()

print(
    "Saved:"
)

print(
    "  "
    f"{OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    "  "
    f"{PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}"
)

print()

print(
    "===================================================="
)

print(
    "                 STEP 77E COMPLETE"
)

print(
    "===================================================="
)

print()

print(
    "Final holdout has now been scored in both "
    "teacher-forced and closed-loop modes."
)

print()

print(
    "Do not tune the architecture or decoder "
    "from these final results."
)


if __name__ == "__main__":
    pass
