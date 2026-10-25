from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pyfly.dataset import Example, load_dataset
from pyfly.generator_dataset import (
    build_generator_examples,
    save_generator_jsonl,
)
from pyfly.tokenizer import code_to_tokens


# ============================================================
# Step 77A
# Freeze the winning Step-62 protocol and create a fresh
# logical-prompt holdout for final evaluation.
#
# IMPORTANT:
#   * Do NOT use this holdout for model/hyperparameter selection.
#   * Do NOT change the architecture after evaluating it.
#   * The MaleCNS recurrent weights remain frozen.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sequence_challenge.jsonl"
)

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
)

FINAL_GENERATOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_final_holdout.jsonl"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

DECODER_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "generator_decoder_comparison.json"
)

DUAL_MAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)

SOURCE_FILES_TO_FREEZE = [
    PROJECT_ROOT
    / "src"
    / "pyfly"
    / "dual_channel_map.py",

    PROJECT_ROOT
    / "src"
    / "pyfly"
    / "generator_dataset.py",

    PROJECT_ROOT
    / "src"
    / "pyfly"
    / "tokenizer.py",

    PROJECT_ROOT
    / "scripts"
    / "extract_generator_features.py",

    PROJECT_ROOT
    / "scripts"
    / "compare_generator_decoders.py",

    PROJECT_ROOT
    / "scripts"
    / "generate_closed_loop_best_decoder.py",
]


# ============================================================
# Frozen Step-62 configuration
# ============================================================

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

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


FEATURE_BLOCKS = [
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
]


# ============================================================
# Six frozen sequence-challenge intents
# ============================================================

INTENT_TO_CODE = {
    "sub_ab": "y = a - b",
    "sub_ba": "y = b - a",

    "copy_a_x": "x = a",
    "copy_x_a": "a = x",

    "div_ab": "y = a / b",
    "div_ba": "y = b / a",
}


# ============================================================
# Fresh final holdout
#
# 24 logical prompts:
#
#     6 intents
#     ×
#     4 new prompts
#
# Important:
#
# Every prompt uses only existing vocabulary.
#
# Every (token, position) channel has already appeared
# somewhere in the 48-prompt development dataset.
#
# Therefore we are testing new prompt combinations rather
# than introducing completely unseen neural input channels.
# ============================================================

FINAL_HOLDOUT_RECORDS = [

    # ========================================================
    # SUBTRACT
    # y = a - b
    # ========================================================

    {
        "id": "final_sub_ab_01",
        "intent": "sub_ab",

        "prompt_tokens": [
            "take",
            "value",
            "b",
            "from",
            "value",
            "a",
        ],

        "code": "y = a - b",
    },

    {
        "id": "final_sub_ab_02",
        "intent": "sub_ab",

        "prompt_tokens": [
            "remove",
            "value",
            "b",
            "from",
            "value",
            "a",
        ],

        "code": "y = a - b",
    },

    {
        "id": "final_sub_ab_03",
        "intent": "sub_ab",

        "prompt_tokens": [
            "compute",
            "a",
            "minus",
            "b",
        ],

        "code": "y = a - b",
    },

    {
        "id": "final_sub_ab_04",
        "intent": "sub_ab",

        "prompt_tokens": [
            "difference",
            "value",
            "b",
            "from",
            "value",
            "a",
        ],

        "code": "y = a - b",
    },


    # ========================================================
    # SUBTRACT
    # y = b - a
    # ========================================================

    {
        "id": "final_sub_ba_01",
        "intent": "sub_ba",

        "prompt_tokens": [
            "take",
            "value",
            "a",
            "from",
            "value",
            "b",
        ],

        "code": "y = b - a",
    },

    {
        "id": "final_sub_ba_02",
        "intent": "sub_ba",

        "prompt_tokens": [
            "remove",
            "value",
            "a",
            "from",
            "value",
            "b",
        ],

        "code": "y = b - a",
    },

    {
        "id": "final_sub_ba_03",
        "intent": "sub_ba",

        "prompt_tokens": [
            "compute",
            "b",
            "minus",
            "a",
        ],

        "code": "y = b - a",
    },

    {
        "id": "final_sub_ba_04",
        "intent": "sub_ba",

        "prompt_tokens": [
            "difference",
            "value",
            "a",
            "from",
            "value",
            "b",
        ],

        "code": "y = b - a",
    },


    # ========================================================
    # COPY
    # x = a
    # ========================================================

    {
        "id": "final_copy_a_x_01",
        "intent": "copy_a_x",

        "prompt_tokens": [
            "assign",
            "a",
            "into",
            "x",
        ],

        "code": "x = a",
    },

    {
        "id": "final_copy_a_x_02",
        "intent": "copy_a_x",

        "prompt_tokens": [
            "put",
            "a",
            "to",
            "x",
        ],

        "code": "x = a",
    },

    {
        "id": "final_copy_a_x_03",
        "intent": "copy_a_x",

        "prompt_tokens": [
            "store",
            "a",
            "to",
            "x",
        ],

        "code": "x = a",
    },

    {
        "id": "final_copy_a_x_04",
        "intent": "copy_a_x",

        "prompt_tokens": [
            "write",
            "a",
            "into",
            "x",
        ],

        "code": "x = a",
    },


    # ========================================================
    # COPY
    # a = x
    # ========================================================

    {
        "id": "final_copy_x_a_01",
        "intent": "copy_x_a",

        "prompt_tokens": [
            "assign",
            "x",
            "into",
            "a",
        ],

        "code": "a = x",
    },

    {
        "id": "final_copy_x_a_02",
        "intent": "copy_x_a",

        "prompt_tokens": [
            "put",
            "x",
            "to",
            "a",
        ],

        "code": "a = x",
    },

    {
        "id": "final_copy_x_a_03",
        "intent": "copy_x_a",

        "prompt_tokens": [
            "store",
            "x",
            "to",
            "a",
        ],

        "code": "a = x",
    },

    {
        "id": "final_copy_x_a_04",
        "intent": "copy_x_a",

        "prompt_tokens": [
            "write",
            "x",
            "into",
            "a",
        ],

        "code": "a = x",
    },


    # ========================================================
    # DIVIDE
    # y = a / b
    # ========================================================

    {
        "id": "final_div_ab_01",
        "intent": "div_ab",

        "prompt_tokens": [
            "quotient",
            "a",
            "over",
            "b",
        ],

        "code": "y = a / b",
    },

    {
        "id": "final_div_ab_02",
        "intent": "div_ab",

        "prompt_tokens": [
            "ratio",
            "a",
            "by",
            "b",
        ],

        "code": "y = a / b",
    },

    {
        "id": "final_div_ab_03",
        "intent": "div_ab",

        "prompt_tokens": [
            "result",
            "a",
            "over",
            "b",
        ],

        "code": "y = a / b",
    },

    {
        "id": "final_div_ab_04",
        "intent": "div_ab",

        "prompt_tokens": [
            "compute",
            "a",
            "by",
            "b",
        ],

        "code": "y = a / b",
    },


    # ========================================================
    # DIVIDE
    # y = b / a
    # ========================================================

    {
        "id": "final_div_ba_01",
        "intent": "div_ba",

        "prompt_tokens": [
            "quotient",
            "b",
            "over",
            "a",
        ],

        "code": "y = b / a",
    },

    {
        "id": "final_div_ba_02",
        "intent": "div_ba",

        "prompt_tokens": [
            "ratio",
            "b",
            "by",
            "a",
        ],

        "code": "y = b / a",
    },

    {
        "id": "final_div_ba_03",
        "intent": "div_ba",

        "prompt_tokens": [
            "result",
            "b",
            "over",
            "a",
        ],

        "code": "y = b / a",
    },

    {
        "id": "final_div_ba_04",
        "intent": "div_ba",

        "prompt_tokens": [
            "compute",
            "b",
            "by",
            "a",
        ],

        "code": "y = b / a",
    },
]


# ============================================================
# SHA-256 helper
# ============================================================

def sha256_file(
    path: Path,
) -> str | None:

    if not path.exists():
        return None

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


# ============================================================
# Relative project path
# ============================================================

def relative_path(
    path: Path,
) -> str:

    try:

        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    except ValueError:

        return str(
            path
        )


# ============================================================
# Write logical holdout JSONL
# ============================================================

def write_logical_holdout(
    records: list[dict],
) -> None:

    FINAL_HOLDOUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FINAL_HOLDOUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            file.write(
                "\n"
            )


# ============================================================
# Validate holdout before saving
# ============================================================

def validate_holdout(
    development_examples: list[Example],
) -> dict:

    if len(
        FINAL_HOLDOUT_RECORDS
    ) != 24:

        raise ValueError(
            "Final holdout must contain "
            "exactly 24 logical prompts."
        )


    development_ids = {
        example.id
        for example
        in development_examples
    }


    development_prompts = {
        tuple(
            example.prompt_tokens
        )

        for example
        in development_examples
    }


    development_vocab = {
        token

        for example
        in development_examples

        for token
        in example.prompt_tokens
    }


    development_token_positions = {

        (
            token,
            position,
        )

        for example
        in development_examples

        for position, token
        in enumerate(
            example.prompt_tokens
        )
    }


    holdout_ids = set()

    holdout_prompts = set()

    intent_counts = Counter()


    used_tokens = set()

    used_token_positions = set()


    generator_rows = 0


    # ========================================================
    # Validate each final prompt
    # ========================================================

    for record in FINAL_HOLDOUT_RECORDS:

        example_id = record[
            "id"
        ]

        intent = record[
            "intent"
        ]

        prompt_tokens = record[
            "prompt_tokens"
        ]

        code = record[
            "code"
        ]


        # ----------------------------------------------------
        # ID leakage
        # ----------------------------------------------------

        if example_id in development_ids:

            raise ValueError(
                f"Holdout ID already exists "
                f"in development data: "
                f"{example_id}"
            )


        if example_id in holdout_ids:

            raise ValueError(
                f"Duplicate holdout ID: "
                f"{example_id}"
            )


        holdout_ids.add(
            example_id
        )


        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------

        if intent not in INTENT_TO_CODE:

            raise ValueError(
                f"Unknown holdout intent: "
                f"{intent}"
            )


        # ----------------------------------------------------
        # Correct target program
        # ----------------------------------------------------

        if (
            code
            !=
            INTENT_TO_CODE[
                intent
            ]
        ):

            raise ValueError(
                f"Wrong code target for "
                f"{example_id}: "
                f"{code}"
            )


        # ----------------------------------------------------
        # Prompt length
        # ----------------------------------------------------

        if not (
            1
            <=
            len(
                prompt_tokens
            )
            <=
            MAX_PROMPT_POSITIONS
        ):

            raise ValueError(
                f"Prompt length out of "
                f"range for {example_id}: "
                f"{len(prompt_tokens)}"
            )


        prompt_tuple = tuple(
            prompt_tokens
        )


        # ----------------------------------------------------
        # No old prompt duplication
        # ----------------------------------------------------

        if (
            prompt_tuple
            in
            development_prompts
        ):

            raise ValueError(
                f"Holdout prompt duplicates "
                f"development prompt: "
                f"{example_id}"
            )


        if (
            prompt_tuple
            in
            holdout_prompts
        ):

            raise ValueError(
                f"Duplicate prompt inside "
                f"holdout: "
                f"{example_id}"
            )


        holdout_prompts.add(
            prompt_tuple
        )


        # ----------------------------------------------------
        # Vocabulary check
        # ----------------------------------------------------

        unknown_tokens = (
            set(
                prompt_tokens
            )
            -
            development_vocab
        )


        if unknown_tokens:

            raise ValueError(
                f"Unknown prompt token(s) "
                f"in {example_id}: "
                f"{sorted(unknown_tokens)}"
            )


        # ----------------------------------------------------
        # Position-channel check
        #
        # Because our prompt encoder is position-aware,
        # (token, position) is the actual neural channel.
        #
        # We therefore avoid creating a final benchmark
        # containing neural channels that never appeared
        # anywhere in development.
        # ----------------------------------------------------

        for (
            position,
            token,
        ) in enumerate(
            prompt_tokens
        ):

            pair = (
                token,
                position,
            )


            if (
                pair
                not in
                development_token_positions
            ):

                raise ValueError(
                    "Final holdout introduced "
                    "an unseen (token, position) "
                    "neural channel: "
                    f"{pair} "
                    f"in {example_id}"
                )


            used_tokens.add(
                token
            )

            used_token_positions.add(
                pair
            )


        # ----------------------------------------------------
        # Verify target is supported by frozen tokenizer
        # ----------------------------------------------------

        target_tokens = (
            code_to_tokens(
                code
            )
        )


        generator_rows += len(
            target_tokens
        )


        intent_counts[
            intent
        ] += 1


    # ========================================================
    # Intent balance
    # ========================================================

    if (
        set(
            intent_counts
        )
        !=
        set(
            INTENT_TO_CODE
        )
    ):

        raise ValueError(
            "Final holdout does not "
            "contain exactly the six "
            "frozen intents."
        )


    for intent in sorted(
        INTENT_TO_CODE
    ):

        if (
            intent_counts[
                intent
            ]
            !=
            4
        ):

            raise ValueError(
                f"Expected 4 holdout "
                f"prompts for {intent}, "
                f"got "
                f"{intent_counts[intent]}"
            )


    return {

        "logical_prompts": (
            len(
                FINAL_HOLDOUT_RECORDS
            )
        ),

        "intent_counts": dict(
            sorted(
                intent_counts.items()
            )
        ),

        "development_prompt_vocabulary_size": (
            len(
                development_vocab
            )
        ),

        "holdout_used_prompt_tokens": (
            len(
                used_tokens
            )
        ),

        "holdout_used_token_position_channels": (
            len(
                used_token_positions
            )
        ),

        "generator_teacher_forcing_rows": (
            generator_rows
        ),

        "all_tokens_known": True,

        "all_token_position_channels_previously_seen": (
            True
        ),

        "development_prompt_overlap": 0,

        "max_prompt_length": max(
            len(
                record[
                    "prompt_tokens"
                ]
            )

            for record
            in FINAL_HOLDOUT_RECORDS
        ),
    }


# ============================================================
# Convert logical holdout -> teacher-forcing dataset
# ============================================================

def build_generator_holdout() -> int:

    logical_examples = load_dataset(
        FINAL_HOLDOUT_FILE
    )


    generator_samples = []


    for example in logical_examples:

        generator_samples.extend(

            build_generator_examples(
                example=example,
                split="final_holdout",
            )

        )


    save_generator_jsonl(
        generator_samples,
        FINAL_GENERATOR_FILE,
    )


    return len(
        generator_samples
    )


# ============================================================
# Load the already-selected Step-62 decoder configuration
#
# We READ the winner.
#
# We DO NOT perform another model search.
# ============================================================

def load_decoder_selection() -> dict:

    if not DECODER_METRICS_FILE.exists():

        raise FileNotFoundError(
            "Missing decoder metrics: "
            f"{DECODER_METRICS_FILE}"
        )


    with DECODER_METRICS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        metrics = json.load(
            file
        )


    required = {

        "selected_family",
        "selected_configuration",

        "validation_accuracy",
        "validation_macro_f1",

        "test_top1_accuracy",
        "test_macro_f1",

        "semantic_position_accuracy",
    }


    missing = (
        required
        -
        metrics.keys()
    )


    if missing:

        raise ValueError(
            "Decoder metrics are missing "
            "required fields: "
            f"{sorted(missing)}"
        )


    return {

        "selected_family": (
            metrics[
                "selected_family"
            ]
        ),

        "selected_configuration": (
            metrics[
                "selected_configuration"
            ]
        ),

        "development_validation_accuracy": (
            metrics[
                "validation_accuracy"
            ]
        ),

        "development_validation_macro_f1": (
            metrics[
                "validation_macro_f1"
            ]
        ),

        "development_diagnostic_test_top1": (
            metrics[
                "test_top1_accuracy"
            ]
        ),

        "development_diagnostic_test_macro_f1": (
            metrics[
                "test_macro_f1"
            ]
        ),

        "development_diagnostic_semantic_accuracy": (
            metrics[
                "semantic_position_accuracy"
            ]
        ),
    }


# ============================================================
# Freeze final protocol
# ============================================================

def create_protocol(
    validation_summary: dict,
) -> dict:

    decoder_selection = (
        load_decoder_selection()
    )


    frozen_files = {}


    files_to_hash = [

        DEV_DATASET_FILE,

        FINAL_HOLDOUT_FILE,

        FINAL_GENERATOR_FILE,

        DECODER_METRICS_FILE,

        DUAL_MAP_FILE,

        *SOURCE_FILES_TO_FREEZE,
    ]


    for path in files_to_hash:

        frozen_files[
            relative_path(
                path
            )
        ] = {

            "exists": (
                path.exists()
            ),

            "sha256": (
                sha256_file(
                    path
                )
            ),
        }


    protocol = {

        "protocol": (
            "PyFly final evaluation"
        ),

        "protocol_version": 1,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FROZEN_BEFORE_FINAL_EVALUATION"
        ),


        # ====================================================
        # Frozen architecture
        # ====================================================

        "architecture": {

            "name": (
                "Step-62 original "
                "7-block generator"
            ),

            "reservoir": (
                "MaleCNS-derived FlyBrain"
            ),

            "recurrent_weights": (
                "frozen"
            ),

            "prompt_encoder": (
                "position-aware "
                "token-position population code"
            ),

            "feedback_encoder": (
                "non-positional "
                "code-token population code"
            ),

            "readout_population": (
                "indirect visual_projection"
            ),

            "directly_stimulated_neurons_excluded": (
                True
            ),

            "feature_blocks": (
                FEATURE_BLOCKS
            ),

            "feature_block_count": (
                len(
                    FEATURE_BLOCKS
                )
            ),

            "max_prompt_positions": (
                MAX_PROMPT_POSITIONS
            ),

            "prompt_neurons_per_channel": (
                PROMPT_NEURONS_PER_CHANNEL
            ),

            "feedback_neurons_per_token": (
                FEEDBACK_NEURONS_PER_TOKEN
            ),

            "prompt_token_steps": (
                PROMPT_TOKEN_STEPS
            ),

            "prompt_gap_steps": (
                PROMPT_GAP_STEPS
            ),

            "feedback_token_steps": (
                FEEDBACK_TOKEN_STEPS
            ),

            "feedback_gap_steps": (
                FEEDBACK_GAP_STEPS
            ),

            "warmup_steps": (
                WARMUP_STEPS
            ),

            "amplitude": (
                AMPLITUDE
            ),

            "trace_tau": (
                TRACE_TAU
            ),

            "mapping_seed": (
                MAP_SEED
            ),

            "neural_seeds": (
                NEURAL_SEEDS
            ),
        },


        # ====================================================
        # Frozen decoder selection
        # ====================================================

        "decoder_selection": (
            decoder_selection
        ),


        # ====================================================
        # Holdout metadata
        # ====================================================

        "final_holdout": (
            validation_summary
        ),


        # ====================================================
        # Scientific rules
        # ====================================================

        "rules": [

            (
                "Do not select architecture "
                "from final-holdout results."
            ),

            (
                "Do not select decoder family "
                "or hyperparameters from "
                "final-holdout results."
            ),

            (
                "Refit the already-selected "
                "Step-62 decoder on all 48 "
                "development prompts before "
                "final scoring."
            ),

            (
                "Keep the MaleCNS recurrent "
                "weights frozen."
            ),

            (
                "Keep the neural mapping seed "
                "and encoder settings frozen."
            ),

            (
                "Use the same five neural seeds "
                "for the final MaleCNS evaluation."
            ),

            (
                "Report macro F1, "
                "semantic-position accuracy, "
                "exact-program accuracy, "
                "valid-program rate, "
                "EOS completion, "
                "per-token confusion, "
                "per-position accuracy, "
                "and uncertainty across "
                "neural seeds."
            ),

            (
                "Compare MaleCNS with "
                "encoder-only, matched random "
                "recurrent reservoir, and "
                "shuffled-label controls before "
                "making scientific claims."
            ),
        ],


        # ====================================================
        # Integrity hashes
        # ====================================================

        "frozen_file_hashes": (
            frozen_files
        ),
    }


    return protocol


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        "       PyFly STEP 77A - FREEZE FINAL PROTOCOL"
    )

    print(
        "===================================================="
    )

    print()


    # ========================================================
    # Load original 48 development prompts
    # ========================================================

    development_examples = load_dataset(
        DEV_DATASET_FILE
    )


    if (
        len(
            development_examples
        )
        !=
        48
    ):

        raise ValueError(
            "Expected the frozen development "
            "dataset to contain 48 logical "
            "prompts, got "
            f"{len(development_examples)}"
        )


    # ========================================================
    # Validate fresh holdout
    # ========================================================

    validation_summary = (
        validate_holdout(
            development_examples
        )
    )


    # ========================================================
    # Save logical holdout
    # ========================================================

    write_logical_holdout(
        FINAL_HOLDOUT_RECORDS
    )


    # ========================================================
    # Build teacher-forcing version
    # ========================================================

    generator_rows = (
        build_generator_holdout()
    )


    if (
        generator_rows
        !=
        validation_summary[
            "generator_teacher_forcing_rows"
        ]
    ):

        raise RuntimeError(
            "Generator row count changed "
            "during save/reload."
        )


    # ========================================================
    # Create protocol
    # ========================================================

    PROTOCOL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    protocol = create_protocol(
        validation_summary
    )


    PROTOCOL_FILE.write_text(

        json.dumps(
            protocol,
            indent=2,
        ),

        encoding="utf-8",
    )


    # ========================================================
    # Report
    # ========================================================

    print(
        "Development logical prompts: 48"
    )

    print(
        "Development prompt vocabulary: "
        f"{validation_summary['development_prompt_vocabulary_size']} "
        "tokens"
    )

    print()


    print(
        "Final holdout validation:"
    )


    print(
        "  logical prompts:            "
        f"{validation_summary['logical_prompts']}"
    )


    print(
        "  prompts per intent:         4"
    )


    print(
        "  exact dev-prompt overlap:   "
        f"{validation_summary['development_prompt_overlap']}"
    )


    print(
        "  only known prompt tokens:   "
        f"{validation_summary['all_tokens_known']}"
    )


    print(
        "  known token-position pairs: "
        f"{validation_summary['all_token_position_channels_previously_seen']}"
    )


    print(
        "  maximum prompt length:      "
        f"{validation_summary['max_prompt_length']}"
    )


    print(
        "  teacher-forcing rows:       "
        f"{validation_summary['generator_teacher_forcing_rows']}"
    )


    print()


    # ========================================================
    # Frozen winner
    # ========================================================

    decoder = protocol[
        "decoder_selection"
    ]


    print(
        "Frozen winner:"
    )


    print(
        "  family:        "
        f"{decoder['selected_family']}"
    )


    print(
        "  configuration: "
        f"{decoder['selected_configuration']}"
    )


    print()


    # ========================================================
    # Files
    # ========================================================

    print(
        "Created:"
    )


    print(
        "  "
        f"{relative_path(FINAL_HOLDOUT_FILE)}"
    )


    print(
        "  "
        f"{relative_path(FINAL_GENERATOR_FILE)}"
    )


    print(
        "  "
        f"{relative_path(PROTOCOL_FILE)}"
    )


    print()


    print(
        "Holdout SHA-256: "
        f"{sha256_file(FINAL_HOLDOUT_FILE)}"
    )


    print()


    print(
        "===================================================="
    )

    print(
        "             STEP 77A COMPLETE"
    )

    print(
        "===================================================="
    )

    print()


    print(
        "Do NOT tune architecture or "
        "hyperparameters from the final "
        "holdout after this point."
    )


    print()


    print(
        "Next: Step 77B - refit the frozen "
        "Step-62 decoder on all 48 "
        "development prompts."
    )


if __name__ == "__main__":
    main()