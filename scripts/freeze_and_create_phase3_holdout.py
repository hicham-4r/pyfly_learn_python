from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pyfly.generator_dataset import load_generator_jsonl
from pyfly.tokenizer import tokens_to_code


# ============================================================
# Phase 3 — Trainable PyFly
# Step 82 — FREEZE PIPELINE + CREATE FRESH HOLDOUT
#
# IMPORTANT:
#   This script DOES NOT evaluate the new holdout.
#
# It:
#   1. Freezes/hashes the Step-80 model, scaler and mapping.
#   2. Reads only the already-used development splits.
#   3. Reconstructs the six intent-specific prompt grammars.
#   4. Generates NEW prompt-token combinations never present
#      in train / validation / development-test.
#   5. Creates 4 new logical prompts per intent = 24 prompts.
#   6. Saves and hashes the holdout.
#
# The new holdout is generated AFTER the Step-80/81 pipeline
# is frozen and is not scored here.
#
# This is a controlled COMPOSITIONAL holdout:
# every prompt token at each position was already observed for
# that same intent, but the complete token combination is new.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Already-used development teacher-forcing splits
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

DEV_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_test.jsonl"
)


# ============================================================
# Frozen Phase-3 pipeline
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
# New Phase-3 holdout
# ============================================================

HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phase3_fresh_holdout.jsonl"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_final_evaluation_protocol.json"
)


# ============================================================
# Precommitted holdout construction
# ============================================================

HOLDOUT_SEED = 82_001

PROMPTS_PER_INTENT = 4

EXPECTED_INTENTS = 6

EXPECTED_HOLDOUT_PROMPTS = (
    EXPECTED_INTENTS
    *
    PROMPTS_PER_INTENT
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
            key=lambda row:
                row.target_position
        )

        positions = [
            int(
                row.target_position
            )

            for row
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
                "Non-contiguous teacher-forcing positions "
                f"for {example_id}: {positions}"
            )

    return grouped


def prompt_tuple(
    example_rows,
) -> tuple[str, ...]:

    prompt = tuple(
        str(
            token
        )

        for token
        in example_rows[
            0
        ].prompt_tokens
    )

    for row in example_rows:

        other = tuple(
            str(
                token
            )

            for token
            in row.prompt_tokens
        )

        if other != prompt:

            raise ValueError(
                "Teacher-forcing rows disagree on "
                f"prompt tokens for {row.example_id}"
            )

    return prompt


def target_tuple(
    example_rows,
) -> tuple[str, ...]:

    return tuple(
        str(
            row.target_token
        )

        for row
        in example_rows
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
        "       STEP 82 - FREEZE + FRESH HOLDOUT"
    )

    print(
        "===================================================="
    )

    print()

    required = [
        TRAIN_FILE,
        VALIDATION_FILE,
        DEV_TEST_FILE,
        MODEL_FILE,
        SCALER_FILE,
        MAPPING_FILE,
        STEP80_METRICS_FILE,
        STEP81_METRICS_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    step80 = load_json(
        STEP80_METRICS_FILE
    )

    step81 = load_json(
        STEP81_METRICS_FILE
    )

    if (
        step80.get(
            "status"
        )
        !=
        "GPU_NEURAL_ADAPTER_TRAINING_COMPLETE"
    ):

        raise ValueError(
            "Step 80 is not marked complete."
        )

    if (
        step81.get(
            "status"
        )
        !=
        "DEVELOPMENT_CLOSED_LOOP_COMPLETE"
    ):

        raise ValueError(
            "Step 81 is not marked complete."
        )

    print(
        f"Frozen Step-80 best epoch: "
        f"{step80.get('best_epoch')}"
    )

    print(
        "Step-81 closed-loop diagnostic: COMPLETE"
    )

    print()

    print(
        "NO Phase-3 final-holdout scoring "
        "will occur in Step 82."
    )

    # ========================================================
    # Load all already-used development data
    # ========================================================

    train_samples = load_generator_jsonl(
        TRAIN_FILE
    )

    validation_samples = (
        load_generator_jsonl(
            VALIDATION_FILE
        )
    )

    test_samples = load_generator_jsonl(
        DEV_TEST_FILE
    )

    all_samples = (
        list(
            train_samples
        )
        +
        list(
            validation_samples
        )
        +
        list(
            test_samples
        )
    )

    grouped = group_by_example(
        all_samples
    )

    # ========================================================
    # Recover each logical development prompt
    # ========================================================

    logical_examples = []

    for example_id in sorted(
        grouped
    ):

        rows = grouped[
            example_id
        ]

        intent = str(
            rows[
                0
            ].intent
        )

        for row in rows:

            if str(
                row.intent
            ) != intent:

                raise ValueError(
                    f"Intent mismatch inside {example_id}"
                )

        logical_examples.append(
            {
                "example_id": (
                    example_id
                ),

                "intent": (
                    intent
                ),

                "prompt_tokens": (
                    prompt_tuple(
                        rows
                    )
                ),

                "target_tokens": (
                    target_tuple(
                        rows
                    )
                ),
            }
        )

    intents = sorted(
        {
            row[
                "intent"
            ]

            for row
            in logical_examples
        }
    )

    if (
        len(
            intents
        )
        !=
        EXPECTED_INTENTS
    ):

        raise ValueError(
            f"Expected {EXPECTED_INTENTS} intents, "
            f"found {len(intents)}: {intents}"
        )

    print()

    print(
        f"Development logical prompts: "
        f"{len(logical_examples)}"
    )

    print(
        f"Intents:                    "
        f"{len(intents)}"
    )

    for intent in intents:

        count = sum(
            row[
                "intent"
            ]
            ==
            intent

            for row
            in logical_examples
        )

        print(
            f"  {intent:12} "
            f"{count} development prompts"
        )

    # ========================================================
    # Existing exact prompt signatures
    # ========================================================

    existing_prompts = {
        (
            row[
                "intent"
            ],
            row[
                "prompt_tokens"
            ],
        )

        for row
        in logical_examples
    }

    # ========================================================
    # Generate fresh compositional prompts intent-by-intent
    # ========================================================

    rng = np.random.default_rng(
        HOLDOUT_SEED
    )

    holdout_rows = []

    generation_diagnostics = {}

    for intent_index, intent in enumerate(
        intents
    ):

        intent_rows = [
            row

            for row
            in logical_examples

            if row[
                "intent"
            ]
            ==
            intent
        ]

        # ----------------------------------------------------
        # Target program MUST be invariant within intent.
        # Otherwise position-wise prompt recombination would
        # not have a well-defined target.
        # ----------------------------------------------------

        target_counter = Counter(
            row[
                "target_tokens"
            ]

            for row
            in intent_rows
        )

        if len(
            target_counter
        ) != 1:

            raise ValueError(
                f"Intent {intent!r} has multiple target "
                "programs; cannot safely create a "
                "compositional holdout."
            )

        target_tokens = next(
            iter(
                target_counter
            )
        )

        # ----------------------------------------------------
        # Prompt length must also be invariant within intent.
        # ----------------------------------------------------

        lengths = {
            len(
                row[
                    "prompt_tokens"
                ]
            )

            for row
            in intent_rows
        }

        if len(
            lengths
        ) != 1:

            raise ValueError(
                f"Intent {intent!r} has multiple prompt "
                f"lengths: {sorted(lengths)}"
            )

        prompt_length = next(
            iter(
                lengths
            )
        )

        # ----------------------------------------------------
        # Position-wise token vocabulary for this intent.
        # ----------------------------------------------------

        position_options = []

        for position in range(
            prompt_length
        ):

            options = sorted(
                {
                    row[
                        "prompt_tokens"
                    ][
                        position
                    ]

                    for row
                    in intent_rows
                }
            )

            if not options:

                raise RuntimeError(
                    f"No tokens at {intent} position "
                    f"{position}"
                )

            position_options.append(
                options
            )

        total_combinations = int(
            np.prod(
                [
                    len(
                        options
                    )

                    for options
                    in position_options
                ],
                dtype=np.int64,
            )
        )

        # This project has short prompts. Materializing the
        # product is safe and makes rejection auditable.
        candidates = []

        for combination in itertools.product(
            *position_options
        ):

            candidate = tuple(
                str(
                    token
                )

                for token
                in combination
            )

            if (
                intent,
                candidate,
            ) in existing_prompts:

                continue

            candidates.append(
                candidate
            )

        # Ensure deterministic uniqueness.
        candidates = sorted(
            set(
                candidates
            )
        )

        if (
            len(
                candidates
            )
            <
            PROMPTS_PER_INTENT
        ):

            raise RuntimeError(
                f"Intent {intent!r} produced only "
                f"{len(candidates)} unseen prompt "
                f"combinations; need "
                f"{PROMPTS_PER_INTENT}."
            )

        permutation = rng.permutation(
            len(
                candidates
            )
        )

        selected = [
            candidates[
                int(
                    index
                )
            ]

            for index
            in permutation[
                :PROMPTS_PER_INTENT
            ]
        ]

        # ----------------------------------------------------
        # Convert target token sequence back to source code.
        # ----------------------------------------------------

        code = tokens_to_code(
            list(
                target_tokens
            )
        )

        generation_diagnostics[
            intent
        ] = {
            "prompt_length": int(
                prompt_length
            ),

            "position_option_counts": [
                int(
                    len(
                        options
                    )
                )

                for options
                in position_options
            ],

            "cartesian_combinations": (
                total_combinations
            ),

            "unseen_candidates": int(
                len(
                    candidates
                )
            ),

            "selected": int(
                len(
                    selected
                )
            ),
        }

        for local_index, prompt in enumerate(
            selected,
            start=1,
        ):

            holdout_rows.append(
                {
                    "id": (
                        f"phase3_{intent}_"
                        f"{local_index:02}"
                    ),

                    "intent": (
                        intent
                    ),

                    "prompt_tokens": list(
                        prompt
                    ),

                    "code": (
                        code
                    ),
                }
            )

    # ========================================================
    # Final leakage / uniqueness verification
    # ========================================================

    if (
        len(
            holdout_rows
        )
        !=
        EXPECTED_HOLDOUT_PROMPTS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_HOLDOUT_PROMPTS} holdout "
            f"prompts, generated "
            f"{len(holdout_rows)}."
        )

    new_signatures = [
        (
            row[
                "intent"
            ],
            tuple(
                row[
                    "prompt_tokens"
                ]
            ),
        )

        for row
        in holdout_rows
    ]

    if (
        len(
            set(
                new_signatures
            )
        )
        !=
        len(
            new_signatures
        )
    ):

        raise RuntimeError(
            "Duplicate prompt in new holdout."
        )

    overlap = (
        set(
            new_signatures
        )
        &
        existing_prompts
    )

    if overlap:

        raise RuntimeError(
            f"Fresh holdout leaked "
            f"{len(overlap)} development prompts."
        )

    per_intent = Counter(
        row[
            "intent"
        ]

        for row
        in holdout_rows
    )

    for intent in intents:

        if (
            per_intent[
                intent
            ]
            !=
            PROMPTS_PER_INTENT
        ):

            raise RuntimeError(
                f"Intent {intent!r} has "
                f"{per_intent[intent]} holdout prompts, "
                f"expected {PROMPTS_PER_INTENT}."
            )

    # ========================================================
    # Save fresh holdout — DO NOT SCORE IT
    # ========================================================

    HOLDOUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HOLDOUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in holdout_rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )

    holdout_hash = sha256_file(
        HOLDOUT_FILE
    )

    # ========================================================
    # Freeze full final-evaluation protocol
    # ========================================================

    protocol = {
        "phase": 3,

        "step": "82",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FROZEN_BEFORE_PHASE3_FINAL_EVALUATION"
        ),

        "holdout_type": (
            "fresh compositional logical-prompt holdout"
        ),

        "holdout_seed": (
            HOLDOUT_SEED
        ),

        "holdout_prompts": (
            EXPECTED_HOLDOUT_PROMPTS
        ),

        "prompts_per_intent": (
            PROMPTS_PER_INTENT
        ),

        "intents": (
            intents
        ),

        "construction": (
            "For each intent, take the observed token "
            "options at each prompt position across all "
            "development prompts, enumerate position-wise "
            "combinations, reject every complete prompt "
            "already used in train/validation/test, then "
            "sample four unseen combinations deterministically."
        ),

        "generation_diagnostics": (
            generation_diagnostics
        ),

        "development_prompt_overlap": (
            0
        ),

        "scoring_performed_in_step82": (
            False
        ),

        "frozen_artifacts": {
            "model": {
                "path": (
                    relative(
                        MODEL_FILE
                    )
                ),

                "sha256": (
                    sha256_file(
                        MODEL_FILE
                    )
                ),
            },

            "scaler": {
                "path": (
                    relative(
                        SCALER_FILE
                    )
                ),

                "sha256": (
                    sha256_file(
                        SCALER_FILE
                    )
                ),
            },

            "mapping": {
                "path": (
                    relative(
                        MAPPING_FILE
                    )
                ),

                "sha256": (
                    sha256_file(
                        MAPPING_FILE
                    )
                ),
            },

            "step80_metrics": {
                "path": (
                    relative(
                        STEP80_METRICS_FILE
                    )
                ),

                "sha256": (
                    sha256_file(
                        STEP80_METRICS_FILE
                    )
                ),
            },

            "step81_metrics": {
                "path": (
                    relative(
                        STEP81_METRICS_FILE
                    )
                ),

                "sha256": (
                    sha256_file(
                        STEP81_METRICS_FILE
                    )
                ),
            },

            "fresh_holdout": {
                "path": (
                    relative(
                        HOLDOUT_FILE
                    )
                ),

                "sha256": (
                    holdout_hash
                ),
            },
        },

        "rules_after_freeze": [
            (
                "Do not retrain the Step-80 adapter before "
                "Phase-3 final evaluation."
            ),

            (
                "Do not change neural mapping, stimulation "
                "settings, architecture, scaler, decoding "
                "logic, or closed-loop feedback policy."
            ),

            (
                "Do not inspect final holdout performance "
                "until the evaluation script is ready."
            ),

            (
                "After final evaluation, results may be "
                "analyzed but must not be used to retroactively "
                "tune the reported Phase-3 model."
            ),
        ],
    }

    PROTOCOL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    print()

    print(
        "===================================================="
    )

    print(
        "       FRESH PHASE-3 HOLDOUT CREATED"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Logical prompts:            "
        f"{len(holdout_rows)}"
    )

    print(
        f"Intents:                    "
        f"{len(intents)}"
    )

    print(
        f"Prompts per intent:         "
        f"{PROMPTS_PER_INTENT}"
    )

    print(
        f"Development prompt overlap: "
        f"{len(overlap)}"
    )

    print()

    for intent in intents:

        info = generation_diagnostics[
            intent
        ]

        print(
            f"{intent:12} "
            f"options="
            f"{info['position_option_counts']} "
            f"unseen="
            f"{info['unseen_candidates']} "
            f"selected="
            f"{info['selected']}"
        )

    print()

    print(
        "Created:"
    )

    print(
        "  "
        f"{relative(HOLDOUT_FILE)}"
    )

    print(
        "  "
        f"{relative(PROTOCOL_FILE)}"
    )

    print()

    print(
        f"Holdout SHA-256:"
    )

    print(
        f"  {holdout_hash}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 82 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "The Step-80/81 Phase-3 pipeline is now frozen."
    )

    print()

    print(
        "The new 24-prompt holdout has NOT been scored."
    )

    print()

    print(
        "Next: Step 83 - one-time teacher-forced and "
        "true closed-loop final evaluation on this "
        "frozen Phase-3 holdout."
    )


if __name__ == "__main__":
    main()
