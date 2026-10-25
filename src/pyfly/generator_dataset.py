from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pyfly.dataset import Example
from pyfly.tokenizer import build_teacher_forcing_pairs


# ============================================================
# Generator Training Example
# ============================================================

@dataclass(frozen=True)
class GeneratorExample:
    sample_id: str

    example_id: str
    intent: str

    prompt_tokens: list[str]

    context_tokens: list[str]

    target_token: str

    target_position: int

    code: str

    split: str


# ============================================================
# Convert one logical prompt into next-token samples
# ============================================================

def build_generator_examples(
    example: Example,
    split: str,
) -> list[GeneratorExample]:

    teacher_pairs = build_teacher_forcing_pairs(
        example.code
    )

    samples: list[GeneratorExample] = []


    for target_position, (
        context_tokens,
        target_token,
    ) in enumerate(
        teacher_pairs
    ):

        sample_id = (
            f"{example.id}"
            f"_step_{target_position:02}"
        )


        samples.append(
            GeneratorExample(
                sample_id=sample_id,

                example_id=example.id,

                intent=example.intent,

                prompt_tokens=(
                    example.prompt_tokens.copy()
                ),

                context_tokens=(
                    context_tokens.copy()
                ),

                target_token=target_token,

                target_position=(
                    target_position
                ),

                code=example.code,

                split=split,
            )
        )


    return samples


# ============================================================
# Save JSONL
# ============================================================

def save_generator_jsonl(
    samples: list[GeneratorExample],
    path: str | Path,
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for sample in samples:

            record = {
                "sample_id": (
                    sample.sample_id
                ),

                "example_id": (
                    sample.example_id
                ),

                "intent": (
                    sample.intent
                ),

                "prompt_tokens": (
                    sample.prompt_tokens
                ),

                "context_tokens": (
                    sample.context_tokens
                ),

                "target_token": (
                    sample.target_token
                ),

                "target_position": (
                    sample.target_position
                ),

                "code": (
                    sample.code
                ),

                "split": (
                    sample.split
                ),
            }


            file.write(
                json.dumps(
                    record
                )
            )

            file.write(
                "\n"
            )


# ============================================================
# Load JSONL
# ============================================================

def load_generator_jsonl(
    path: str | Path,
) -> list[GeneratorExample]:

    path = Path(
        path
    )


    samples = []


    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue


            record = json.loads(
                line
            )


            samples.append(
                GeneratorExample(
                    sample_id=record[
                        "sample_id"
                    ],

                    example_id=record[
                        "example_id"
                    ],

                    intent=record[
                        "intent"
                    ],

                    prompt_tokens=record[
                        "prompt_tokens"
                    ],

                    context_tokens=record[
                        "context_tokens"
                    ],

                    target_token=record[
                        "target_token"
                    ],

                    target_position=record[
                        "target_position"
                    ],

                    code=record[
                        "code"
                    ],

                    split=record[
                        "split"
                    ],
                )
            )


    return samples