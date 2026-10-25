from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    intent: str
    prompt_tokens: list[str]
    code: str


def load_dataset(path: str | Path) -> list[Example]:
    """
    Load the PyFly JSONL dataset.

    Each line must contain:
        id
        intent
        prompt_tokens
        code
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[Example] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            required_fields = {
                "id",
                "intent",
                "prompt_tokens",
                "code",
            }

            missing = required_fields - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} missing fields: {sorted(missing)}"
                )

            if not isinstance(record["prompt_tokens"], list):
                raise TypeError(
                    f"prompt_tokens must be a list on line {line_number}"
                )

            examples.append(
                Example(
                    id=record["id"],
                    intent=record["intent"],
                    prompt_tokens=record["prompt_tokens"],
                    code=record["code"],
                )
            )

    if not examples:
        raise ValueError("Dataset is empty")

    return examples