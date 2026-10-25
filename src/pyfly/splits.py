from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pyfly.dataset import Example


def create_splits(
    examples: list[Example],
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Create deterministic stratified train/validation/test splits.

    For each intent:
        6 train
        1 validation
        1 test

    Assumes 8 logical prompts per intent.
    """

    rng = random.Random(seed)

    by_intent: dict[str, list[Example]] = defaultdict(list)

    for example in examples:
        by_intent[example.intent].append(example)

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for intent in sorted(by_intent):

        group = by_intent[intent].copy()

        rng.shuffle(group)

        if len(group) != 8:
            raise ValueError(
                f"Expected 8 examples for intent '{intent}', "
                f"found {len(group)}"
            )

        train = group[:6]
        validation = group[6:7]
        test = group[7:8]

        splits["train"].extend(
            example.id for example in train
        )

        splits["validation"].extend(
            example.id for example in validation
        )

        splits["test"].extend(
            example.id for example in test
        )

    return splits


def save_splits(
    splits: dict[str, list[str]],
    path: str | Path,
    seed: int,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "seed": seed,
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )


def load_splits(
    path: str | Path,
) -> dict:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)