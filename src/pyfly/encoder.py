from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StimulusEvent:
    token: str
    neurons: np.ndarray
    steps: int
    amplitude: float


def build_prompt_schedule(
    prompt_tokens: list[str],
    sensory_map,
    token_steps: int = 8,
    gap_steps: int = 2,
    amplitude: float = 0.8,
):
    """
    Convert a sequence such as:

        ["add", "a", "b"]

    into a temporal stimulation schedule.

    Each token is presented for token_steps,
    followed by gap_steps with no stimulation.
    """

    schedule = []

    for token in prompt_tokens:

        neurons = sensory_map.neurons_for(token)

        schedule.append(
            StimulusEvent(
                token=token,
                neurons=neurons,
                steps=token_steps,
                amplitude=amplitude,
            )
        )

        if gap_steps > 0:

            schedule.append(
                StimulusEvent(
                    token="<GAP>",
                    neurons=np.array([], dtype=np.int64),
                    steps=gap_steps,
                    amplitude=0.0,
                )
            )

    return schedule