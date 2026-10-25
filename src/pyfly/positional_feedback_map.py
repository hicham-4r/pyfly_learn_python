from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PositionalFeedbackMap:
    """
    Two neural channels:

    1. Prompt:
       (prompt_token, prompt_position) -> neuron population

    2. Generator feedback:
       (code_token, generation_position) -> neuron population

    All populations are disjoint.
    """

    prompt_mapping: dict[tuple[str, int], np.ndarray]

    feedback_mapping: dict[
        tuple[str, int],
        np.ndarray,
    ]

    prompt_neurons_per_channel: int

    feedback_neurons_per_channel: int

    max_prompt_positions: int

    max_feedback_positions: int

    seed: int


    # ========================================================
    # Prompt
    # ========================================================

    def prompt_neurons_for(
        self,
        token: str,
        position: int,
    ) -> np.ndarray:

        key = (
            token,
            position,
        )

        if key not in self.prompt_mapping:

            raise KeyError(
                f"No prompt mapping for "
                f"{token!r} at position "
                f"{position}"
            )

        return self.prompt_mapping[
            key
        ]


    # ========================================================
    # Feedback
    # ========================================================

    def feedback_neurons_for(
        self,
        token: str,
        position: int,
    ) -> np.ndarray:

        key = (
            token,
            position,
        )

        if key not in self.feedback_mapping:

            raise KeyError(
                f"No feedback mapping for "
                f"{token!r} at position "
                f"{position}"
            )

        return self.feedback_mapping[
            key
        ]


    # ========================================================
    # All directly stimulated neurons
    # ========================================================

    def all_input_neurons(
        self,
    ) -> np.ndarray:

        neurons = []


        for population in (
            self.prompt_mapping.values()
        ):

            neurons.extend(
                population.tolist()
            )


        for population in (
            self.feedback_mapping.values()
        ):

            neurons.extend(
                population.tolist()
            )


        return np.asarray(
            sorted(
                set(neurons)
            ),
            dtype=np.int64,
        )


    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        prompt_keys = sorted(
            self.prompt_mapping.keys(),
            key=lambda x: (
                x[1],
                x[0],
            ),
        )


        feedback_keys = sorted(
            self.feedback_mapping.keys(),
            key=lambda x: (
                x[1],
                x[0],
            ),
        )


        np.savez(
            path,

            prompt_tokens=np.asarray([
                key[0]
                for key in prompt_keys
            ]),

            prompt_positions=np.asarray([
                key[1]
                for key in prompt_keys
            ]),

            prompt_neurons=np.stack([
                self.prompt_mapping[key]
                for key in prompt_keys
            ]),

            feedback_tokens=np.asarray([
                key[0]
                for key in feedback_keys
            ]),

            feedback_positions=np.asarray([
                key[1]
                for key in feedback_keys
            ]),

            feedback_neurons=np.stack([
                self.feedback_mapping[key]
                for key in feedback_keys
            ]),

            prompt_neurons_per_channel=(
                self.prompt_neurons_per_channel
            ),

            feedback_neurons_per_channel=(
                self.feedback_neurons_per_channel
            ),

            max_prompt_positions=(
                self.max_prompt_positions
            ),

            max_feedback_positions=(
                self.max_feedback_positions
            ),

            seed=self.seed,
        )


# ============================================================
# Factory
# ============================================================

def create_positional_feedback_map(
    prompt_tokens: list[str],
    feedback_tokens: list[str],
    neuron_pool: np.ndarray,

    max_prompt_positions: int = 6,

    max_feedback_positions: int = 7,

    prompt_neurons_per_channel: int = 16,

    feedback_neurons_per_channel: int = 16,

    seed: int = 123,

) -> PositionalFeedbackMap:

    prompt_tokens = sorted(
        set(prompt_tokens)
    )

    feedback_tokens = sorted(
        set(feedback_tokens)
    )

    neuron_pool = np.asarray(
        neuron_pool,
        dtype=np.int64,
    )


    # ========================================================
    # Required neuron count
    # ========================================================

    prompt_channels = (
        len(prompt_tokens)
        *
        max_prompt_positions
    )


    feedback_channels = (
        len(feedback_tokens)
        *
        max_feedback_positions
    )


    required_prompt_neurons = (
        prompt_channels
        *
        prompt_neurons_per_channel
    )


    required_feedback_neurons = (
        feedback_channels
        *
        feedback_neurons_per_channel
    )


    required_total = (
        required_prompt_neurons
        +
        required_feedback_neurons
    )


    if required_total > len(
        neuron_pool
    ):

        raise ValueError(
            f"Need {required_total} neurons "
            f"but pool contains only "
            f"{len(neuron_pool)}."
        )


    # ========================================================
    # Deterministic assignment
    # ========================================================

    rng = np.random.default_rng(
        seed
    )


    shuffled = rng.permutation(
        neuron_pool
    )


    cursor = 0


    # ========================================================
    # Prompt mapping
    # ========================================================

    prompt_mapping = {}


    for position in range(
        max_prompt_positions
    ):

        for token in prompt_tokens:

            end = (
                cursor
                +
                prompt_neurons_per_channel
            )


            prompt_mapping[
                (
                    token,
                    position,
                )
            ] = shuffled[
                cursor:end
            ].copy()


            cursor = end


    # ========================================================
    # Positional feedback mapping
    # ========================================================

    feedback_mapping = {}


    for position in range(
        max_feedback_positions
    ):

        for token in feedback_tokens:

            end = (
                cursor
                +
                feedback_neurons_per_channel
            )


            feedback_mapping[
                (
                    token,
                    position,
                )
            ] = shuffled[
                cursor:end
            ].copy()


            cursor = end


    return PositionalFeedbackMap(
        prompt_mapping=(
            prompt_mapping
        ),

        feedback_mapping=(
            feedback_mapping
        ),

        prompt_neurons_per_channel=(
            prompt_neurons_per_channel
        ),

        feedback_neurons_per_channel=(
            feedback_neurons_per_channel
        ),

        max_prompt_positions=(
            max_prompt_positions
        ),

        max_feedback_positions=(
            max_feedback_positions
        ),

        seed=seed,
    )