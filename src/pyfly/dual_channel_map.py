from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DualChannelMap:
    """
    Neural mapping for two different information streams:

    1. Prompt tokens:
       (token, position) -> neuron population

    2. Generated-code feedback tokens:
       code_token -> separate neuron population
    """

    prompt_mapping: dict[tuple[str, int], np.ndarray]

    feedback_mapping: dict[str, np.ndarray]

    prompt_neurons_per_channel: int
    feedback_neurons_per_token: int

    max_prompt_positions: int

    seed: int


    # ========================================================
    # Prompt neurons
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
                f"token={token!r}, "
                f"position={position}"
            )

        return self.prompt_mapping[
            key
        ]


    # ========================================================
    # Feedback neurons
    # ========================================================

    def feedback_neurons_for(
        self,
        token: str,
    ) -> np.ndarray:

        if token not in self.feedback_mapping:

            raise KeyError(
                f"No feedback mapping "
                f"for code token={token!r}"
            )

        return self.feedback_mapping[
            token
        ]


    # ========================================================
    # All input neurons
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
    # Save mapping
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
            key=lambda item: (
                item[1],
                item[0],
            ),
        )


        prompt_tokens = np.asarray([
            token
            for token, position
            in prompt_keys
        ])


        prompt_positions = np.asarray([
            position
            for token, position
            in prompt_keys
        ])


        prompt_neurons = np.stack([
            self.prompt_mapping[key]
            for key in prompt_keys
        ])


        feedback_tokens = sorted(
            self.feedback_mapping.keys()
        )


        feedback_neurons = np.stack([
            self.feedback_mapping[token]
            for token in feedback_tokens
        ])


        np.savez(
            path,

            prompt_tokens=prompt_tokens,

            prompt_positions=(
                prompt_positions
            ),

            prompt_neurons=(
                prompt_neurons
            ),

            feedback_tokens=np.asarray(
                feedback_tokens
            ),

            feedback_neurons=(
                feedback_neurons
            ),

            prompt_neurons_per_channel=(
                self.prompt_neurons_per_channel
            ),

            feedback_neurons_per_token=(
                self.feedback_neurons_per_token
            ),

            max_prompt_positions=(
                self.max_prompt_positions
            ),

            seed=self.seed,
        )


# ============================================================
# Factory
# ============================================================

def create_dual_channel_map(
    prompt_tokens: list[str],
    feedback_tokens: list[str],
    neuron_pool: np.ndarray,
    max_prompt_positions: int = 6,
    prompt_neurons_per_channel: int = 16,
    feedback_neurons_per_token: int = 32,
    seed: int = 123,
) -> DualChannelMap:

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
    # Number of neural channels
    # ========================================================

    prompt_channel_count = (
        len(prompt_tokens)
        *
        max_prompt_positions
    )


    prompt_neuron_count = (
        prompt_channel_count
        *
        prompt_neurons_per_channel
    )


    feedback_neuron_count = (
        len(feedback_tokens)
        *
        feedback_neurons_per_token
    )


    required_neurons = (
        prompt_neuron_count
        +
        feedback_neuron_count
    )


    if required_neurons > len(
        neuron_pool
    ):

        raise ValueError(
            f"Dual encoder requires "
            f"{required_neurons} neurons, "
            f"but pool contains only "
            f"{len(neuron_pool)}."
        )


    # ========================================================
    # Deterministic shuffled pool
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
    # Feedback mapping
    # ========================================================

    feedback_mapping = {}


    for token in feedback_tokens:

        end = (
            cursor
            +
            feedback_neurons_per_token
        )


        feedback_mapping[
            token
        ] = shuffled[
            cursor:end
        ].copy()


        cursor = end


    return DualChannelMap(
        prompt_mapping=prompt_mapping,

        feedback_mapping=feedback_mapping,

        prompt_neurons_per_channel=(
            prompt_neurons_per_channel
        ),

        feedback_neurons_per_token=(
            feedback_neurons_per_token
        ),

        max_prompt_positions=(
            max_prompt_positions
        ),

        seed=seed,
    )