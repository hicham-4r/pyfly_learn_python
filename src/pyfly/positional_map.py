from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PositionalSensoryMap:
    """
    Maps:

        (token, position)

    to a deterministic population of fly neurons.

    Example:

        ("A", 0) -> neurons [...]
        ("A", 1) -> different neurons [...]
        ("B", 0) -> different neurons [...]
    """

    mapping: dict[tuple[str, int], np.ndarray]

    neurons_per_channel: int
    max_positions: int
    seed: int

    def neurons_for(
        self,
        token: str,
        position: int,
    ) -> np.ndarray:

        key = (
            token,
            position,
        )

        if key not in self.mapping:
            raise KeyError(
                f"No positional neural mapping "
                f"for token={token!r}, "
                f"position={position}"
            )

        return self.mapping[key]


    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        keys = sorted(
            self.mapping.keys(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )


        tokens = np.asarray(
            [
                token
                for token, position
                in keys
            ]
        )


        positions = np.asarray(
            [
                position
                for token, position
                in keys
            ],
            dtype=np.int32,
        )


        neurons = np.stack(
            [
                self.mapping[key]
                for key in keys
            ]
        )


        np.savez(
            path,

            tokens=tokens,

            positions=positions,

            neurons=neurons,

            neurons_per_channel=(
                self.neurons_per_channel
            ),

            max_positions=(
                self.max_positions
            ),

            seed=self.seed,
        )


def create_positional_sensory_map(
    tokens: list[str],
    neuron_pool: np.ndarray,
    max_positions: int,
    neurons_per_channel: int = 32,
    seed: int = 42,
) -> PositionalSensoryMap:

    """
    Create completely disjoint neural populations for
    every (token, position) pair.

    Example with:

        tokens = ["A", "B"]
        max_positions = 3

    creates:

        ("A", 0)
        ("B", 0)

        ("A", 1)
        ("B", 1)

        ("A", 2)
        ("B", 2)
    """

    if max_positions <= 0:
        raise ValueError(
            "max_positions must be > 0"
        )

    if neurons_per_channel <= 0:
        raise ValueError(
            "neurons_per_channel must be > 0"
        )


    tokens = sorted(
        set(tokens)
    )


    neuron_pool = np.asarray(
        neuron_pool,
        dtype=np.int64,
    )


    number_of_channels = (
        len(tokens)
        * max_positions
    )


    required_neurons = (
        number_of_channels
        * neurons_per_channel
    )


    if required_neurons > len(
        neuron_pool
    ):
        raise ValueError(
            f"Need {required_neurons} neurons "
            f"for positional encoding, but "
            f"the pool contains only "
            f"{len(neuron_pool)}."
        )


    rng = np.random.default_rng(
        seed
    )


    shuffled_pool = rng.permutation(
        neuron_pool
    )


    mapping: dict[
        tuple[str, int],
        np.ndarray,
    ] = {}


    cursor = 0


    # Position first, then token.
    # This makes the mapping structure easy to inspect.
    for position in range(
        max_positions
    ):

        for token in tokens:

            end = (
                cursor
                + neurons_per_channel
            )


            mapping[
                (
                    token,
                    position,
                )
            ] = shuffled_pool[
                cursor:end
            ].copy()


            cursor = end


    return PositionalSensoryMap(
        mapping=mapping,

        neurons_per_channel=(
            neurons_per_channel
        ),

        max_positions=(
            max_positions
        ),

        seed=seed,
    )