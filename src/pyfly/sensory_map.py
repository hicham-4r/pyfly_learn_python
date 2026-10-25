from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SensoryMap:
    token_to_neurons: dict[str, np.ndarray]
    neurons_per_token: int
    seed: int

    def neurons_for(self, token: str) -> np.ndarray:
        if token not in self.token_to_neurons:
            raise KeyError(f"No neural mapping for token: {token}")

        return self.token_to_neurons[token]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tokens = list(self.token_to_neurons.keys())

        neuron_matrix = np.stack(
            [self.token_to_neurons[token] for token in tokens]
        )

        np.savez(
            path,
            tokens=np.array(tokens),
            neurons=neuron_matrix,
            neurons_per_token=self.neurons_per_token,
            seed=self.seed,
        )


def create_disjoint_sensory_map(
    tokens: list[str],
    neuron_pool: np.ndarray,
    neurons_per_token: int = 24,
    seed: int = 42,
) -> SensoryMap:

    tokens = sorted(set(tokens))
    neuron_pool = np.asarray(neuron_pool, dtype=np.int64)

    required_neurons = len(tokens) * neurons_per_token

    if required_neurons > len(neuron_pool):
        raise ValueError(
            f"Need {required_neurons} neurons but pool only has "
            f"{len(neuron_pool)}"
        )

    rng = np.random.default_rng(seed)

    shuffled_pool = rng.permutation(neuron_pool)

    token_to_neurons: dict[str, np.ndarray] = {}

    start = 0

    for token in tokens:
        end = start + neurons_per_token

        token_to_neurons[token] = shuffled_pool[start:end].copy()

        start = end

    return SensoryMap(
        token_to_neurons=token_to_neurons,
        neurons_per_token=neurons_per_token,
        seed=seed,
    )