from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl


# ============================================================
# Step 78B-1
# DEGREE-PRESERVING RANDOM-REWIRED RESERVOIR FEATURES
#
# Scientific control:
#   Same FlyBrain neuron count
#   Same number of recurrent edges
#   Same presynaptic neurons
#   Same synaptic weights
#   Same global in-degree sequence
#   Same PyFly encoder / mapping / timings
#   Same indirect visual_projection readout
#
# Difference:
#   Postsynaptic targets are shuffled in memory before any
#   PyFly episodes are run.
#
# This follows the degree-preserving scramble used by the
# fly.ai repository's own flytalk.py control.
#
# IMPORTANT:
#   * Original MaleCNS files on disk are NEVER modified.
#   * The random rewiring exists only in this process.
#   * No final-holdout results are used for tuning.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
)

GENERATOR_FILES = {
    "train": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_train.jsonl"
    ),

    "validation": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_validation.jsonl"
    ),

    "test": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_test.jsonl"
    ),

    "final_holdout": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_final_holdout.jsonl"
    ),
}

OUTPUT_FILES = {
    split_name: (
        OUTPUT_DIR
        /
        f"generator_random_rewired_{split_name}.npz"
    )

    for split_name
    in GENERATOR_FILES
}

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
)

FROZEN_MAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)

REFERENCE_FEATURE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_train.npz"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_reservoir_feature_manifest.json"
)


# ============================================================
# Frozen Step-62 PyFly configuration
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

FEATURE_BLOCK_NAMES = [
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
]


# ============================================================
# Random wiring seed
#
# This is chosen BEFORE running the random control.
# Do not change it after seeing results.
# ============================================================

REWIRE_SEED = 78001


# ============================================================
# Expected frozen project sizes
# ============================================================

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873
EXPECTED_FEATURE_DIMENSIONS = 41_111

EXPECTED_ROWS = {
    "train": 1_140,
    "validation": 190,
    "test": 190,
    "final_holdout": 760,
}

EXPECTED_LOGICAL_PROMPTS = {
    "train": 36,
    "validation": 6,
    "test": 6,
    "final_holdout": 24,
}


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


def normalized_relative(
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


def frozen_hash_for(
    protocol: dict,
    path: Path,
) -> str:

    wanted = normalized_relative(
        path
    )

    hashes = protocol.get(
        "frozen_file_hashes",
        {},
    )

    for stored_path, info in hashes.items():

        normalized = (
            stored_path.replace(
                "\\",
                "/",
            )
        )

        if normalized == wanted:

            value = info.get(
                "sha256"
            )

            if not value:

                raise ValueError(
                    "Protocol contains no "
                    f"SHA-256 for {wanted}"
                )

            return str(
                value
            )

    raise ValueError(
        "Protocol contains no frozen "
        f"hash entry for {wanted}"
    )


# ============================================================
# Verify final-evaluation integrity
# ============================================================

def verify_protocol() -> dict:

    required_files = [
        PROTOCOL_FILE,
        FINAL_HOLDOUT_FILE,
        FROZEN_MAP_FILE,
        *GENERATOR_FILES.values(),
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file is missing: "
                f"{path}"
            )

    protocol = load_json(
        PROTOCOL_FILE
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Final-evaluation protocol is "
            "not marked frozen."
        )

    architecture = protocol.get(
        "architecture",
        {},
    )

    expected = {
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

        "mapping_seed": (
            MAP_SEED
        ),

        "neural_seeds": (
            NEURAL_SEEDS
        ),

        "feature_blocks": (
            FEATURE_BLOCK_NAMES
        ),
    }

    for key, wanted in expected.items():

        actual = architecture.get(
            key
        )

        if actual != wanted:

            raise ValueError(
                "Frozen protocol mismatch "
                f"for {key}: expected "
                f"{wanted!r}, got "
                f"{actual!r}"
            )

    if not np.isclose(
        float(
            architecture.get(
                "amplitude"
            )
        ),
        AMPLITUDE,
    ):

        raise ValueError(
            "Frozen amplitude changed."
        )

    if not np.isclose(
        float(
            architecture.get(
                "trace_tau"
            )
        ),
        TRACE_TAU,
    ):

        raise ValueError(
            "Frozen trace tau changed."
        )

    # --------------------------------------------------------
    # Final holdout and neural mapping must remain identical.
    # --------------------------------------------------------

    for path in [
        FINAL_HOLDOUT_FILE,
        FROZEN_MAP_FILE,
    ]:

        expected_hash = (
            frozen_hash_for(
                protocol,
                path,
            )
        )

        current_hash = (
            sha256_file(
                path
            )
        )

        if (
            current_hash
            !=
            expected_hash
        ):

            raise ValueError(
                "Frozen-file integrity "
                f"failure for "
                f"{normalized_relative(path)}"
            )

    return protocol


# ============================================================
# Load the EXACT frozen PyFly neural mapping
# ============================================================

def load_frozen_dual_map(
    path: Path,
) -> DualChannelMap:

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "prompt_tokens",
        "prompt_positions",
        "prompt_neurons",
        "feedback_tokens",
        "feedback_neurons",
        "prompt_neurons_per_channel",
        "feedback_neurons_per_token",
        "max_prompt_positions",
        "seed",
    }

    missing = (
        required
        -
        set(
            data.files
        )
    )

    if missing:

        raise ValueError(
            "Frozen map is missing "
            f"arrays: {sorted(missing)}"
        )

    prompt_tokens = (
        data[
            "prompt_tokens"
        ].astype(str)
    )

    prompt_positions = (
        data[
            "prompt_positions"
        ].astype(
            np.int64
        )
    )

    prompt_neurons = (
        data[
            "prompt_neurons"
        ].astype(
            np.int64
        )
    )

    feedback_tokens = (
        data[
            "feedback_tokens"
        ].astype(str)
    )

    feedback_neurons = (
        data[
            "feedback_neurons"
        ].astype(
            np.int64
        )
    )

    prompt_mapping = {}

    for row, (
        token,
        position,
    ) in enumerate(
        zip(
            prompt_tokens,
            prompt_positions,
        )
    ):

        key = (
            str(
                token
            ),
            int(
                position
            ),
        )

        if key in prompt_mapping:

            raise ValueError(
                f"Duplicate prompt key: "
                f"{key}"
            )

        prompt_mapping[
            key
        ] = (
            prompt_neurons[
                row
            ].copy()
        )

    feedback_mapping = {}

    for row, token in enumerate(
        feedback_tokens
    ):

        token = str(
            token
        )

        if token in feedback_mapping:

            raise ValueError(
                "Duplicate feedback token: "
                f"{token}"
            )

        feedback_mapping[
            token
        ] = (
            feedback_neurons[
                row
            ].copy()
        )

    dual_map = DualChannelMap(
        prompt_mapping=(
            prompt_mapping
        ),

        feedback_mapping=(
            feedback_mapping
        ),

        prompt_neurons_per_channel=int(
            np.asarray(
                data[
                    "prompt_neurons_per_channel"
                ]
            ).item()
        ),

        feedback_neurons_per_token=int(
            np.asarray(
                data[
                    "feedback_neurons_per_token"
                ]
            ).item()
        ),

        max_prompt_positions=int(
            np.asarray(
                data[
                    "max_prompt_positions"
                ]
            ).item()
        ),

        seed=int(
            np.asarray(
                data[
                    "seed"
                ]
            ).item()
        ),
    )

    if (
        dual_map.prompt_neurons_per_channel
        !=
        PROMPT_NEURONS_PER_CHANNEL
    ):

        raise ValueError(
            "Prompt mapping width changed."
        )

    if (
        dual_map.feedback_neurons_per_token
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):

        raise ValueError(
            "Feedback mapping width changed."
        )

    if (
        dual_map.max_prompt_positions
        !=
        MAX_PROMPT_POSITIONS
    ):

        raise ValueError(
            "Frozen prompt maximum changed."
        )

    if (
        dual_map.seed
        !=
        MAP_SEED
    ):

        raise ValueError(
            "Frozen neural mapping seed "
            "changed."
        )

    return dual_map


# ============================================================
# Degree-preserving rewiring
#
# Adapted from fly.ai/flytalk.py:
#
#   brain.indices = rng.permutation(brain.indices)
#
# The CSC column pointers and weights stay fixed.
#
# Therefore:
#   * every presynaptic neuron keeps its outgoing-edge count
#   * every synaptic weight stays in the same presynaptic column
#   * every postsynaptic neuron keeps the same global in-degree
#   * who connects to whom is randomized
# ============================================================

def degree_preserving_rewire(
    brain: FlyBrain,
    seed: int,
) -> None:

    rng = np.random.default_rng(
        seed
    )

    original_indices = (
        brain.indices.copy()
    )

    original_weights = (
        brain.weights.copy()
    )

    original_indptr = (
        brain.indptr.copy()
    )

    # --------------------------------------------------------
    # Shuffle ONLY postsynaptic target indices.
    # --------------------------------------------------------

    brain.indices = (
        rng.permutation(
            brain.indices
        ).astype(
            brain.indices.dtype,
            copy=False,
        )
    )

    # --------------------------------------------------------
    # Basic structural invariants
    # --------------------------------------------------------

    if not np.array_equal(
        brain.weights,
        original_weights,
    ):

        raise RuntimeError(
            "Rewiring unexpectedly changed "
            "synaptic weights."
        )

    if not np.array_equal(
        brain.indptr,
        original_indptr,
    ):

        raise RuntimeError(
            "Rewiring unexpectedly changed "
            "CSC column pointers."
        )

    if not np.array_equal(
        np.sort(
            brain.indices.astype(
                np.int64
            )
        ),
        np.sort(
            original_indices.astype(
                np.int64
            )
        ),
    ):

        raise RuntimeError(
            "Rewiring failed to preserve "
            "the postsynaptic index multiset."
        )

    if np.array_equal(
        brain.indices,
        original_indices,
    ):

        raise RuntimeError(
            "Random rewiring produced the "
            "original connectivity ordering."
        )

    # --------------------------------------------------------
    # Rebuild GPU sparse matrix exactly as fly.ai's
    # degree-preserving control does.
    # --------------------------------------------------------

    if brain.device == "cuda":

        from cupyx.scipy import sparse as cusparse

        W = sparse.csc_matrix(
            (
                brain.weights,
                brain.indices,
                brain.indptr,
            ),
            shape=(
                brain.n,
                brain.n,
            ),
        )

        W.sum_duplicates()

        brain._W = cusparse.csr_matrix(
            W.tocsr().astype(
                np.float32
            )
        )


# ============================================================
# Group teacher-forcing rows by logical prompt
# ============================================================

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
            key=lambda sample:
                sample.target_position
        )

        positions = [
            sample.target_position

            for sample
            in grouped[
                example_id
            ]
        ]

        expected = list(
            range(
                len(
                    positions
                )
            )
        )

        if (
            positions
            !=
            expected
        ):

            raise ValueError(
                "Non-contiguous target "
                f"positions for {example_id}: "
                f"{positions}"
            )

    return grouped


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        " PyFly STEP 78B-1 - RANDOM-REWIRED FEATURES"
    )

    print(
        "===================================================="
    )

    print()

    protocol = verify_protocol()

    print(
        "Frozen PyFly protocol:       PASS"
    )

    print(
        "Frozen final holdout:        PASS"
    )

    print(
        "Frozen encoder mapping:      PASS"
    )

    print(
        f"Random rewiring seed:        "
        f"{REWIRE_SEED}"
    )

    # ========================================================
    # Load all generator datasets
    # ========================================================

    generator_samples = {}

    for split_name, path in (
        GENERATOR_FILES.items()
    ):

        samples = (
            load_generator_jsonl(
                path
            )
        )

        generator_samples[
            split_name
        ] = samples

        print(
            f"{split_name:14} "
            f"teacher rows="
            f"{len(samples):3}"
        )

    # ========================================================
    # Load exact frozen PyFly map
    # ========================================================

    dual_map = (
        load_frozen_dual_map(
            FROZEN_MAP_FILE
        )
    )

    # ========================================================
    # Verify all data can be represented by frozen map
    # ========================================================

    for split_name, samples in (
        generator_samples.items()
    ):

        grouped = group_by_example(
            samples
        )

        wanted_logical = (
            EXPECTED_LOGICAL_PROMPTS[
                split_name
            ]
        )

        if (
            len(
                grouped
            )
            !=
            wanted_logical
        ):

            raise ValueError(
                f"{split_name}: expected "
                f"{wanted_logical} logical "
                f"prompts, got "
                f"{len(grouped)}"
            )

        for (
            example_id,
            example_samples,
        ) in grouped.items():

            prompt_tokens = (
                example_samples[
                    0
                ].prompt_tokens
            )

            for (
                position,
                token,
            ) in enumerate(
                prompt_tokens
            ):

                dual_map.prompt_neurons_for(
                    token,
                    position,
                )

            for (
                position,
                sample,
            ) in enumerate(
                example_samples
            ):

                feedback_token = (
                    "BOS"
                    if position == 0
                    else
                    example_samples[
                        position - 1
                    ].target_token
                )

                dual_map.feedback_neurons_for(
                    feedback_token
                )

                if (
                    sample.context_tokens[
                        -1
                    ]
                    !=
                    feedback_token
                ):

                    raise ValueError(
                        "Teacher-forcing "
                        f"context mismatch for "
                        f"{sample.sample_id}"
                    )

    # ========================================================
    # Load real FlyBrain, then scramble connectivity IN MEMORY
    # ========================================================

    print()

    print(
        "Loading FlyBrain on GPU..."
    )

    brain = FlyBrain(
        device=DEVICE,
        seed=0,
    )

    if (
        brain.n
        !=
        EXPECTED_NEURONS
    ):

        raise ValueError(
            f"Expected {EXPECTED_NEURONS:,} "
            f"neurons, got {brain.n:,}"
        )

    if (
        len(
            brain.weights
        )
        !=
        EXPECTED_CONNECTIONS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_CONNECTIONS:,} "
            "connections, got "
            f"{len(brain.weights):,}"
        )

    print(
        f"Original neurons:            "
        f"{brain.n:,}"
    )

    print(
        f"Original connections:        "
        f"{len(brain.weights):,}"
    )

    print()

    print(
        "Applying degree-preserving "
        "random rewiring..."
    )

    degree_preserving_rewire(
        brain,
        REWIRE_SEED,
    )

    print(
        "Degree-preserving rewiring:  PASS"
    )

    print(
        "Original brain files changed:NO"
    )

    # ========================================================
    # Frozen PyFly input/readout populations
    # ========================================================

    visual_pool = np.asarray(
        brain.cells(
            [
                "visual_projection"
            ]
        ),
        dtype=np.int64,
    )

    input_neurons = (
        dual_map.all_input_neurons()
    )

    if not np.all(
        np.isin(
            input_neurons,
            visual_pool,
        )
    ):

        raise ValueError(
            "Frozen input map contains "
            "neurons outside "
            "visual_projection."
        )

    readout_population = np.setdiff1d(
        visual_pool,
        input_neurons,
    )

    if (
        len(
            input_neurons
        )
        !=
        EXPECTED_DIRECT_INPUT_NEURONS
    ):

        raise ValueError(
            "Direct-input population "
            f"changed: "
            f"{len(input_neurons)}"
        )

    if (
        len(
            readout_population
        )
        !=
        EXPECTED_READOUT_NEURONS
    ):

        raise ValueError(
            "Indirect-readout population "
            f"changed: "
            f"{len(readout_population)}"
        )

    if len(
        np.intersect1d(
            input_neurons,
            readout_population,
        )
    ) != 0:

        raise RuntimeError(
            "Direct input neurons leaked "
            "into random-reservoir readout."
        )

    # ========================================================
    # Strong compatibility check with real Step-62 features
    # ========================================================

    if REFERENCE_FEATURE_FILE.exists():

        reference = np.load(
            REFERENCE_FEATURE_FILE,
            allow_pickle=False,
        )

        if not np.array_equal(
            reference[
                "input_neurons"
            ].astype(
                np.int64
            ),
            input_neurons,
        ):

            raise ValueError(
                "Frozen input population "
                "does not match Step-62."
            )

        if not np.array_equal(
            reference[
                "readout_population"
            ].astype(
                np.int64
            ),
            readout_population,
        ):

            raise ValueError(
                "Frozen readout population "
                "does not match Step-62."
            )

    feature_dimensions = (
        len(
            FEATURE_BLOCK_NAMES
        )
        *
        len(
            readout_population
        )
    )

    if (
        feature_dimensions
        !=
        EXPECTED_FEATURE_DIMENSIONS
    ):

        raise ValueError(
            "Feature dimensions changed: "
            f"{feature_dimensions}"
        )

    print()

    print(
        f"Direct input neurons:        "
        f"{len(input_neurons):,}"
    )

    print(
        f"Indirect readout neurons:    "
        f"{len(readout_population):,}"
    )

    print(
        f"Feature blocks:              "
        f"{len(FEATURE_BLOCK_NAMES)}"
    )

    print(
        f"Feature dimensions:          "
        f"{feature_dimensions:,}"
    )

    # ========================================================
    # Trace recorder on same indirect visual population
    # ========================================================

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    # ========================================================
    # Frozen Step-62 prompt stimulation
    # ========================================================

    def stimulate_prompt(
        prompt_tokens,
    ):

        states = []

        for (
            position,
            token,
        ) in enumerate(
            prompt_tokens
        ):

            neurons = (
                dual_map.prompt_neurons_for(
                    token,
                    position,
                )
            )

            for _ in range(
                PROMPT_TOKEN_STEPS
            ):

                fired = brain.step(
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ]
                )

                state = np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()

                states.append(
                    state
                )

            for _ in range(
                PROMPT_GAP_STEPS
            ):

                fired = brain.step()

                state = np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()

                states.append(
                    state
                )

        states = np.stack(
            states
        )

        return (
            states.mean(
                axis=0
            ),

            states.max(
                axis=0
            ),

            states[
                -1
            ],
        )

    # ========================================================
    # Frozen Step-62 feedback stimulation
    # ========================================================

    def stimulate_feedback(
        token,
    ):

        neurons = (
            dual_map.feedback_neurons_for(
                token
            )
        )

        states = []

        for _ in range(
            FEEDBACK_TOKEN_STEPS
        ):

            fired = brain.step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ]
            )

            state = np.asarray(
                trace.observe(
                    fired
                )
            ).copy()

            states.append(
                state
            )

        states = np.stack(
            states
        )

        feedback_mean = (
            states.mean(
                axis=0
            )
        )

        feedback_max = (
            states.max(
                axis=0
            )
        )

        feedback_final = (
            states[
                -1
            ]
        )

        delayed_state = (
            feedback_final
        )

        for _ in range(
            FEEDBACK_GAP_STEPS
        ):

            fired = brain.step()

            delayed_state = (
                np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()
            )

        return (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        )

    # ========================================================
    # Run one complete teacher-forcing episode
    # ========================================================

    def run_episode(
        example_samples,
        neural_seed,
    ):

        brain.reset(
            seed=neural_seed
        )

        trace.reset()

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        for _ in range(
            WARMUP_STEPS
        ):

            fired = brain.step()

            trace.observe(
                fired
            )

        # ----------------------------------------------------
        # Prompt once
        # ----------------------------------------------------

        prompt_tokens = (
            example_samples[
                0
            ].prompt_tokens
        )

        (
            prompt_mean,
            prompt_max,
            prompt_final,
        ) = stimulate_prompt(
            prompt_tokens
        )

        rows = []

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        for (
            position,
            sample,
        ) in enumerate(
            example_samples
        ):

            feedback_token = (
                "BOS"
                if position == 0
                else
                example_samples[
                    position - 1
                ].target_token
            )

            (
                feedback_mean,
                feedback_max,
                feedback_final,
                delayed_state,
            ) = stimulate_feedback(
                feedback_token
            )

            feature = np.concatenate(
                [
                    prompt_mean,
                    prompt_max,
                    prompt_final,
                    feedback_mean,
                    feedback_max,
                    feedback_final,
                    delayed_state,
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            if (
                feature.shape
                !=
                (
                    feature_dimensions,
                )
            ):

                raise RuntimeError(
                    "Random-reservoir "
                    "feature shape mismatch: "
                    f"{feature.shape}"
                )

            if not np.isfinite(
                feature
            ).all():

                raise RuntimeError(
                    "Non-finite random "
                    "reservoir feature."
                )

            rows.append(
                (
                    feature,
                    sample,
                    feedback_token,
                )
            )

        return rows

    # ========================================================
    # Extract one split
    # ========================================================

    def extract_split(
        split_name,
    ):

        samples = (
            generator_samples[
                split_name
            ]
        )

        grouped = (
            group_by_example(
                samples
            )
        )

        features = []
        targets = []
        sample_ids = []
        example_ids = []
        intents = []
        positions = []
        feedback_used = []
        seeds_used = []

        logical_prompts = sorted(
            grouped.keys()
        )

        total_episodes = (
            len(
                logical_prompts
            )
            *
            len(
                NEURAL_SEEDS
            )
        )

        print()

        print(
            "===================================================="
        )

        print(
            f" RANDOM-REWIRED "
            f"{split_name.upper()} EXTRACTION"
        )

        print(
            "===================================================="
        )

        print()

        episode_number = 0

        for example_id in logical_prompts:

            example_samples = (
                grouped[
                    example_id
                ]
            )

            for neural_seed in (
                NEURAL_SEEDS
            ):

                episode_number += 1

                print(
                    f"[{episode_number:03}/"
                    f"{total_episodes:03}] "
                    f"{example_id:20} "
                    f"neural_seed="
                    f"{neural_seed} "
                    f"steps="
                    f"{len(example_samples)}"
                )

                rows = run_episode(
                    example_samples,
                    neural_seed,
                )

                for (
                    feature,
                    sample,
                    feedback_token,
                ) in rows:

                    features.append(
                        feature
                    )

                    targets.append(
                        sample.target_token
                    )

                    sample_ids.append(
                        sample.sample_id
                    )

                    example_ids.append(
                        sample.example_id
                    )

                    intents.append(
                        sample.intent
                    )

                    positions.append(
                        sample.target_position
                    )

                    feedback_used.append(
                        feedback_token
                    )

                    seeds_used.append(
                        neural_seed
                    )

        X = np.stack(
            features
        ).astype(
            np.float32,
            copy=False,
        )

        targets = np.asarray(
            targets
        )

        sample_ids = np.asarray(
            sample_ids
        )

        example_ids = np.asarray(
            example_ids
        )

        intents = np.asarray(
            intents
        )

        positions = np.asarray(
            positions,
            dtype=np.int32,
        )

        feedback_used = np.asarray(
            feedback_used
        )

        seeds_used = np.asarray(
            seeds_used,
            dtype=np.int32,
        )

        if (
            len(
                X
            )
            !=
            EXPECTED_ROWS[
                split_name
            ]
        ):

            raise ValueError(
                f"{split_name}: expected "
                f"{EXPECTED_ROWS[split_name]} "
                f"rows, got {len(X)}"
            )

        if (
            X.shape[
                1
            ]
            !=
            EXPECTED_FEATURE_DIMENSIONS
        ):

            raise ValueError(
                f"{split_name}: unexpected "
                "feature dimensions "
                f"{X.shape[1]}"
            )

        if not np.isfinite(
            X
        ).all():

            raise RuntimeError(
                f"{split_name}: non-finite "
                "features found."
            )

        output_file = (
            OUTPUT_FILES[
                split_name
            ]
        )

        np.savez(
            output_file,

            X=X,

            targets=targets,

            sample_ids=sample_ids,

            example_ids=example_ids,

            intents=intents,

            target_positions=positions,

            feedback_tokens=feedback_used,

            seeds=seeds_used,

            input_neurons=(
                input_neurons
            ),

            readout_population=(
                readout_population
            ),

            feature_block_names=np.asarray(
                FEATURE_BLOCK_NAMES
            ),

            rewire_seed=np.asarray(
                REWIRE_SEED,
                dtype=np.int64,
            ),
        )

        print()

        print(
            f"Logical prompts:             "
            f"{len(logical_prompts)}"
        )

        print(
            f"Neural episodes:             "
            f"{total_episodes}"
        )

        print(
            f"Neural rows:                 "
            f"{len(X)}"
        )

        print(
            f"Feature dimensions:          "
            f"{X.shape[1]:,}"
        )

        print(
            f"Finite:                      "
            f"{np.isfinite(X).all()}"
        )

        print(
            f"Saved:                       "
            f"{normalized_relative(output_file)}"
        )

        return {
            "logical_prompts": int(
                len(
                    logical_prompts
                )
            ),

            "neural_episodes": int(
                total_episodes
            ),

            "neural_rows": int(
                len(
                    X
                )
            ),

            "feature_dimensions": int(
                X.shape[
                    1
                ]
            ),

            "file": (
                normalized_relative(
                    output_file
                )
            ),

            "sha256": (
                sha256_file(
                    output_file
                )
            ),
        }

    # ========================================================
    # Extract development + frozen final holdout
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_manifests = {}

    for split_name in [
        "train",
        "validation",
        "test",
        "final_holdout",
    ]:

        split_manifests[
            split_name
        ] = (
            extract_split(
                split_name
            )
        )

    # ========================================================
    # Save control manifest
    # ========================================================

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "step": (
            "78B-1"
        ),

        "control": (
            "degree_preserving_random_rewired_reservoir"
        ),

        "rewire_seed": int(
            REWIRE_SEED
        ),

        "device": (
            DEVICE
        ),

        "neurons": int(
            brain.n
        ),

        "connections": int(
            len(
                brain.weights
            )
        ),

        "direct_input_neurons": int(
            len(
                input_neurons
            )
        ),

        "indirect_readout_neurons": int(
            len(
                readout_population
            )
        ),

        "feature_blocks": (
            FEATURE_BLOCK_NAMES
        ),

        "feature_dimensions": int(
            feature_dimensions
        ),

        "neural_seeds": (
            NEURAL_SEEDS
        ),

        "mapping_sha256": (
            sha256_file(
                FROZEN_MAP_FILE
            )
        ),

        "final_holdout_sha256": (
            sha256_file(
                FINAL_HOLDOUT_FILE
            )
        ),

        "splits": (
            split_manifests
        ),

        "rewiring_definition": {
            "presynaptic_edge_counts_preserved": (
                True
            ),

            "synaptic_weights_preserved": (
                True
            ),

            "postsynaptic_index_multiset_preserved": (
                True
            ),

            "postsynaptic_targets_shuffled": (
                True
            ),

            "original_files_modified": (
                False
            ),
        },
    }

    METADATA_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "===================================================="
    )

    print(
        "              STEP 78B-1 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Random rewiring seed:        "
        f"{REWIRE_SEED}"
    )

    print(
        f"Feature dimensions:          "
        f"{feature_dimensions:,}"
    )

    print()

    print(
        "Created:"
    )

    for split_name in [
        "train",
        "validation",
        "test",
        "final_holdout",
    ]:

        print(
            "  "
            f"{normalized_relative(OUTPUT_FILES[split_name])}"
        )

    print(
        "  "
        f"{normalized_relative(METADATA_FILE)}"
    )

    print()

    print(
        "Next: Step 78B-2 - select the "
        "random-reservoir readout on development "
        "validation only, refit on all development "
        "prompts, then score the frozen final holdout."
    )


if __name__ == "__main__":
    main()
