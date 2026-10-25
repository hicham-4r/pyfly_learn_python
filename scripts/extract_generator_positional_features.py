from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.generator_dataset import load_generator_jsonl
from pyfly.positional_feedback_map import (
    create_positional_feedback_map,
)
from pyfly.tokenizer import build_code_vocabulary


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

OUTPUT_DIR = Path(
    "artifacts/features"
)

GENERATOR_FILES = {
    "train": (
        "data/processed/"
        "generator_train.jsonl"
    ),

    "validation": (
        "data/processed/"
        "generator_validation.jsonl"
    ),

    "test": (
        "data/processed/"
        "generator_test.jsonl"
    ),
}


# ============================================================
# Neural seeds
# ============================================================

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


# ============================================================
# Prompt channel
# ============================================================

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS_PER_CHANNEL = 16

PROMPT_TOKEN_STEPS = 8

PROMPT_GAP_STEPS = 2


# ============================================================
# Positional feedback channel
# ============================================================

MAX_FEEDBACK_POSITIONS = 7

FEEDBACK_NEURONS_PER_CHANNEL = 16

FEEDBACK_TOKEN_STEPS = 8

FEEDBACK_GAP_STEPS = 1


# ============================================================
# Reservoir
# ============================================================

WARMUP_STEPS = 10

AMPLITUDE = 0.8

TRACE_TAU = 0.25

MAP_SEED = 123


# ============================================================
# Header
# ============================================================

print()
print(
    "================================================"
)

print(
    " PyFly POSITIONAL FEEDBACK FEATURE EXTRACTION"
)

print(
    "================================================"
)

print()


# ============================================================
# Load generator datasets
# ============================================================

generator_samples = {}


for split_name, path in (
    GENERATOR_FILES.items()
):

    samples = load_generator_jsonl(
        path
    )

    generator_samples[
        split_name
    ] = samples

    print(
        f"{split_name:12} "
        f"samples: "
        f"{len(samples)}"
    )


# ============================================================
# Prompt vocabulary
# ============================================================

prompt_tokens = sorted({
    token

    for samples in (
        generator_samples.values()
    )

    for sample in samples

    for token in sample.prompt_tokens
})


print()
print(
    f"Prompt vocabulary: "
    f"{len(prompt_tokens)}"
)


# ============================================================
# Feedback vocabulary
# ============================================================

code_vocab = (
    build_code_vocabulary()
)


feedback_tokens = sorted([
    token

    for token in (
        code_vocab.token_to_id
    )

    if token != "PAD"
])


print(
    f"Feedback vocabulary: "
    f"{len(feedback_tokens)}"
)


# ============================================================
# Load FlyBrain
# ============================================================

print()
print(
    "Loading FlyBrain on GPU..."
)


brain = FlyBrain(
    device=DEVICE,
    seed=0,
)


print(
    "FlyBrain loaded."
)

print(
    f"Neurons: "
    f"{brain.n:,}"
)

print(
    f"Connections: "
    f"{len(brain.weights):,}"
)


# ============================================================
# Visual projection pool
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print()
print(
    f"Visual-projection pool: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Build positional prompt + feedback mapping
# ============================================================

mapping = (
    create_positional_feedback_map(
        prompt_tokens=prompt_tokens,

        feedback_tokens=feedback_tokens,

        neuron_pool=visual_pool,

        max_prompt_positions=(
            MAX_PROMPT_POSITIONS
        ),

        max_feedback_positions=(
            MAX_FEEDBACK_POSITIONS
        ),

        prompt_neurons_per_channel=(
            PROMPT_NEURONS_PER_CHANNEL
        ),

        feedback_neurons_per_channel=(
            FEEDBACK_NEURONS_PER_CHANNEL
        ),

        seed=MAP_SEED,
    )
)


# ============================================================
# Remove all directly stimulated neurons from readout
# ============================================================

input_neurons = (
    mapping.all_input_neurons()
)


readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()
print(
    f"Direct input neurons: "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout neurons: "
    f"{len(readout_population):,}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


if len(readout_population) == 0:

    raise RuntimeError(
        "No indirect visual neurons remain."
    )


# ============================================================
# Save exact map
# ============================================================

mapping.save(
    "data/mappings/"
    "generator_positional_feedback_map.npz"
)


# ============================================================
# Trace
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Group teacher-forcing samples by logical prompt
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


    return grouped


# ============================================================
# Stimulate prompt
# ============================================================

def stimulate_prompt(
    tokens,
):

    states = []


    for position, token in enumerate(
        tokens
    ):

        if position >= (
            MAX_PROMPT_POSITIONS
        ):

            raise ValueError(
                f"Prompt too long: "
                f"{tokens}"
            )


        neurons = (
            mapping.prompt_neurons_for(
                token,
                position,
            )
        )


        # ----------------------------------------------------
        # Prompt token
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Prompt gap
        # ----------------------------------------------------

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


    prompt_mean = states.mean(
        axis=0
    )

    prompt_max = states.max(
        axis=0
    )

    prompt_final = states[
        -1
    ]


    return (
        prompt_mean,
        prompt_max,
        prompt_final,
    )


# ============================================================
# Stimulate positional feedback
# ============================================================

def stimulate_feedback(
    token,
    feedback_position,
):

    if (
        feedback_position
        >= MAX_FEEDBACK_POSITIONS
    ):

        raise ValueError(
            f"Feedback position "
            f"{feedback_position} exceeds "
            f"maximum "
            f"{MAX_FEEDBACK_POSITIONS - 1}"
        )


    neurons = (
        mapping.feedback_neurons_for(
            token,
            feedback_position,
        )
    )


    states = []


    # --------------------------------------------------------
    # Feedback stimulation
    # --------------------------------------------------------

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


    feedback_mean = states.mean(
        axis=0
    )

    feedback_max = states.max(
        axis=0
    )

    feedback_final = states[
        -1
    ]


    delayed_state = (
        feedback_final
    )


    # --------------------------------------------------------
    # Feedback gap
    # --------------------------------------------------------

    for _ in range(
        FEEDBACK_GAP_STEPS
    ):

        fired = brain.step()


        delayed_state = np.asarray(
            trace.observe(
                fired
            )
        ).copy()


    return (
        feedback_mean,
        feedback_max,
        feedback_final,
        delayed_state,
    )


# ============================================================
# One teacher-forced logical episode
# ============================================================

def run_generator_episode(
    samples,
    seed,
):

    brain.reset(
        seed=seed
    )

    trace.reset()


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()

        trace.observe(
            fired
        )


    # --------------------------------------------------------
    # Same prompt across all generation steps
    # --------------------------------------------------------

    prompt = (
        samples[0]
        .prompt_tokens
    )


    for sample in samples:

        assert (
            sample.prompt_tokens
            == prompt
        )


    # --------------------------------------------------------
    # Encode prompt once
    # --------------------------------------------------------

    (
        prompt_mean,
        prompt_max,
        prompt_final,
    ) = stimulate_prompt(
        prompt
    )


    rows = []


    # ========================================================
    # Teacher forcing
    # ========================================================

    for generation_position, sample in enumerate(
        samples
    ):

        assert (
            sample.target_position
            == generation_position
        )


        # ----------------------------------------------------
        # Select feedback token
        #
        # Position 0:
        #   BOS
        #
        # Position n:
        #   previous correct target token
        # ----------------------------------------------------

        if generation_position == 0:

            feedback_token = (
                "BOS"
            )

        else:

            feedback_token = (
                samples[
                    generation_position - 1
                ].target_token
            )


        # ----------------------------------------------------
        # Validate teacher-forcing context
        # ----------------------------------------------------

        assert (
            sample.context_tokens[-1]
            == feedback_token
        )


        # ----------------------------------------------------
        # CRITICAL CHANGE:
        #
        # feedback token AND generation position
        #
        # Example:
        #
        # NAME_A @ position 2
        # ≠
        # NAME_A @ position 4
        # ----------------------------------------------------

        (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        ) = stimulate_feedback(
            token=feedback_token,

            feedback_position=(
                generation_position
            ),
        )


        # ----------------------------------------------------
        # Build neural feature
        # ----------------------------------------------------

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
            np.float32
        )


        rows.append(
            (
                feature,
                sample,
                feedback_token,
                generation_position,
            )
        )


    return rows


# ============================================================
# Extract split
# ============================================================

def extract_split(
    split_name,
):

    samples = (
        generator_samples[
            split_name
        ]
    )


    grouped = group_by_example(
        samples
    )


    logical_ids = sorted(
        grouped.keys()
    )


    features = []

    targets = []

    sample_ids = []

    example_ids = []

    intents = []

    target_positions = []

    feedback_tokens_used = []

    feedback_positions_used = []

    seeds_used = []


    total_episodes = (
        len(logical_ids)
        *
        len(NEURAL_SEEDS)
    )


    episode_number = 0


    print()
    print(
        "================================================"
    )

    print(
        f" EXTRACTING "
        f"{split_name.upper()}"
    )

    print(
        "================================================"
    )

    print()


    for example_id in logical_ids:

        example_samples = (
            grouped[
                example_id
            ]
        )


        for seed in NEURAL_SEEDS:

            episode_number += 1


            print(
                f"[{episode_number:03}/"
                f"{total_episodes}] "
                f"{example_id:16} "
                f"seed={seed} "
                f"steps="
                f"{len(example_samples)}"
            )


            rows = (
                run_generator_episode(
                    samples=(
                        example_samples
                    ),

                    seed=seed,
                )
            )


            for (
                feature,
                sample,
                feedback_token,
                feedback_position,
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

                target_positions.append(
                    sample.target_position
                )

                feedback_tokens_used.append(
                    feedback_token
                )

                feedback_positions_used.append(
                    feedback_position
                )

                seeds_used.append(
                    seed
                )


    # ========================================================
    # Arrays
    # ========================================================

    X = np.stack(
        features
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

    target_positions = np.asarray(
        target_positions,
        dtype=np.int32,
    )

    feedback_tokens_used = np.asarray(
        feedback_tokens_used
    )

    feedback_positions_used = np.asarray(
        feedback_positions_used,
        dtype=np.int32,
    )

    seeds_used = np.asarray(
        seeds_used,
        dtype=np.int32,
    )


    # ========================================================
    # Validation
    # ========================================================

    assert np.isfinite(
        X
    ).all()


    assert np.array_equal(
        target_positions,
        feedback_positions_used,
    )


    print()
    print(
        f"Logical prompts:       "
        f"{len(logical_ids)}"
    )

    print(
        f"Neural episodes:       "
        f"{total_episodes}"
    )

    print(
        f"Next-token examples:   "
        f"{len(X)}"
    )

    print(
        f"Readout neurons:       "
        f"{len(readout_population):,}"
    )

    print(
        f"Feature dimensions:    "
        f"{X.shape[1]:,}"
    )

    print(
        f"Finite features:       "
        f"{np.isfinite(X).all()}"
    )


    # ========================================================
    # Save
    # ========================================================

    output_file = (
        OUTPUT_DIR
        /
        (
            "generator_positional_"
            f"{split_name}.npz"
        )
    )


    np.savez(
        output_file,

        X=X,

        targets=targets,

        sample_ids=sample_ids,

        example_ids=example_ids,

        intents=intents,

        target_positions=(
            target_positions
        ),

        feedback_tokens=(
            feedback_tokens_used
        ),

        feedback_positions=(
            feedback_positions_used
        ),

        seeds=seeds_used,

        input_neurons=(
            input_neurons
        ),

        readout_population=(
            readout_population
        ),

        trace_tau=np.asarray(
            TRACE_TAU
        ),

        map_seed=np.asarray(
            MAP_SEED
        ),
    )


    print(
        f"Saved: "
        f"{output_file}"
    )


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Extract train / validation / test
# ============================================================

extract_split(
    "train"
)

extract_split(
    "validation"
)

extract_split(
    "test"
)


# ============================================================
# Final summary
# ============================================================

expected_train = (
    len(
        generator_samples[
            "train"
        ]
    )
    *
    len(NEURAL_SEEDS)
)


expected_validation = (
    len(
        generator_samples[
            "validation"
        ]
    )
    *
    len(NEURAL_SEEDS)
)


expected_test = (
    len(
        generator_samples[
            "test"
        ]
    )
    *
    len(NEURAL_SEEDS)
)


print()
print(
    "================================================"
)

print(
    " POSITIONAL FEEDBACK EXTRACTION COMPLETE"
)

print(
    "================================================"
)

print()


print(
    f"Train samples:      "
    f"{expected_train}"
)

print(
    f"Validation samples: "
    f"{expected_validation}"
)

print(
    f"Test samples:       "
    f"{expected_test}"
)


print()
print(
    f"Prompt neurons/channel:   "
    f"{PROMPT_NEURONS_PER_CHANNEL}"
)

print(
    f"Feedback neurons/channel: "
    f"{FEEDBACK_NEURONS_PER_CHANNEL}"
)

print(
    f"Feedback positions:       "
    f"{MAX_FEEDBACK_POSITIONS}"
)

print(
    f"Indirect readout neurons: "
    f"{len(readout_population):,}"
)


print()
print(
    "✅ Positional-feedback generator extraction PASSED"
)