from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import create_dual_channel_map
from pyfly.generator_dataset import load_generator_jsonl
from pyfly.tokenizer import build_code_vocabulary


# ============================================================
# Step 76B
# Rich prompt + generator-state next-token features
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
# Neural repetitions
#
# MUST remain identical to the previous generator experiment.
# ============================================================

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


# ============================================================
# Prompt encoder
#
# MUST match original Step-62 generator.
# ============================================================

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS_PER_CHANNEL = 16

PROMPT_TOKEN_STEPS = 8

PROMPT_GAP_STEPS = 2


# ============================================================
# Feedback encoder
#
# IMPORTANT:
# Keep the ORIGINAL NON-POSITIONAL feedback channel.
# ============================================================

FEEDBACK_NEURONS_PER_TOKEN = 32

FEEDBACK_TOKEN_STEPS = 8

FEEDBACK_GAP_STEPS = 1


# ============================================================
# Reservoir configuration
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
    "===================================================="
)

print(
    " PyFly STEP 76B - RICH GENERATOR FEATURE EXTRACTION"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load teacher-forcing samples
# ============================================================

generator_samples = {}


for split_name, path in GENERATOR_FILES.items():

    samples = load_generator_jsonl(
        path
    )

    generator_samples[
        split_name
    ] = samples

    print(
        f"{split_name:12} "
        f"teacher-forcing samples: "
        f"{len(samples)}"
    )


# ============================================================
# Build prompt vocabulary
# ============================================================

all_prompt_tokens = sorted({

    token

    for samples in generator_samples.values()

    for sample in samples

    for token in sample.prompt_tokens
})


print()

print(
    f"Prompt vocabulary size:   "
    f"{len(all_prompt_tokens)}"
)


# ============================================================
# Feedback vocabulary
# ============================================================

code_vocab = (
    build_code_vocabulary()
)


# PAD is never injected.
feedback_tokens = sorted([

    token

    for token in code_vocab.token_to_id

    if token != "PAD"
])


print(
    f"Feedback vocabulary size: "
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
    "Brain loaded."
)

print(
    f"Neurons:     "
    f"{brain.n:,}"
)

print(
    f"Connections: "
    f"{len(brain.weights):,}"
)


# ============================================================
# visual_projection neuron pool
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        [
            "visual_projection"
        ]
    ),
    dtype=np.int64,
)


print()

print(
    f"Visual-projection neurons: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Recreate EXACT original Step-62 dual neural map
#
# Prompt:
#
#     (token, prompt position)
#
# Feedback:
#
#     code token
#
# Feedback intentionally remains NON-POSITIONAL.
# ============================================================

dual_map = create_dual_channel_map(

    prompt_tokens=(
        all_prompt_tokens
    ),

    feedback_tokens=(
        feedback_tokens
    ),

    neuron_pool=(
        visual_pool
    ),

    max_prompt_positions=(
        MAX_PROMPT_POSITIONS
    ),

    prompt_neurons_per_channel=(
        PROMPT_NEURONS_PER_CHANNEL
    ),

    feedback_neurons_per_token=(
        FEEDBACK_NEURONS_PER_TOKEN
    ),

    seed=(
        MAP_SEED
    ),
)


# ============================================================
# Exclude ALL directly stimulated neurons
# ============================================================

input_neurons = (
    dual_map.all_input_neurons()
)


readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


print()

print(
    f"Direct input neurons:   "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout:       "
    f"{len(readout_population):,}"
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


if len(
    readout_population
) == 0:

    raise RuntimeError(
        "Dual encoder consumed "
        "the entire visual-projection "
        "population."
    )


# ============================================================
# Verify recreated map against existing Step-62 mapping
# ============================================================

SAVED_MAP_FILE = Path(
    "data/mappings/"
    "generator_dual_channel_map.npz"
)


def verify_saved_mapping():

    if not SAVED_MAP_FILE.exists():

        print()

        print(
            "Saved Step-62 map not found."
        )

        print(
            "Using deterministic recreation."
        )

        return


    saved = np.load(
        SAVED_MAP_FILE,
        allow_pickle=True,
    )


    # --------------------------------------------------------
    # Prompt mapping
    # --------------------------------------------------------

    saved_prompt_tokens = (
        saved[
            "prompt_tokens"
        ].astype(str)
    )


    saved_prompt_positions = (
        saved[
            "prompt_positions"
        ].astype(
            np.int64
        )
    )


    saved_prompt_neurons = (
        saved[
            "prompt_neurons"
        ].astype(
            np.int64
        )
    )


    for row_index, (
        token,
        position,
    ) in enumerate(
        zip(
            saved_prompt_tokens,
            saved_prompt_positions,
        )
    ):

        current = (
            dual_map.prompt_neurons_for(
                str(
                    token
                ),
                int(
                    position
                ),
            )
        )


        if not np.array_equal(
            current,
            saved_prompt_neurons[
                row_index
            ],
        ):

            raise RuntimeError(
                "Recreated prompt map "
                "does not match saved "
                "Step-62 map."
            )


    # --------------------------------------------------------
    # Feedback mapping
    # --------------------------------------------------------

    saved_feedback_tokens = (
        saved[
            "feedback_tokens"
        ].astype(str)
    )


    saved_feedback_neurons = (
        saved[
            "feedback_neurons"
        ].astype(
            np.int64
        )
    )


    for row_index, token in enumerate(
        saved_feedback_tokens
    ):

        current = (
            dual_map.feedback_neurons_for(
                str(
                    token
                )
            )
        )


        if not np.array_equal(
            current,
            saved_feedback_neurons[
                row_index
            ],
        ):

            raise RuntimeError(
                "Recreated feedback map "
                "does not match saved "
                "Step-62 map."
            )


    print()

    print(
        "Saved Step-62 neural map "
        "check: PASS"
    )


verify_saved_mapping()


# ============================================================
# Trace
# ============================================================

trace = Trace(
    brain,
    idx=readout_population,
    tau=TRACE_TAU,
)


# ============================================================
# Feature layout
#
# 6 position-sensitive prompt blocks
# +
# 3 global prompt blocks
# +
# 4 feedback / generator-state blocks
#
# = 13 neural blocks
# ============================================================

FEATURE_BLOCK_NAMES = [

    "prompt_position_0_mean",

    "prompt_position_1_mean",

    "prompt_position_2_mean",

    "prompt_position_3_mean",

    "prompt_position_4_mean",

    "prompt_position_5_mean",

    "prompt_mean",

    "prompt_max",

    "prompt_final",

    "feedback_mean",

    "feedback_max",

    "feedback_final",

    "delayed_state",
]


assert len(
    FEATURE_BLOCK_NAMES
) == 13


expected_feature_dimensions = (

    len(
        FEATURE_BLOCK_NAMES
    )

    *

    len(
        readout_population
    )
)


print()

print(
    f"Feature blocks:         "
    f"{len(FEATURE_BLOCK_NAMES)}"
)

print(
    f"Neurons per block:      "
    f"{len(readout_population):,}"
)

print(
    f"Expected dimensions:    "
    f"{expected_feature_dimensions:,}"
)


# ============================================================
# Group generator samples by logical prompt
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


    # --------------------------------------------------------
    # Order by output position
    # --------------------------------------------------------

    for example_id in grouped:

        grouped[
            example_id
        ].sort(
            key=lambda sample:
            sample.target_position
        )


        positions = [

            sample.target_position

            for sample in grouped[
                example_id
            ]
        ]


        expected_positions = list(
            range(
                len(
                    positions
                )
            )
        )


        if (
            positions
            !=
            expected_positions
        ):

            raise ValueError(
                f"Non-contiguous target "
                f"positions for "
                f"{example_id}: "
                f"{positions}"
            )


    return grouped


# ============================================================
# Stimulate prompt once
#
# NEW Step-76 representation:
#
#     P0 mean
#     P1 mean
#     P2 mean
#     P3 mean
#     P4 mean
#     P5 mean
#
#     global mean
#     global max
#     global final
#
# Position means are calculated only over the token
# stimulation period.
#
# Global summaries use token + gap states, matching
# the original Step-62 generator behavior.
# ============================================================

def stimulate_prompt(
    prompt_tokens,
):

    if len(
        prompt_tokens
    ) == 0:

        raise ValueError(
            "Prompt must contain "
            "at least one token."
        )


    if len(
        prompt_tokens
    ) > MAX_PROMPT_POSITIONS:

        raise ValueError(
            f"Prompt has more than "
            f"{MAX_PROMPT_POSITIONS} "
            f"tokens: "
            f"{prompt_tokens}"
        )


    position_means = []

    all_prompt_states = []


    # ========================================================
    # Process prompt positions
    # ========================================================

    for position, token in enumerate(
        prompt_tokens
    ):

        neurons = (
            dual_map.prompt_neurons_for(
                token,
                position,
            )
        )


        token_states = []


        # ----------------------------------------------------
        # Token stimulation
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


            token_states.append(
                state
            )


            all_prompt_states.append(
                state
            )


        token_states = np.stack(
            token_states
        )


        # ----------------------------------------------------
        # Position-sensitive mean
        # ----------------------------------------------------

        position_mean = (
            token_states.mean(
                axis=0
            )
        )


        position_means.append(
            position_mean
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


            all_prompt_states.append(
                state
            )


    # ========================================================
    # Zero-pad unused prompt positions
    # ========================================================

    zero_state = np.zeros(
        len(
            readout_population
        ),
        dtype=np.float32,
    )


    while len(
        position_means
    ) < MAX_PROMPT_POSITIONS:

        position_means.append(
            zero_state.copy()
        )


    if len(
        position_means
    ) != MAX_PROMPT_POSITIONS:

        raise RuntimeError(
            "Prompt-position padding error."
        )


    # ========================================================
    # Global prompt summaries
    # ========================================================

    all_prompt_states = np.stack(
        all_prompt_states
    )


    prompt_mean = (
        all_prompt_states.mean(
            axis=0
        )
    )


    prompt_max = (
        all_prompt_states.max(
            axis=0
        )
    )


    prompt_final = (
        all_prompt_states[
            -1
        ]
    )


    return (

        position_means,

        prompt_mean,

        prompt_max,

        prompt_final,
    )


# ============================================================
# Stimulate feedback/code token
#
# IMPORTANT:
#
# This remains the ORIGINAL non-positional feedback channel.
# ============================================================

def stimulate_feedback(
    token,
):

    neurons = (
        dual_map.feedback_neurons_for(
            token
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


    # --------------------------------------------------------
    # Feedback gap
    # --------------------------------------------------------

    delayed_state = (
        feedback_final
    )


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
# Run one complete teacher-forced generator episode
# ============================================================

def run_generator_episode(
    samples,
    seed,
):

    """
    Run one logical prompt with teacher forcing.

    Brain is reset ONCE before the prompt.

    Brain is NOT reset between output positions.
    """


    brain.reset(
        seed=seed
    )


    trace.reset()


    # ========================================================
    # Warm-up
    # ========================================================

    for _ in range(
        WARMUP_STEPS
    ):

        fired = brain.step()


        trace.observe(
            fired
        )


    # ========================================================
    # Validate common logical prompt
    # ========================================================

    prompt_tokens = (
        samples[
            0
        ].prompt_tokens
    )


    for sample in samples:

        assert (
            sample.prompt_tokens
            ==
            prompt_tokens
        )


    # ========================================================
    # Process prompt ONCE
    # ========================================================

    (
        position_means,

        prompt_mean,

        prompt_max,

        prompt_final,

    ) = stimulate_prompt(
        prompt_tokens
    )


    feature_rows = []


    # ========================================================
    # Teacher forcing
    # ========================================================

    for position, sample in enumerate(
        samples
    ):

        # ----------------------------------------------------
        # Feedback token
        #
        # Position 0:
        #     BOS -> predict token 0
        #
        # Later:
        #     previous TRUE token
        #     -> predict next token
        # ----------------------------------------------------

        if position == 0:

            feedback_token = (
                "BOS"
            )

        else:

            feedback_token = (
                samples[
                    position - 1
                ].target_token
            )


        # ----------------------------------------------------
        # Check stored teacher-forcing context
        # ----------------------------------------------------

        expected_context = (
            sample.context_tokens
        )


        if len(
            expected_context
        ) == 0:

            raise ValueError(
                f"Empty teacher-forcing "
                f"context for "
                f"{sample.sample_id}"
            )


        assert (
            expected_context[
                -1
            ]
            ==
            feedback_token
        ), (
            f"Feedback mismatch for "
            f"{sample.sample_id}: "
            f"expected "
            f"{expected_context[-1]!r}, "
            f"got "
            f"{feedback_token!r}"
        )


        # ====================================================
        # Stimulate feedback
        # ====================================================

        (
            feedback_mean,

            feedback_max,

            feedback_final,

            delayed_state,

        ) = stimulate_feedback(
            feedback_token
        )


        # ====================================================
        # Step-76 rich feature
        #
        # 6 prompt-position blocks
        #
        # +
        #
        # prompt_mean
        # prompt_max
        # prompt_final
        #
        # +
        #
        # feedback_mean
        # feedback_max
        # feedback_final
        # delayed_state
        #
        # =
        #
        # 13 blocks
        # ====================================================

        feature_blocks = (

            position_means

            +

            [
                prompt_mean,

                prompt_max,

                prompt_final,

                feedback_mean,

                feedback_max,

                feedback_final,

                delayed_state,
            ]
        )


        if len(
            feature_blocks
        ) != 13:

            raise RuntimeError(
                f"Expected 13 feature "
                f"blocks, got "
                f"{len(feature_blocks)}"
            )


        feature = np.concatenate(
            feature_blocks
        ).astype(
            np.float32,
            copy=False,
        )


        if (
            feature.shape[
                0
            ]
            !=
            expected_feature_dimensions
        ):

            raise RuntimeError(
                f"Expected feature size "
                f"{expected_feature_dimensions}, "
                f"got "
                f"{feature.shape[0]}"
            )


        feature_rows.append(
            (
                feature,

                sample,

                feedback_token,
            )
        )


    return feature_rows


# ============================================================
# Extract one dataset split
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


    features = []

    targets = []

    sample_ids = []

    example_ids = []

    intents = []

    target_positions = []

    feedback_tokens_used = []

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


    episode_number = 0


    print()

    print(
        "===================================================="
    )

    print(
        f" EXTRACTING RICH GENERATOR "
        f"{split_name.upper()}"
    )

    print(
        "===================================================="
    )

    print()


    # ========================================================
    # Logical prompts
    # ========================================================

    for example_id in logical_prompts:

        example_samples = (
            grouped[
                example_id
            ]
        )


        # ====================================================
        # Neural seeds
        # ====================================================

        for seed in NEURAL_SEEDS:

            episode_number += 1


            print(
                f"[{episode_number:03}/"
                f"{total_episodes:03}] "
                f"{example_id:16} "
                f"seed={seed} "
                f"steps="
                f"{len(example_samples)}"
            )


            rows = (
                run_generator_episode(
                    example_samples,
                    seed,
                )
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


                target_positions.append(
                    sample.target_position
                )


                feedback_tokens_used.append(
                    feedback_token
                )


                seeds_used.append(
                    seed
                )


    # ========================================================
    # NumPy arrays
    # ========================================================

    X = np.stack(
        features
    ).astype(
        np.float32,
        copy=False,
    )


    targets_array = np.asarray(
        targets
    )


    sample_ids_array = np.asarray(
        sample_ids
    )


    example_ids_array = np.asarray(
        example_ids
    )


    intents_array = np.asarray(
        intents
    )


    positions_array = np.asarray(
        target_positions,
        dtype=np.int32,
    )


    feedback_array = np.asarray(
        feedback_tokens_used
    )


    seeds_array = np.asarray(
        seeds_used,
        dtype=np.int32,
    )


    # ========================================================
    # Sanity checks
    # ========================================================

    assert (
        X.ndim
        ==
        2
    )


    assert (
        X.shape[
            1
        ]
        ==
        expected_feature_dimensions
    )


    assert np.isfinite(
        X
    ).all()


    row_count = len(
        X
    )


    assert (
        len(
            targets_array
        )
        ==
        row_count
    )


    assert (
        len(
            sample_ids_array
        )
        ==
        row_count
    )


    assert (
        len(
            example_ids_array
        )
        ==
        row_count
    )


    assert (
        len(
            intents_array
        )
        ==
        row_count
    )


    assert (
        len(
            positions_array
        )
        ==
        row_count
    )


    assert (
        len(
            feedback_array
        )
        ==
        row_count
    )


    assert (
        len(
            seeds_array
        )
        ==
        row_count
    )


    # ========================================================
    # Statistics
    # ========================================================

    print()

    print(
        f"Logical prompts:     "
        f"{len(logical_prompts)}"
    )


    print(
        f"Neural episodes:     "
        f"{total_episodes}"
    )


    print(
        f"Next-token rows:     "
        f"{len(X)}"
    )


    print(
        f"Feature dimensions:  "
        f"{X.shape[1]:,}"
    )


    print(
        f"Finite:              "
        f"{np.isfinite(X).all()}"
    )


    # ========================================================
    # Save
    # ========================================================

    output_file = (

        OUTPUT_DIR

        /

        f"generator_rich_"
        f"{split_name}.npz"
    )


    np.savez(

        output_file,

        X=X,

        targets=(
            targets_array
        ),

        sample_ids=(
            sample_ids_array
        ),

        example_ids=(
            example_ids_array
        ),

        intents=(
            intents_array
        ),

        target_positions=(
            positions_array
        ),

        feedback_tokens=(
            feedback_array
        ),

        seeds=(
            seeds_array
        ),

        input_neurons=(
            input_neurons
        ),

        readout_population=(
            readout_population
        ),

        feature_block_names=np.asarray(
            FEATURE_BLOCK_NAMES
        ),

        block_size=np.asarray(
            len(
                readout_population
            ),
            dtype=np.int32,
        ),

        max_prompt_positions=np.asarray(
            MAX_PROMPT_POSITIONS,
            dtype=np.int32,
        ),

        trace_tau=np.asarray(
            TRACE_TAU,
            dtype=np.float32,
        ),

        map_seed=np.asarray(
            MAP_SEED,
            dtype=np.int32,
        ),
    )


    size_mb = (

        output_file.stat().st_size

        /

        (
            1024
            *
            1024
        )
    )


    print(
        f"Saved:               "
        f"{output_file}"
    )


    print(
        f"File size:           "
        f"{size_mb:.2f} MB"
    )


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Extract all three splits
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

print()

print(
    "===================================================="
)

print(
    "      STEP 76B RICH FEATURE EXTRACTION COMPLETE"
)

print(
    "===================================================="
)

print()


print(
    "Expected neural rows:"
)


print(
    "  train:       1,140"
)


print(
    "  validation:    190"
)


print(
    "  test:          190"
)


print()


print(
    f"Expected feature dimensions: "
    f"{expected_feature_dimensions:,}"
)


print()


print(
    "Created:"
)


print(
    "  artifacts/features/"
    "generator_rich_train.npz"
)


print(
    "  artifacts/features/"
    "generator_rich_validation.npz"
)


print(
    "  artifacts/features/"
    "generator_rich_test.npz"
)


print()


print(
    "Next:"
)


print(
    "Compare all six linear decoder "
    "families on these rich features."
)