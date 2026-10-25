from collections import defaultdict
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import create_dual_channel_map
from pyfly.generator_dataset import load_generator_jsonl
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


# ------------------------------------------------------------
# Neural repetitions
# ------------------------------------------------------------

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


# ------------------------------------------------------------
# Prompt encoder
# ------------------------------------------------------------

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS_PER_CHANNEL = 16

PROMPT_TOKEN_STEPS = 8
PROMPT_GAP_STEPS = 2


# ------------------------------------------------------------
# Feedback encoder
# ------------------------------------------------------------

FEEDBACK_NEURONS_PER_TOKEN = 32

FEEDBACK_TOKEN_STEPS = 8
FEEDBACK_GAP_STEPS = 1


# ------------------------------------------------------------
# Reservoir
# ------------------------------------------------------------

WARMUP_STEPS = 10

AMPLITUDE = 0.8

TRACE_TAU = 0.25

MAP_SEED = 123


# ============================================================
# Header
# ============================================================

print()
print(
    "============================================"
)

print(
    "   PyFly GENERATOR FEATURE EXTRACTION"
)

print(
    "============================================"
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
    f"Prompt vocabulary size: "
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
    f"Neurons:     {brain.n:,}"
)

print(
    f"Connections: "
    f"{len(brain.weights):,}"
)


# ============================================================
# Input neuron pool
# ============================================================

visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print()
print(
    f"Visual-projection neurons: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Build dual neural map
# ============================================================

dual_map = create_dual_channel_map(
    prompt_tokens=all_prompt_tokens,

    feedback_tokens=feedback_tokens,

    neuron_pool=visual_pool,

    max_prompt_positions=(
        MAX_PROMPT_POSITIONS
    ),

    prompt_neurons_per_channel=(
        PROMPT_NEURONS_PER_CHANNEL
    ),

    feedback_neurons_per_token=(
        FEEDBACK_NEURONS_PER_TOKEN
    ),

    seed=MAP_SEED,
)


# ============================================================
# Remove every directly stimulated neuron
# from the reservoir readout.
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
        "Dual encoder consumed the entire "
        "visual-projection population."
    )


# ============================================================
# Save mapping
# ============================================================

dual_map.save(
    "data/mappings/"
    "generator_dual_channel_map.npz"
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
# Group teacher-forcing samples
# by logical prompt.
# ============================================================

def group_by_example(samples):

    grouped = defaultdict(
        list
    )


    for sample in samples:

        grouped[
            sample.example_id
        ].append(
            sample
        )


    # Make sure output positions are ordered.
    for example_id in grouped:

        grouped[
            example_id
        ].sort(
            key=lambda sample:
                sample.target_position
        )


    return grouped


# ============================================================
# Stimulate prompt once
# ============================================================

def stimulate_prompt(
    prompt_tokens,
):

    states = []


    for position, token in enumerate(
        prompt_tokens
    ):

        if (
            position
            >= MAX_PROMPT_POSITIONS
        ):

            raise ValueError(
                f"Prompt has more than "
                f"{MAX_PROMPT_POSITIONS} tokens: "
                f"{prompt_tokens}"
            )


        neurons = (
            dual_map.prompt_neurons_for(
                token,
                position,
            )
        )


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


            states.append(
                state
            )


        # ----------------------------------------------------
        # Gap
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


    # --------------------------------------------------------
    # Reservoir-derived prompt summary.
    #
    # These are NOT raw prompt features.
    # They are FlyBrain responses.
    # --------------------------------------------------------

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
# Stimulate one feedback/code token
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
    # Feedback token
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
        states[-1]
    )


    # --------------------------------------------------------
    # Small feedback gap
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
# Run one complete teacher-forced trajectory
# ============================================================

def run_generator_episode(
    samples,
    seed,
):

    """
    One logical prompt produces several next-token features.

    Example:

        prompt
        ↓
        BOS             -> predict NAME_Y
        NAME_Y          -> predict ASSIGN
        ASSIGN          -> predict NAME_A
        NAME_A          -> predict MINUS
        ...
        NEWLINE         -> predict EOS

    The brain is NOT reset between output positions.
    """

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
    # All samples from one logical prompt must have
    # exactly the same prompt.
    # --------------------------------------------------------

    prompt_tokens = (
        samples[0]
        .prompt_tokens
    )


    for sample in samples:

        assert (
            sample.prompt_tokens
            == prompt_tokens
        )


    # --------------------------------------------------------
    # Process prompt ONCE.
    # --------------------------------------------------------

    (
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
        # What feedback token should be injected?
        #
        # Step 0:
        #   BOS -> predict first target
        #
        # Step n:
        #   previous correct target
        #   -> predict next target
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
        # Sanity-check against the teacher-forcing context
        # created in Step 57.
        # ----------------------------------------------------

        expected_context = (
            sample.context_tokens
        )


        assert (
            expected_context[-1]
            == feedback_token
        ), (
            f"Feedback mismatch for "
            f"{sample.sample_id}: "
            f"expected last context token "
            f"{expected_context[-1]!r}, "
            f"got {feedback_token!r}"
        )


        # ----------------------------------------------------
        # Inject feedback token
        # ----------------------------------------------------

        (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        ) = stimulate_feedback(
            feedback_token
        )


        # ----------------------------------------------------
        # Neural next-token feature
        #
        # 3 prompt-response summaries
        # +
        # 4 current feedback/reservoir summaries
        #
        # The current brain state also contains the
        # accumulated previous feedback trajectory.
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


        feature_rows.append(
            (
                feature,
                sample,
                feedback_token,
            )
        )


    return feature_rows


# ============================================================
# Extract one split
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
        len(logical_prompts)
        *
        len(NEURAL_SEEDS)
    )


    episode_number = 0


    print()
    print(
        "============================================"
    )

    print(
        f" EXTRACTING GENERATOR "
        f"{split_name.upper()}"
    )

    print(
        "============================================"
    )

    print()


    for example_id in logical_prompts:

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


            rows = run_generator_episode(
                example_samples,
                seed,
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
        f"Next-token examples: "
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


    assert np.isfinite(
        X
    ).all()


    # ========================================================
    # Save
    # ========================================================

    output_file = (
        OUTPUT_DIR
        /
        f"generator_{split_name}.npz"
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
    )


    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Extract all splits
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
    "============================================"
)

print(
    "   GENERATOR EXTRACTION COMPLETE"
)

print(
    "============================================"
)

print()


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


print(
    f"Train next-token neural samples: "
    f"{expected_train}"
)

print(
    f"Validation neural samples:       "
    f"{expected_validation}"
)

print(
    f"Test neural samples:             "
    f"{expected_test}"
)


print()
print(
    "✅ PyFly generator feature extraction PASSED"
)