from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import create_dual_channel_map
from pyfly.tokenizer import (
    build_code_vocabulary,
    code_to_tokens,
    tokens_to_code,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

DATASET_FILE = (
    "data/raw/"
    "sequence_challenge.jsonl"
)

SPLIT_FILE = (
    "data/processed/"
    "sequence_splits.json"
)

MODEL_FILE = (
    "artifacts/models/"
    "generator_best_decoder.joblib"
)

OUTPUT_FILE = Path(
    "artifacts/metrics/"
    "closed_loop_best_decoder.json"
)


# ============================================================
# MUST MATCH STEP 62
# ============================================================

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


# ============================================================
# Same evaluation seeds as before
#
# This is a diagnostic comparison, not a fresh untouched test.
# ============================================================

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


MAX_GENERATION_STEPS = 10


# ============================================================
# Header
# ============================================================

print()
print(
    "===================================================="
)

print(
    " PyFly CLOSED-LOOP - BEST BALANCED DECODER"
)

print(
    "===================================================="
)

print()


# ============================================================
# Load dataset
# ============================================================

examples = load_dataset(
    DATASET_FILE
)


example_by_id = {
    example.id: example
    for example in examples
}


# ============================================================
# Load split
# ============================================================

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8",
) as file:

    split_data = json.load(
        file
    )


test_ids = split_data[
    "test"
]


print(
    f"Test logical prompts: "
    f"{len(test_ids)}"
)


# ============================================================
# Load winning decoder
# ============================================================

print()
print(
    "Loading winning decoder..."
)


model = joblib.load(
    MODEL_FILE
)


print(
    "Decoder loaded."
)


# ============================================================
# Get classifier classes
#
# Step 68 uses pipeline key:
#
#   "classifier"
#
# ============================================================

classifier = model.named_steps[
    "classifier"
]


model_classes = np.asarray(
    classifier.classes_
).astype(str)


print()
print(
    f"Decoder type: "
    f"{classifier.__class__.__name__}"
)

print(
    f"Classes: "
    f"{len(model_classes)}"
)

print(
    model_classes.tolist()
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


# ============================================================
# Prompt vocabulary
#
# Must match Step 62 exactly.
# ============================================================

prompt_tokens = sorted({
    token

    for example in examples

    for token in example.prompt_tokens
})


print()
print(
    f"Prompt vocabulary: "
    f"{len(prompt_tokens)}"
)

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


visual_pool = np.asarray(
    brain.cells(
        ["visual_projection"]
    )
)


print(
    f"Visual-projection pool: "
    f"{len(visual_pool):,}"
)


# ============================================================
# Recreate original Step 62 dual-channel map
# ============================================================

dual_map = create_dual_channel_map(
    prompt_tokens=prompt_tokens,

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
# Indirect readout
# ============================================================

input_neurons = (
    dual_map.all_input_neurons()
)


readout_population = np.setdiff1d(
    visual_pool,
    input_neurons,
)


assert len(
    np.intersect1d(
        input_neurons,
        readout_population,
    )
) == 0


print()
print(
    f"Direct input neurons: "
    f"{len(input_neurons):,}"
)

print(
    f"Indirect readout: "
    f"{len(readout_population):,}"
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
# Prompt stimulation
# ============================================================

def stimulate_prompt(
    tokens,
):

    states = []


    for position, token in enumerate(
        tokens
    ):

        if (
            position
            >= MAX_PROMPT_POSITIONS
        ):

            raise ValueError(
                f"Prompt too long: "
                f"{tokens}"
            )


        neurons = (
            dual_map.prompt_neurons_for(
                token,
                position,
            )
        )


        # ----------------------------------------------------
        # Token
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
# Feedback stimulation
#
# IMPORTANT:
# This is NON-POSITIONAL feedback, exactly like Step 62.
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
# Build exactly the same 7-block feature as Step 62
# ============================================================

def build_feature(
    prompt_summary,
    feedback_summary,
):

    (
        prompt_mean,
        prompt_max,
        prompt_final,
    ) = prompt_summary


    (
        feedback_mean,
        feedback_max,
        feedback_final,
        delayed_state,
    ) = feedback_summary


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


    return feature


# ============================================================
# Ranked prediction
# ============================================================

def predict_ranked_tokens(
    feature,
):

    X = feature.reshape(
        1,
        -1,
    )


    # --------------------------------------------------------
    # Logistic Regression has decision_function().
    # --------------------------------------------------------

    scores = model.decision_function(
        X
    )


    scores = np.asarray(
        scores
    )


    if scores.ndim == 1:

        scores = scores.reshape(
            1,
            -1
        )


    row = scores[
        0
    ]


    ranked_indices = np.argsort(
        row
    )[
        ::-1
    ]


    ranked = [
        (
            str(
                model_classes[
                    index
                ]
            ),

            float(
                row[
                    index
                ]
            ),
        )

        for index in ranked_indices
    ]


    return ranked


# ============================================================
# Generate one program
# ============================================================

def generate_program(
    prompt,
    seed,
):

    # --------------------------------------------------------
    # Reset reservoir
    # --------------------------------------------------------

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
    # Encode prompt ONCE
    # --------------------------------------------------------

    prompt_summary = (
        stimulate_prompt(
            prompt
        )
    )


    # --------------------------------------------------------
    # Begin generation with BOS
    # --------------------------------------------------------

    feedback_token = (
        "BOS"
    )


    generated_tokens = []

    generation_trace = []


    # ========================================================
    # Closed loop
    # ========================================================

    for step in range(
        MAX_GENERATION_STEPS
    ):

        # ----------------------------------------------------
        # Inject previous predicted token
        # ----------------------------------------------------

        feedback_summary = (
            stimulate_feedback(
                feedback_token
            )
        )


        # ----------------------------------------------------
        # Neural feature
        # ----------------------------------------------------

        feature = build_feature(
            prompt_summary,
            feedback_summary,
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        ranked = (
            predict_ranked_tokens(
                feature
            )
        )


        predicted_token = (
            ranked[
                0
            ][
                0
            ]
        )


        generated_tokens.append(
            predicted_token
        )


        # ----------------------------------------------------
        # Save top-3 diagnostic
        # ----------------------------------------------------

        generation_trace.append(
            {
                "step": int(
                    step
                ),

                "feedback": str(
                    feedback_token
                ),

                "prediction": str(
                    predicted_token
                ),

                "top3": [
                    {
                        "token": token,

                        "score": float(
                            score
                        ),
                    }

                    for token, score
                    in ranked[
                        :3
                    ]
                ],
            }
        )


        # ----------------------------------------------------
        # Stop on EOS
        # ----------------------------------------------------

        if (
            predicted_token
            ==
            "EOS"
        ):

            break


        # ----------------------------------------------------
        # CLOSED LOOP:
        #
        # feed the MODEL'S OWN prediction back into MaleCNS
        # ----------------------------------------------------

        feedback_token = (
            predicted_token
        )


    return (
        generated_tokens,
        generation_trace,
    )


# ============================================================
# Token sequence accuracy
# ============================================================

def sequence_token_accuracy(
    target,
    predicted,
):

    denominator = max(
        len(
            target
        ),

        len(
            predicted
        ),
    )


    if denominator == 0:

        return 1.0


    correct = 0


    for index in range(
        denominator
    ):

        target_token = (
            target[
                index
            ]

            if index
            <
            len(
                target
            )

            else None
        )


        predicted_token = (
            predicted[
                index
            ]

            if index
            <
            len(
                predicted
            )

            else None
        )


        if (
            target_token
            ==
            predicted_token
        ):

            correct += 1


    return (
        correct
        /
        denominator
    )


# ============================================================
# Try decoding complete program
# ============================================================

def try_decode_code(
    tokens,
):

    try:

        code = tokens_to_code(
            tokens
        )


        return (
            True,
            code,
        )


    except ValueError:

        return (
            False,
            None,
        )


# ============================================================
# Run closed-loop evaluation
# ============================================================

results = []


exact_count = 0

valid_count = 0

eos_count = 0

token_accuracy_sum = 0.0


total_episodes = (
    len(
        test_ids
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
    "             GENERATION EPISODES"
)

print(
    "===================================================="
)

print()


for example_id in test_ids:

    example = (
        example_by_id[
            example_id
        ]
    )


    target_tokens = (
        code_to_tokens(
            example.code
        )
    )


    for seed in NEURAL_SEEDS:

        episode_number += 1


        (
            predicted_tokens,
            generation_trace,
        ) = generate_program(
            prompt=(
                example.prompt_tokens
            ),

            seed=seed,
        )


        # ----------------------------------------------------
        # Exact sequence
        # ----------------------------------------------------

        exact = (
            predicted_tokens
            ==
            target_tokens
        )


        # ----------------------------------------------------
        # Token accuracy
        # ----------------------------------------------------

        token_accuracy = (
            sequence_token_accuracy(
                target_tokens,
                predicted_tokens,
            )
        )


        # ----------------------------------------------------
        # Valid program
        # ----------------------------------------------------

        (
            valid_program,
            generated_code,
        ) = try_decode_code(
            predicted_tokens
        )


        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        produced_eos = (
            len(
                predicted_tokens
            )
            >
            0
            and
            predicted_tokens[
                -1
            ]
            ==
            "EOS"
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if exact:

            exact_count += 1


        if valid_program:

            valid_count += 1


        if produced_eos:

            eos_count += 1


        token_accuracy_sum += (
            token_accuracy
        )


        # ----------------------------------------------------
        # Print episode
        # ----------------------------------------------------

        mark = (
            "✓"
            if exact
            else "✗"
        )


        print(
            "----------------------------------------------------"
        )

        print(
            f"[{episode_number:02}/"
            f"{total_episodes}] "
            f"{mark} "
            f"{example.id} "
            f"seed={seed}"
        )


        print(
            f"Prompt: "
            f"{' '.join(example.prompt_tokens)}"
        )


        print(
            f"Target code: "
            f"{example.code}"
        )


        print(
            f"Target tokens: "
            f"{target_tokens}"
        )


        print(
            f"Generated: "
            f"{predicted_tokens}"
        )


        if valid_program:

            print(
                f"Generated code: "
                f"{generated_code}"
            )

        else:

            print(
                "Generated code: "
                "<INVALID>"
            )


        print(
            f"Token accuracy: "
            f"{token_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Trajectory
        # ----------------------------------------------------

        print(
            "Trajectory:"
        )


        for row in generation_trace:

            top3_string = ", ".join(
                (
                    f"{candidate['token']}"
                    f"({candidate['score']:.3f})"
                )

                for candidate in (
                    row[
                        "top3"
                    ]
                )
            )


            print(
                f"  step="
                f"{row['step']} "
                f"feedback="
                f"{row['feedback']:10} "
                f"-> "
                f"{row['prediction']:10} "
                f"| top3: "
                f"{top3_string}"
            )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "example_id": (
                    example.id
                ),

                "intent": (
                    example.intent
                ),

                "seed": int(
                    seed
                ),

                "prompt_tokens": (
                    example.prompt_tokens
                ),

                "target_code": (
                    example.code
                ),

                "target_tokens": (
                    target_tokens
                ),

                "generated_tokens": (
                    predicted_tokens
                ),

                "generated_code": (
                    generated_code
                ),

                "valid_program": bool(
                    valid_program
                ),

                "exact_program": bool(
                    exact
                ),

                "produced_eos": bool(
                    produced_eos
                ),

                "token_accuracy": float(
                    token_accuracy
                ),

                "generation_trace": (
                    generation_trace
                ),
            }
        )


# ============================================================
# Overall metrics
# ============================================================

mean_token_accuracy = (
    token_accuracy_sum
    /
    total_episodes
)


exact_accuracy = (
    exact_count
    /
    total_episodes
)


valid_rate = (
    valid_count
    /
    total_episodes
)


eos_rate = (
    eos_count
    /
    total_episodes
)


# ============================================================
# Prompt-level stats
# ============================================================

prompt_stats = {}


for example_id in test_ids:

    rows = [
        row

        for row in results

        if row[
            "example_id"
        ]
        ==
        example_id
    ]


    prompt_stats[
        example_id
    ] = {
        "exact_accuracy": float(
            np.mean([
                row[
                    "exact_program"
                ]

                for row in rows
            ])
        ),

        "valid_rate": float(
            np.mean([
                row[
                    "valid_program"
                ]

                for row in rows
            ])
        ),

        "mean_token_accuracy": float(
            np.mean([
                row[
                    "token_accuracy"
                ]

                for row in rows
            ])
        ),
    }


# ============================================================
# Common generated sequences
# ============================================================

sequence_counter = Counter(
    tuple(
        row[
            "generated_tokens"
        ]
    )

    for row in results
)


# ============================================================
# Final result
# ============================================================

print()
print()
print(
    "===================================================="
)

print(
    "      PyFly BEST DECODER CLOSED-LOOP RESULT"
)

print(
    "===================================================="
)

print()


print(
    f"Generation episodes:       "
    f"{total_episodes}"
)

print(
    f"Mean token accuracy:       "
    f"{mean_token_accuracy * 100:.2f}%"
)

print(
    f"Exact-program accuracy:    "
    f"{exact_accuracy * 100:.2f}%"
)

print(
    f"Valid-program rate:        "
    f"{valid_rate * 100:.2f}%"
)

print(
    f"EOS completion rate:       "
    f"{eos_rate * 100:.2f}%"
)


# ============================================================
# Old baseline comparison
# ============================================================

OLD_TOKEN_ACCURACY = 0.4857
OLD_EXACT = 0.0
OLD_VALID = 0.0333
OLD_EOS = 1.0


print()
print(
    "===================================================="
)

print(
    "             CLOSED-LOOP COMPARISON"
)

print(
    "===================================================="
)

print()


print(
    "OLD Ridge decoder:"
)

print(
    f"  Token accuracy: "
    f"{OLD_TOKEN_ACCURACY * 100:.2f}%"
)

print(
    f"  Exact programs: "
    f"{OLD_EXACT * 100:.2f}%"
)

print(
    f"  Valid programs: "
    f"{OLD_VALID * 100:.2f}%"
)

print(
    f"  EOS completion: "
    f"{OLD_EOS * 100:.2f}%"
)


print()
print(
    "NEW best decoder:"
)

print(
    f"  Token accuracy: "
    f"{mean_token_accuracy * 100:.2f}%"
)

print(
    f"  Exact programs: "
    f"{exact_accuracy * 100:.2f}%"
)

print(
    f"  Valid programs: "
    f"{valid_rate * 100:.2f}%"
)

print(
    f"  EOS completion: "
    f"{eos_rate * 100:.2f}%"
)


print()
print(
    f"Token accuracy change: "
    f"{(mean_token_accuracy - OLD_TOKEN_ACCURACY) * 100:+.2f} pp"
)

print(
    f"Exact-program change: "
    f"{(exact_accuracy - OLD_EXACT) * 100:+.2f} pp"
)

print(
    f"Valid-program change: "
    f"{(valid_rate - OLD_VALID) * 100:+.2f} pp"
)


# ============================================================
# Accuracy by prompt
# ============================================================

print()
print(
    "Accuracy by logical prompt:"
)

print()


for example_id in test_ids:

    stats = (
        prompt_stats[
            example_id
        ]
    )


    print(
        f"{example_id:16} "
        f"exact="
        f"{stats['exact_accuracy'] * 100:6.2f}% "
        f"token="
        f"{stats['mean_token_accuracy'] * 100:6.2f}% "
        f"valid="
        f"{stats['valid_rate'] * 100:6.2f}%"
    )


# ============================================================
# Most common outputs
# ============================================================

print()
print(
    "Most common generated sequences:"
)

print()


for sequence, count in (
    sequence_counter.most_common(
        10
    )
):

    print(
        f"{count:3}x "
        f"{list(sequence)}"
    )


# ============================================================
# Save
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output = {
    "decoder": (
        classifier.__class__.__name__
    ),

    "total_episodes": int(
        total_episodes
    ),

    "mean_token_accuracy": float(
        mean_token_accuracy
    ),

    "exact_program_accuracy": float(
        exact_accuracy
    ),

    "valid_program_rate": float(
        valid_rate
    ),

    "eos_completion_rate": float(
        eos_rate
    ),

    "old_baseline": {
        "mean_token_accuracy": (
            OLD_TOKEN_ACCURACY
        ),

        "exact_program_accuracy": (
            OLD_EXACT
        ),

        "valid_program_rate": (
            OLD_VALID
        ),

        "eos_completion_rate": (
            OLD_EOS
        ),
    },

    "accuracy_by_prompt": (
        prompt_stats
    ),

    "episodes": (
        results
    ),
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


print()
print(
    f"Saved: "
    f"{OUTPUT_FILE}"
)

print()

print(
    "✅ Best-decoder closed-loop generation completed"
)