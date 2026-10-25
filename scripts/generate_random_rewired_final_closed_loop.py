from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import DualChannelMap
from pyfly.tokenizer import code_to_tokens, tokens_to_code


# ============================================================
# Step 78B-3
# RANDOM-REWIRED FINAL CLOSED-LOOP GENERATION
#
# Use:
#   * the same degree-preserving random rewiring from Step 78B-1
#   * the selected Step-78B-2 random-reservoir decoder
#   * the frozen Step-62 PyFly encoder/mapping/settings
#   * the frozen 24-prompt final holdout
#
# IMPORTANT:
#   * No fitting
#   * No architecture search
#   * No hyperparameter search
#   * Greedy top-1 feedback only
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"


# ============================================================
# Frozen final holdout / protocol
# ============================================================

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_evaluation_protocol.json"
)

FROZEN_MAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)


# ============================================================
# Random control artifacts
# ============================================================

RANDOM_FEATURE_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_reservoir_feature_manifest.json"
)

RANDOM_CONTROL_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_final_control.json"
)

RANDOM_CONTROL_MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "random_rewired_final_control.joblib"
)


# ============================================================
# Comparison artifacts
# ============================================================

MALECNS_CLOSED_LOOP_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_closed_loop_metrics.json"
)

ENCODER_ONLY_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "encoder_only_final_control.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_final_closed_loop.json"
)

OUTPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_rewired_final_closed_loop_predictions.jsonl"
)


# ============================================================
# Frozen Step-62 settings
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

NEURAL_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

MAX_GENERATION_STEPS = 10

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
# Frozen random-control settings
# ============================================================

REWIRE_SEED = 78001

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873
EXPECTED_FEATURE_DIMENSIONS = 41_111

EXPECTED_LOGICAL_PROMPTS = 24
EXPECTED_EPISODES = 120


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

    frozen = protocol.get(
        "frozen_file_hashes",
        {},
    )

    for stored_path, info in frozen.items():

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

            if value:
                return str(
                    value
                )

    raise ValueError(
        "Frozen protocol has no hash for "
        f"{wanted}"
    )


# ============================================================
# Verify all frozen artifacts
# ============================================================

def verify_integrity():

    required = [
        FINAL_HOLDOUT_FILE,
        PROTOCOL_FILE,
        FROZEN_MAP_FILE,
        RANDOM_FEATURE_MANIFEST_FILE,
        RANDOM_CONTROL_METRICS_FILE,
        RANDOM_CONTROL_MODEL_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: "
                f"{path}"
            )

    protocol = load_json(
        PROTOCOL_FILE
    )

    feature_manifest = load_json(
        RANDOM_FEATURE_MANIFEST_FILE
    )

    control_metrics = load_json(
        RANDOM_CONTROL_METRICS_FILE
    )

    # --------------------------------------------------------
    # Frozen protocol
    # --------------------------------------------------------

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Final protocol is not frozen."
        )

    architecture = protocol.get(
        "architecture",
        {},
    )

    checks = {
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

    for key, wanted in checks.items():

        actual = architecture.get(
            key
        )

        if actual != wanted:

            raise ValueError(
                "Frozen Step-62 setting "
                f"changed for {key}: "
                f"{actual!r}"
            )

    if not np.isclose(
        float(
            architecture.get(
                "amplitude",
                np.nan,
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
                "trace_tau",
                np.nan,
            )
        ),
        TRACE_TAU,
    ):

        raise ValueError(
            "Frozen trace tau changed."
        )

    # --------------------------------------------------------
    # Holdout + mapping integrity
    # --------------------------------------------------------

    for path in [
        FINAL_HOLDOUT_FILE,
        FROZEN_MAP_FILE,
    ]:

        if (
            sha256_file(
                path
            )
            !=
            frozen_hash_for(
                protocol,
                path,
            )
        ):

            raise ValueError(
                "Frozen file changed: "
                f"{normalized_relative(path)}"
            )

    # --------------------------------------------------------
    # Random feature manifest
    # --------------------------------------------------------

    if (
        feature_manifest.get(
            "step"
        )
        !=
        "78B-1"
    ):

        raise ValueError(
            "Unexpected random feature "
            "manifest step."
        )

    if (
        int(
            feature_manifest.get(
                "rewire_seed",
                -1,
            )
        )
        !=
        REWIRE_SEED
    ):

        raise ValueError(
            "Random rewiring seed changed."
        )

    if (
        int(
            feature_manifest.get(
                "feature_dimensions",
                -1,
            )
        )
        !=
        EXPECTED_FEATURE_DIMENSIONS
    ):

        raise ValueError(
            "Random feature dimensions changed."
        )

    # --------------------------------------------------------
    # Step-78B-2 metrics/model integrity
    # --------------------------------------------------------

    if (
        control_metrics.get(
            "step"
        )
        !=
        "78B-2"
    ):

        raise ValueError(
            "Unexpected random-control "
            "metrics step."
        )

    if (
        int(
            control_metrics.get(
                "rewire_seed",
                -1,
            )
        )
        !=
        REWIRE_SEED
    ):

        raise ValueError(
            "Step-78B-2 rewiring seed changed."
        )

    integrity = control_metrics.get(
        "integrity",
        {},
    )

    expected_model_hash = integrity.get(
        "model_sha256"
    )

    if (
        not expected_model_hash
        or
        sha256_file(
            RANDOM_CONTROL_MODEL_FILE
        )
        !=
        expected_model_hash
    ):

        raise ValueError(
            "Random-control model hash changed."
        )

    expected_manifest_hash = integrity.get(
        "feature_manifest_sha256"
    )

    if (
        expected_manifest_hash
        and
        sha256_file(
            RANDOM_FEATURE_MANIFEST_FILE
        )
        !=
        expected_manifest_hash
    ):

        raise ValueError(
            "Step-78B-1 feature manifest "
            "changed after model selection."
        )

    return (
        protocol,
        feature_manifest,
        control_metrics,
    )


# ============================================================
# Load exact frozen dual-channel map
# ============================================================

def load_frozen_dual_map(
    path: Path,
) -> DualChannelMap:

    data = np.load(
        path,
        allow_pickle=False,
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

        prompt_mapping[
            (
                str(
                    token
                ),
                int(
                    position
                ),
            )
        ] = (
            prompt_neurons[
                row
            ].copy()
        )

    feedback_mapping = {}

    for row, token in enumerate(
        feedback_tokens
    ):

        feedback_mapping[
            str(
                token
            )
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
            "Prompt-position limit changed."
        )

    if (
        dual_map.seed
        !=
        MAP_SEED
    ):

        raise ValueError(
            "Frozen mapping seed changed."
        )

    return dual_map


# ============================================================
# Degree-preserving rewiring
#
# Same operation used in Step 78B-1.
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

    brain.indices = (
        rng.permutation(
            brain.indices
        ).astype(
            brain.indices.dtype,
            copy=False,
        )
    )

    if not np.array_equal(
        brain.weights,
        original_weights,
    ):

        raise RuntimeError(
            "Rewiring changed weights."
        )

    if not np.array_equal(
        brain.indptr,
        original_indptr,
    ):

        raise RuntimeError(
            "Rewiring changed CSC pointers."
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
            "Rewiring changed postsynaptic "
            "index multiset."
        )

    if np.array_equal(
        brain.indices,
        original_indices,
    ):

        raise RuntimeError(
            "Rewiring produced original "
            "connectivity ordering."
        )

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
# Generation helpers
# ============================================================

def sequence_token_accuracy(
    target: list[str],
    predicted: list[str],
) -> float:

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

        if target_token == predicted_token:

            correct += 1

    return float(
        correct
        /
        denominator
    )


def decode_generated_tokens(
    tokens: list[str],
):

    try:

        code = tokens_to_code(
            tokens
        )

        template_valid = True

    except ValueError:

        return (
            None,
            False,
            False,
        )

    try:

        ast.parse(
            code
        )

        syntax_valid = True

    except SyntaxError:

        syntax_valid = False

    return (
        code,
        template_valid,
        syntax_valid,
    )


def first_error_position(
    target: list[str],
    predicted: list[str],
):

    limit = max(
        len(
            target
        ),
        len(
            predicted
        ),
    )

    for index in range(
        limit
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

        if target_token != predicted_token:

            return int(
                index
            )

    return None


# ============================================================
# Comparison loaders
# ============================================================

def load_real_malecns_closed_loop():

    if not MALECNS_CLOSED_LOOP_FILE.exists():

        return None

    data = load_json(
        MALECNS_CLOSED_LOOP_FILE
    )

    return data.get(
        "aggregate",
        data,
    )


def load_encoder_only_closed_loop():

    if not ENCODER_ONLY_FILE.exists():

        return None

    data = load_json(
        ENCODER_ONLY_FILE
    )

    final_closed = data.get(
        "final_closed_loop",
        {},
    )

    return final_closed.get(
        "aggregate"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        " PyFly STEP 78B-3 - RANDOM CLOSED LOOP"
    )

    print(
        "===================================================="
    )

    print()

    (
        protocol,
        feature_manifest,
        control_metrics,
    ) = verify_integrity()

    print(
        "Frozen protocol integrity:    PASS"
    )

    print(
        "Random feature manifest:      PASS"
    )

    print(
        "Random decoder integrity:     PASS"
    )

    print(
        f"Random rewiring seed:         "
        f"{REWIRE_SEED}"
    )

    # ========================================================
    # Load frozen final prompts
    # ========================================================

    examples = load_dataset(
        FINAL_HOLDOUT_FILE
    )

    if (
        len(
            examples
        )
        !=
        EXPECTED_LOGICAL_PROMPTS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_LOGICAL_PROMPTS} "
            "final prompts, got "
            f"{len(examples)}"
        )

    print(
        f"Final logical prompts:        "
        f"{len(examples)}"
    )

    # ========================================================
    # Load selected random decoder
    # ========================================================

    model = joblib.load(
        RANDOM_CONTROL_MODEL_FILE
    )

    selected = (
        control_metrics.get(
            "selection",
            {}
        )
    )

    print(
        f"Selected decoder family:      "
        f"{selected.get('selected_family')}"
    )

    print(
        f"Selected configuration:       "
        f"{selected.get('selected_configuration')}"
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    model_classes = np.asarray(
        classifier.classes_
    ).astype(str)

    # ========================================================
    # Load exact frozen mapping
    # ========================================================

    dual_map = load_frozen_dual_map(
        FROZEN_MAP_FILE
    )

    required_feedback_tokens = {
        "BOS",
        *model_classes.tolist(),
    }

    missing_feedback = sorted(
        token

        for token
        in required_feedback_tokens

        if token
        not in
        dual_map.feedback_mapping
    )

    if missing_feedback:

        raise ValueError(
            "Frozen mapping is missing "
            "feedback tokens: "
            f"{missing_feedback}"
        )

    for example in examples:

        for (
            position,
            token,
        ) in enumerate(
            example.prompt_tokens
        ):

            dual_map.prompt_neurons_for(
                token,
                position,
            )

    # ========================================================
    # Load real brain and apply exact same rewiring
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
            "Unexpected neuron count."
        )

    if (
        len(
            brain.weights
        )
        !=
        EXPECTED_CONNECTIONS
    ):

        raise ValueError(
            "Unexpected edge count."
        )

    print(
        "Applying the frozen "
        "degree-preserving rewiring..."
    )

    degree_preserving_rewire(
        brain,
        REWIRE_SEED,
    )

    print(
        "Random rewiring:             PASS"
    )

    # ========================================================
    # Same Step-62 indirect visual readout
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
            "Direct-input population changed."
        )

    if (
        len(
            readout_population
        )
        !=
        EXPECTED_READOUT_NEURONS
    ):

        raise ValueError(
            "Indirect readout population changed."
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
            "Feature dimensions changed."
        )

    print(
        f"Direct input neurons:        "
        f"{len(input_neurons):,}"
    )

    print(
        f"Indirect readout neurons:    "
        f"{len(readout_population):,}"
    )

    print(
        f"Feature dimensions:          "
        f"{feature_dimensions:,}"
    )

    # ========================================================
    # Trace
    # ========================================================

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    # ========================================================
    # Frozen Step-62 prompt
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
    # Frozen Step-62 feedback
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

    # ========================================================
    # Generate one program
    # ========================================================

    def generate_program(
        prompt_tokens,
        neural_seed,
    ):

        brain.reset(
            seed=neural_seed
        )

        trace.reset()

        for _ in range(
            WARMUP_STEPS
        ):

            fired = brain.step()

            trace.observe(
                fired
            )

        prompt_summary = (
            stimulate_prompt(
                prompt_tokens
            )
        )

        feedback_token = (
            "BOS"
        )

        generated_tokens = []
        generation_trace = []

        for step in range(
            MAX_GENERATION_STEPS
        ):

            feedback_summary = (
                stimulate_feedback(
                    feedback_token
                )
            )

            feature = np.concatenate(
                [
                    *prompt_summary,
                    *feedback_summary,
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            if (
                feature.shape[
                    0
                ]
                !=
                EXPECTED_FEATURE_DIMENSIONS
            ):

                raise RuntimeError(
                    "Random closed-loop "
                    "feature dimension mismatch."
                )

            predicted_token = str(
                model.predict(
                    feature.reshape(
                        1,
                        -1,
                    )
                )[
                    0
                ]
            )

            generated_tokens.append(
                predicted_token
            )

            generation_trace.append(
                {
                    "step": int(
                        step
                    ),

                    "feedback": (
                        feedback_token
                    ),

                    "prediction": (
                        predicted_token
                    ),
                }
            )

            if (
                predicted_token
                ==
                "EOS"
            ):

                break

            feedback_token = (
                predicted_token
            )

        return (
            generated_tokens,
            generation_trace,
        )

    # ========================================================
    # Final closed-loop episodes
    # ========================================================

    total_episodes = (
        len(
            examples
        )
        *
        len(
            NEURAL_SEEDS
        )
    )

    if (
        total_episodes
        !=
        EXPECTED_EPISODES
    ):

        raise ValueError(
            "Unexpected final episode count."
        )

    results = []

    episode_number = 0

    print()

    print(
        "===================================================="
    )

    print(
        "      RANDOM-REWIRED CLOSED-LOOP EPISODES"
    )

    print(
        "===================================================="
    )

    print()

    for example in examples:

        target_tokens = (
            code_to_tokens(
                example.code
            )
        )

        for neural_seed in (
            NEURAL_SEEDS
        ):

            episode_number += 1

            (
                generated_tokens,
                generation_trace,
            ) = generate_program(
                example.prompt_tokens,
                neural_seed,
            )

            token_accuracy = (
                sequence_token_accuracy(
                    target_tokens,
                    generated_tokens,
                )
            )

            exact_program = (
                generated_tokens
                ==
                target_tokens
            )

            (
                generated_code,
                template_valid,
                syntax_valid,
            ) = decode_generated_tokens(
                generated_tokens
            )

            eos_completed = bool(
                generated_tokens
                and
                generated_tokens[
                    -1
                ]
                ==
                "EOS"
            )

            first_error = (
                first_error_position(
                    target_tokens,
                    generated_tokens,
                )
            )

            print(
                f"[{episode_number:03}/"
                f"{total_episodes:03}] "
                f"{example.id:20} "
                f"seed="
                f"{neural_seed} "
                f"token="
                f"{token_accuracy * 100:6.2f}% "
                f"exact="
                f"{exact_program}"
            )

            results.append(
                {
                    "example_id": (
                        example.id
                    ),

                    "intent": (
                        example.intent
                    ),

                    "neural_seed": int(
                        neural_seed
                    ),

                    "prompt_tokens": list(
                        example.prompt_tokens
                    ),

                    "target_code": (
                        example.code
                    ),

                    "target_tokens": (
                        target_tokens
                    ),

                    "generated_tokens": (
                        generated_tokens
                    ),

                    "generated_code": (
                        generated_code
                    ),

                    "token_accuracy": float(
                        token_accuracy
                    ),

                    "exact_program": bool(
                        exact_program
                    ),

                    "valid_program": bool(
                        template_valid
                    ),

                    "syntax_valid": bool(
                        syntax_valid
                    ),

                    "eos_completed": bool(
                        eos_completed
                    ),

                    "first_error_position": (
                        first_error
                    ),

                    "generation_trace": (
                        generation_trace
                    ),
                }
            )

    # ========================================================
    # Aggregate
    # ========================================================

    aggregate = {
        "mean_token_accuracy": float(
            np.mean(
                [
                    row[
                        "token_accuracy"
                    ]

                    for row
                    in results
                ]
            )
        ),

        "exact_program_accuracy": float(
            np.mean(
                [
                    row[
                        "exact_program"
                    ]

                    for row
                    in results
                ]
            )
        ),

        "valid_program_rate": float(
            np.mean(
                [
                    row[
                        "valid_program"
                    ]

                    for row
                    in results
                ]
            )
        ),

        "syntax_valid_rate": float(
            np.mean(
                [
                    row[
                        "syntax_valid"
                    ]

                    for row
                    in results
                ]
            )
        ),

        "eos_completion_rate": float(
            np.mean(
                [
                    row[
                        "eos_completed"
                    ]

                    for row
                    in results
                ]
            )
        ),
    }

    # ========================================================
    # Per-seed metrics
    # ========================================================

    per_seed = {}

    for neural_seed in NEURAL_SEEDS:

        rows = [
            row

            for row
            in results

            if row[
                "neural_seed"
            ]
            ==
            neural_seed
        ]

        per_seed[
            str(
                neural_seed
            )
        ] = {
            "mean_token_accuracy": float(
                np.mean(
                    [
                        row[
                            "token_accuracy"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

            "exact_program_accuracy": float(
                np.mean(
                    [
                        row[
                            "exact_program"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

            "valid_program_rate": float(
                np.mean(
                    [
                        row[
                            "valid_program"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

            "eos_completion_rate": float(
                np.mean(
                    [
                        row[
                            "eos_completed"
                        ]

                        for row
                        in rows
                    ]
                )
            ),
        }

    mean_sd = {}

    for metric_name in [
        "mean_token_accuracy",
        "exact_program_accuracy",
        "valid_program_rate",
        "eos_completion_rate",
    ]:

        values = np.asarray(
            [
                per_seed[
                    str(
                        seed
                    )
                ][
                    metric_name
                ]

                for seed
                in NEURAL_SEEDS
            ],
            dtype=float,
        )

        mean_sd[
            metric_name
        ] = {
            "mean": float(
                np.mean(
                    values
                )
            ),

            "sd": float(
                np.std(
                    values,
                    ddof=0,
                )
            ),
        }

    # ========================================================
    # Per-intent metrics
    # ========================================================

    per_intent = {}

    for intent in sorted(
        {
            row[
                "intent"
            ]

            for row
            in results
        }
    ):

        rows = [
            row

            for row
            in results

            if row[
                "intent"
            ]
            ==
            intent
        ]

        per_intent[
            intent
        ] = {
            "episodes": int(
                len(
                    rows
                )
            ),

            "mean_token_accuracy": float(
                np.mean(
                    [
                        row[
                            "token_accuracy"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

            "exact_program_accuracy": float(
                np.mean(
                    [
                        row[
                            "exact_program"
                        ]

                        for row
                        in rows
                    ]
                )
            ),

            "valid_program_rate": float(
                np.mean(
                    [
                        row[
                            "valid_program"
                        ]

                        for row
                        in rows
                    ]
                )
            ),
        }

    # ========================================================
    # First errors + common sequences
    # ========================================================

    first_errors = Counter()

    for row in results:

        position = row[
            "first_error_position"
        ]

        if position is None:

            first_errors[
                "exact"
            ] += 1

        else:

            first_errors[
                f"P{position}"
            ] += 1

    common_sequences = Counter(
        tuple(
            row[
                "generated_tokens"
            ]
        )

        for row
        in results
    )

    # ========================================================
    # Compare controls
    # ========================================================

    real_malecns = (
        load_real_malecns_closed_loop()
    )

    encoder_only = (
        load_encoder_only_closed_loop()
    )

    print()

    print(
        "===================================================="
    )

    print(
        " RANDOM-REWIRED FINAL CLOSED-LOOP RESULTS"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Generation episodes:        "
        f"{total_episodes}"
    )

    print(
        f"Mean token accuracy:        "
        f"{aggregate['mean_token_accuracy'] * 100:.2f}%"
    )

    print(
        f"Exact-program accuracy:     "
        f"{aggregate['exact_program_accuracy'] * 100:.2f}%"
    )

    print(
        f"Valid-program rate:         "
        f"{aggregate['valid_program_rate'] * 100:.2f}%"
    )

    print(
        f"Syntax-valid rate:          "
        f"{aggregate['syntax_valid_rate'] * 100:.2f}%"
    )

    print(
        f"EOS completion rate:        "
        f"{aggregate['eos_completion_rate'] * 100:.2f}%"
    )

    print()

    print(
        "Mean +/- SD across neural seeds:"
    )

    print(
        f"  Token accuracy: "
        f"{mean_sd['mean_token_accuracy']['mean'] * 100:.2f}% "
        f"+/- "
        f"{mean_sd['mean_token_accuracy']['sd'] * 100:.2f} pp"
    )

    print(
        f"  Exact programs: "
        f"{mean_sd['exact_program_accuracy']['mean'] * 100:.2f}% "
        f"+/- "
        f"{mean_sd['exact_program_accuracy']['sd'] * 100:.2f} pp"
    )

    print(
        f"  Valid programs: "
        f"{mean_sd['valid_program_rate']['mean'] * 100:.2f}% "
        f"+/- "
        f"{mean_sd['valid_program_rate']['sd'] * 100:.2f} pp"
    )

    print()

    print(
        "Closed-loop by intent:"
    )

    for intent, row in (
        per_intent.items()
    ):

        print(
            f"  {intent:10} "
            f"token="
            f"{row['mean_token_accuracy'] * 100:6.2f}%  "
            f"exact="
            f"{row['exact_program_accuracy'] * 100:6.2f}%  "
            f"valid="
            f"{row['valid_program_rate'] * 100:6.2f}%"
        )

    print()

    print(
        "First error position:"
    )

    for key, count in sorted(
        first_errors.items(),
        key=lambda item: (
            999
            if item[
                0
            ]
            ==
            "exact"
            else
            int(
                item[
                    0
                ][
                    1:
                ]
            )
        ),
    ):

        print(
            f"  {key:6} "
            f"{count:3}/{total_episodes} "
            f"= "
            f"{count / total_episodes * 100:6.2f}%"
        )

    # ========================================================
    # Three-way closed-loop comparison
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "       FINAL CLOSED-LOOP CONTROL COMPARISON"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"{'System':28}"
        f"{'Token':>10}"
        f"{'Exact':>10}"
        f"{'Valid':>10}"
    )

    print(
        "-" * 58
    )

    if encoder_only is not None:

        print(
            f"{'Encoder-only':28}"
            f"{encoder_only['mean_token_accuracy'] * 100:9.2f}%"
            f"{encoder_only['exact_program_accuracy'] * 100:9.2f}%"
            f"{encoder_only['valid_program_rate'] * 100:9.2f}%"
        )

    if real_malecns is not None:

        print(
            f"{'Real MaleCNS':28}"
            f"{real_malecns['mean_token_accuracy'] * 100:9.2f}%"
            f"{real_malecns['exact_program_accuracy'] * 100:9.2f}%"
            f"{real_malecns['valid_program_rate'] * 100:9.2f}%"
        )

    print(
        f"{'Random-rewired MaleCNS':28}"
        f"{aggregate['mean_token_accuracy'] * 100:9.2f}%"
        f"{aggregate['exact_program_accuracy'] * 100:9.2f}%"
        f"{aggregate['valid_program_rate'] * 100:9.2f}%"
    )

    if real_malecns is not None:

        print()

        print(
            "Random-rewired minus real MaleCNS:"
        )

        print(
            f"  Token accuracy: "
            f"{(aggregate['mean_token_accuracy'] - real_malecns['mean_token_accuracy']) * 100:+.2f} pp"
        )

        print(
            f"  Exact programs: "
            f"{(aggregate['exact_program_accuracy'] - real_malecns['exact_program_accuracy']) * 100:+.2f} pp"
        )

        print(
            f"  Valid programs: "
            f"{(aggregate['valid_program_rate'] - real_malecns['valid_program_rate']) * 100:+.2f} pp"
        )

    print()

    print(
        "Most common generated sequences:"
    )

    for sequence, count in (
        common_sequences.most_common(
            10
        )
    ):

        print(
            f"  {count:3}x "
            f"{list(sequence)}"
        )

    # ========================================================
    # Save predictions
    # ========================================================

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PREDICTIONS.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in results:

            file.write(
                json.dumps(
                    row
                )
            )

            file.write(
                "\n"
            )

    # ========================================================
    # Save metrics
    # ========================================================

    output = {
        "step": "78B-3",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "control": (
            "degree_preserving_random_rewired_reservoir"
        ),

        "rewire_seed": (
            REWIRE_SEED
        ),

        "selected_decoder": (
            selected
        ),

        "aggregate": (
            aggregate
        ),

        "per_seed": (
            per_seed
        ),

        "mean_sd_across_neural_seeds": (
            mean_sd
        ),

        "per_intent": (
            per_intent
        ),

        "first_error_position": (
            dict(
                first_errors
            )
        ),

        "most_common_sequences": [
            {
                "tokens": list(
                    sequence
                ),

                "count": int(
                    count
                ),
            }

            for sequence, count
            in common_sequences.most_common(
                20
            )
        ],

        "comparison": {
            "encoder_only": (
                encoder_only
            ),

            "real_malecns": (
                real_malecns
            ),

            "random_rewired_malecns": (
                aggregate
            ),
        },

        "integrity": {
            "protocol_sha256": (
                sha256_file(
                    PROTOCOL_FILE
                )
            ),

            "feature_manifest_sha256": (
                sha256_file(
                    RANDOM_FEATURE_MANIFEST_FILE
                )
            ),

            "random_control_metrics_sha256": (
                sha256_file(
                    RANDOM_CONTROL_METRICS_FILE
                )
            ),

            "random_control_model_sha256": (
                sha256_file(
                    RANDOM_CONTROL_MODEL_FILE
                )
            ),

            "final_holdout_sha256": (
                sha256_file(
                    FINAL_HOLDOUT_FILE
                )
            ),

            "mapping_sha256": (
                sha256_file(
                    FROZEN_MAP_FILE
                )
            ),
        },

        "rules": [
            (
                "No model fitting occurred during "
                "Step 78B-3."
            ),

            (
                "The same precommitted rewiring seed "
                "from Step 78B-1 was used."
            ),

            (
                "The Step-78B-2 selected decoder was "
                "used unchanged."
            ),

            (
                "No final-holdout result was used for "
                "architecture or hyperparameter selection."
            ),
        ],
    }

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_METRICS.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        f"{normalized_relative(OUTPUT_METRICS)}"
    )

    print(
        "  "
        f"{normalized_relative(OUTPUT_PREDICTIONS)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 78B-3 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Next: shuffled-label control and final "
        "scientific summary."
    )


if __name__ == "__main__":
    main()
