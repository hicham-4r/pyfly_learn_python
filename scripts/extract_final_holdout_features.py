from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl


# ============================================================
# Step 77C
# Extract the FINAL HOLDOUT with the frozen Step-62 MaleCNS
# feature pipeline.
#
# IMPORTANT:
#   * The frozen mapping is LOADED, never regenerated.
#   * The MaleCNS recurrent weights remain frozen.
#   * Directly stimulated neurons remain excluded.
#   * No decoder fitting or hyperparameter selection happens here.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "final_holdout.jsonl"
)

FINAL_GENERATOR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_final_holdout.jsonl"
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

REFERENCE_FEATURE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_train.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_final_holdout.npz"
)


# ============================================================
# Frozen Step-62 configuration
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

EXPECTED_LOGICAL_PROMPTS = 24
EXPECTED_TEACHER_FORCING_ROWS = 152
EXPECTED_NEURAL_EPISODES = 120
EXPECTED_NEURAL_ROWS = 760


# ============================================================
# Helpers
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()



def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)



def normalized_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")



def frozen_hash_for(protocol: dict, path: Path) -> str:
    wanted = normalized_relative(path)

    hashes = protocol.get("frozen_file_hashes", {})

    for stored_path, info in hashes.items():
        normalized = stored_path.replace("\\", "/")

        if normalized == wanted:
            value = info.get("sha256")

            if not value:
                raise ValueError(
                    f"Protocol contains no SHA-256 for {wanted}"
                )

            return str(value)

    raise ValueError(
        f"Protocol contains no frozen hash entry for {wanted}"
    )


# ============================================================
# Verify Step-77A protocol and frozen files
# ============================================================


def verify_protocol() -> dict:
    required_files = [
        FINAL_HOLDOUT_FILE,
        FINAL_GENERATOR_FILE,
        PROTOCOL_FILE,
        FROZEN_MAP_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen file is missing: {path}"
            )

    protocol = load_json(PROTOCOL_FILE)

    if protocol.get("status") != "FROZEN_BEFORE_FINAL_EVALUATION":
        raise ValueError(
            "The final-evaluation protocol is not marked as frozen."
        )

    architecture = protocol.get("architecture", {})

    expected_architecture = {
        "recurrent_weights": "frozen",
        "max_prompt_positions": MAX_PROMPT_POSITIONS,
        "prompt_neurons_per_channel": PROMPT_NEURONS_PER_CHANNEL,
        "feedback_neurons_per_token": FEEDBACK_NEURONS_PER_TOKEN,
        "prompt_token_steps": PROMPT_TOKEN_STEPS,
        "prompt_gap_steps": PROMPT_GAP_STEPS,
        "feedback_token_steps": FEEDBACK_TOKEN_STEPS,
        "feedback_gap_steps": FEEDBACK_GAP_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "mapping_seed": MAP_SEED,
        "neural_seeds": NEURAL_SEEDS,
        "feature_blocks": FEATURE_BLOCK_NAMES,
    }

    for key, expected in expected_architecture.items():
        actual = architecture.get(key)

        if actual != expected:
            raise ValueError(
                "Frozen protocol mismatch for "
                f"{key!r}: expected {expected!r}, got {actual!r}"
            )

    for key, expected in {
        "amplitude": AMPLITUDE,
        "trace_tau": TRACE_TAU,
    }.items():
        actual = float(architecture.get(key))

        if not np.isclose(actual, expected):
            raise ValueError(
                "Frozen protocol mismatch for "
                f"{key!r}: expected {expected}, got {actual}"
            )

    # --------------------------------------------------------
    # Final holdout must still be byte-for-byte identical.
    # --------------------------------------------------------

    for path in [
        FINAL_HOLDOUT_FILE,
        FINAL_GENERATOR_FILE,
        FROZEN_MAP_FILE,
    ]:
        expected_hash = frozen_hash_for(protocol, path)
        current_hash = sha256_file(path)

        if current_hash != expected_hash:
            raise ValueError(
                f"Frozen-file integrity failure for {normalized_relative(path)}\n"
                f"Expected: {expected_hash}\n"
                f"Current:  {current_hash}"
            )

    return protocol


# ============================================================
# Load the EXACT frozen neural map
# ============================================================


def load_frozen_dual_map(path: Path) -> DualChannelMap:
    data = np.load(path, allow_pickle=False)

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

    missing = required - set(data.files)

    if missing:
        raise ValueError(
            "Frozen dual-channel map is missing arrays: "
            f"{sorted(missing)}"
        )

    prompt_tokens = data["prompt_tokens"].astype(str)
    prompt_positions = data["prompt_positions"].astype(np.int64)
    prompt_neurons = data["prompt_neurons"].astype(np.int64)

    feedback_tokens = data["feedback_tokens"].astype(str)
    feedback_neurons = data["feedback_neurons"].astype(np.int64)

    prompt_mapping: dict[tuple[str, int], np.ndarray] = {}

    for row, (token, position) in enumerate(
        zip(prompt_tokens, prompt_positions)
    ):
        key = (str(token), int(position))

        if key in prompt_mapping:
            raise ValueError(f"Duplicate prompt-map key: {key}")

        prompt_mapping[key] = prompt_neurons[row].copy()

    feedback_mapping: dict[str, np.ndarray] = {}

    for row, token in enumerate(feedback_tokens):
        token = str(token)

        if token in feedback_mapping:
            raise ValueError(f"Duplicate feedback-map token: {token}")

        feedback_mapping[token] = feedback_neurons[row].copy()

    dual_map = DualChannelMap(
        prompt_mapping=prompt_mapping,
        feedback_mapping=feedback_mapping,
        prompt_neurons_per_channel=int(
            np.asarray(data["prompt_neurons_per_channel"]).item()
        ),
        feedback_neurons_per_token=int(
            np.asarray(data["feedback_neurons_per_token"]).item()
        ),
        max_prompt_positions=int(
            np.asarray(data["max_prompt_positions"]).item()
        ),
        seed=int(np.asarray(data["seed"]).item()),
    )

    if dual_map.prompt_neurons_per_channel != PROMPT_NEURONS_PER_CHANNEL:
        raise ValueError("Frozen prompt-neuron count changed.")

    if dual_map.feedback_neurons_per_token != FEEDBACK_NEURONS_PER_TOKEN:
        raise ValueError("Frozen feedback-neuron count changed.")

    if dual_map.max_prompt_positions != MAX_PROMPT_POSITIONS:
        raise ValueError("Frozen maximum prompt position changed.")

    if dual_map.seed != MAP_SEED:
        raise ValueError("Frozen mapping seed changed.")

    return dual_map


# ============================================================
# Group teacher-forcing rows by logical prompt
# ============================================================


def group_by_example(samples):
    grouped = defaultdict(list)

    for sample in samples:
        grouped[sample.example_id].append(sample)

    for example_id in grouped:
        grouped[example_id].sort(
            key=lambda sample: sample.target_position
        )

        positions = [
            sample.target_position
            for sample in grouped[example_id]
        ]

        expected = list(range(len(positions)))

        if positions != expected:
            raise ValueError(
                f"Non-contiguous target positions for {example_id}: "
                f"{positions}"
            )

    return grouped


# ============================================================
# Main
# ============================================================


def main() -> None:
    print()
    print("====================================================")
    print("  PyFly STEP 77C - FINAL HOLDOUT FEATURE EXTRACTION")
    print("====================================================")
    print()

    protocol = verify_protocol()

    print("Frozen protocol integrity: PASS")
    print("Final holdout integrity:   PASS")
    print("Frozen mapping integrity:  PASS")
    print()

    # ========================================================
    # Load frozen map and final teacher-forcing rows
    # ========================================================

    dual_map = load_frozen_dual_map(FROZEN_MAP_FILE)

    samples = load_generator_jsonl(FINAL_GENERATOR_FILE)

    if len(samples) != EXPECTED_TEACHER_FORCING_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEACHER_FORCING_ROWS} final "
            f"teacher-forcing rows, got {len(samples)}"
        )

    grouped = group_by_example(samples)

    if len(grouped) != EXPECTED_LOGICAL_PROMPTS:
        raise ValueError(
            f"Expected {EXPECTED_LOGICAL_PROMPTS} final logical prompts, "
            f"got {len(grouped)}"
        )

    # Verify every final prompt channel exists in the frozen map.
    for example_id, example_samples in grouped.items():
        prompt_tokens = example_samples[0].prompt_tokens

        if len(prompt_tokens) > MAX_PROMPT_POSITIONS:
            raise ValueError(
                f"Prompt {example_id} exceeds the frozen maximum length."
            )

        for position, token in enumerate(prompt_tokens):
            dual_map.prompt_neurons_for(token, position)

        for position, sample in enumerate(example_samples):
            feedback_token = (
                "BOS"
                if position == 0
                else example_samples[position - 1].target_token
            )

            dual_map.feedback_neurons_for(feedback_token)

            if not sample.context_tokens:
                raise ValueError(
                    f"Empty teacher-forcing context for {sample.sample_id}"
                )

            if sample.context_tokens[-1] != feedback_token:
                raise ValueError(
                    f"Teacher-forcing context mismatch for {sample.sample_id}: "
                    f"expected {sample.context_tokens[-1]!r}, "
                    f"got {feedback_token!r}"
                )

    print(f"Final logical prompts:       {len(grouped)}")
    print(f"Teacher-forcing rows:        {len(samples)}")
    print(f"Frozen prompt channels:      {len(dual_map.prompt_mapping)}")
    print(f"Frozen feedback channels:    {len(dual_map.feedback_mapping)}")
    print()

    # ========================================================
    # Load FlyBrain
    # ========================================================

    print("Loading FlyBrain on GPU...")

    brain = FlyBrain(
        device=DEVICE,
        seed=0,
    )

    print("Brain loaded.")
    print(f"Neurons:                     {brain.n:,}")
    print(f"Connections:                 {len(brain.weights):,}")

    visual_pool = np.asarray(
        brain.cells(["visual_projection"]),
        dtype=np.int64,
    )

    input_neurons = dual_map.all_input_neurons()

    # Every frozen input neuron must still belong to visual_projection.
    if not np.all(np.isin(input_neurons, visual_pool)):
        raise ValueError(
            "Frozen neural map contains neurons outside visual_projection."
        )

    readout_population = np.setdiff1d(
        visual_pool,
        input_neurons,
    )

    if len(np.intersect1d(input_neurons, readout_population)) != 0:
        raise RuntimeError(
            "Directly stimulated neurons leaked into the final readout."
        )

    if len(readout_population) == 0:
        raise RuntimeError("Frozen readout population is empty.")

    print()
    print(f"Visual-projection neurons:   {len(visual_pool):,}")
    print(f"Direct input neurons:        {len(input_neurons):,}")
    print(f"Indirect readout neurons:    {len(readout_population):,}")

    # --------------------------------------------------------
    # Strong compatibility check against original Step-62
    # feature metadata, when that file is available.
    # --------------------------------------------------------

    if REFERENCE_FEATURE_FILE.exists():
        reference = np.load(
            REFERENCE_FEATURE_FILE,
            allow_pickle=False,
        )

        if "input_neurons" not in reference.files:
            raise ValueError(
                "Reference Step-62 feature file has no input_neurons metadata."
            )

        if "readout_population" not in reference.files:
            raise ValueError(
                "Reference Step-62 feature file has no readout_population metadata."
            )

        old_input = reference["input_neurons"].astype(np.int64)
        old_readout = reference["readout_population"].astype(np.int64)

        if not np.array_equal(old_input, input_neurons):
            raise ValueError(
                "Frozen input neurons do not match original Step-62 features."
            )

        if not np.array_equal(old_readout, readout_population):
            raise ValueError(
                "Frozen readout population does not match original Step-62 features."
            )

        print("Step-62 population check:    PASS")
    else:
        print("Step-62 population check:    SKIPPED (reference file absent)")

    feature_dimensions = (
        len(FEATURE_BLOCK_NAMES)
        * len(readout_population)
    )

    print(f"Feature blocks:              {len(FEATURE_BLOCK_NAMES)}")
    print(f"Expected feature dimensions: {feature_dimensions:,}")
    print()

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    # ========================================================
    # Frozen Step-62 prompt stimulation
    # ========================================================

    def stimulate_prompt(prompt_tokens):
        states = []

        for position, token in enumerate(prompt_tokens):
            if position >= MAX_PROMPT_POSITIONS:
                raise ValueError(
                    f"Prompt has more than {MAX_PROMPT_POSITIONS} tokens: "
                    f"{prompt_tokens}"
                )

            neurons = dual_map.prompt_neurons_for(
                token,
                position,
            )

            for _ in range(PROMPT_TOKEN_STEPS):
                fired = brain.step(
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ]
                )

                state = np.asarray(
                    trace.observe(fired)
                ).copy()

                states.append(state)

            for _ in range(PROMPT_GAP_STEPS):
                fired = brain.step()

                state = np.asarray(
                    trace.observe(fired)
                ).copy()

                states.append(state)

        states = np.stack(states)

        return (
            states.mean(axis=0),
            states.max(axis=0),
            states[-1],
        )

    # ========================================================
    # Frozen Step-62 feedback stimulation
    # ========================================================

    def stimulate_feedback(token):
        neurons = dual_map.feedback_neurons_for(token)

        states = []

        for _ in range(FEEDBACK_TOKEN_STEPS):
            fired = brain.step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ]
            )

            state = np.asarray(
                trace.observe(fired)
            ).copy()

            states.append(state)

        states = np.stack(states)

        feedback_mean = states.mean(axis=0)
        feedback_max = states.max(axis=0)
        feedback_final = states[-1]

        delayed_state = feedback_final

        for _ in range(FEEDBACK_GAP_STEPS):
            fired = brain.step()

            delayed_state = np.asarray(
                trace.observe(fired)
            ).copy()

        return (
            feedback_mean,
            feedback_max,
            feedback_final,
            delayed_state,
        )

    # ========================================================
    # One complete teacher-forced trajectory
    # ========================================================

    def run_generator_episode(example_samples, seed):
        brain.reset(seed=seed)
        trace.reset()

        for _ in range(WARMUP_STEPS):
            fired = brain.step()
            trace.observe(fired)

        prompt_tokens = example_samples[0].prompt_tokens

        for sample in example_samples:
            if sample.prompt_tokens != prompt_tokens:
                raise ValueError(
                    f"Prompt mismatch inside logical example {sample.example_id}"
                )

        (
            prompt_mean,
            prompt_max,
            prompt_final,
        ) = stimulate_prompt(prompt_tokens)

        rows = []

        for position, sample in enumerate(example_samples):
            feedback_token = (
                "BOS"
                if position == 0
                else example_samples[position - 1].target_token
            )

            expected_context = sample.context_tokens

            if expected_context[-1] != feedback_token:
                raise ValueError(
                    f"Feedback mismatch for {sample.sample_id}: "
                    f"expected {expected_context[-1]!r}, "
                    f"got {feedback_token!r}"
                )

            (
                feedback_mean,
                feedback_max,
                feedback_final,
                delayed_state,
            ) = stimulate_feedback(feedback_token)

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

            if feature.shape != (feature_dimensions,):
                raise RuntimeError(
                    f"Feature shape mismatch: expected "
                    f"({feature_dimensions},), got {feature.shape}"
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
    # Extract the frozen final holdout
    # ========================================================

    features = []
    targets = []
    sample_ids = []
    example_ids = []
    intents = []
    target_positions = []
    feedback_tokens_used = []
    seeds_used = []

    logical_prompts = sorted(grouped.keys())

    total_episodes = (
        len(logical_prompts)
        * len(NEURAL_SEEDS)
    )

    if total_episodes != EXPECTED_NEURAL_EPISODES:
        raise ValueError(
            f"Expected {EXPECTED_NEURAL_EPISODES} neural episodes, "
            f"got {total_episodes}"
        )

    episode_number = 0

    print("====================================================")
    print("       EXTRACTING FINAL HOLDOUT ON MALECNS")
    print("====================================================")
    print()

    for example_id in logical_prompts:
        example_samples = grouped[example_id]

        for seed in NEURAL_SEEDS:
            episode_number += 1

            print(
                f"[{episode_number:03}/{total_episodes:03}] "
                f"{example_id:20} "
                f"seed={seed} "
                f"steps={len(example_samples)}"
            )

            rows = run_generator_episode(
                example_samples,
                seed,
            )

            for feature, sample, feedback_token in rows:
                features.append(feature)
                targets.append(sample.target_token)
                sample_ids.append(sample.sample_id)
                example_ids.append(sample.example_id)
                intents.append(sample.intent)
                target_positions.append(sample.target_position)
                feedback_tokens_used.append(feedback_token)
                seeds_used.append(seed)

    X = np.stack(features).astype(
        np.float32,
        copy=False,
    )

    targets_array = np.asarray(targets)
    sample_ids_array = np.asarray(sample_ids)
    example_ids_array = np.asarray(example_ids)
    intents_array = np.asarray(intents)

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

    if X.shape != (
        EXPECTED_NEURAL_ROWS,
        feature_dimensions,
    ):
        raise ValueError(
            "Final feature matrix has unexpected shape: "
            f"{X.shape}; expected "
            f"({EXPECTED_NEURAL_ROWS}, {feature_dimensions})"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Final holdout contains non-finite neural features."
        )

    # Every teacher-forcing row should have exactly five neural replicas.
    unique_samples, sample_counts = np.unique(
        sample_ids_array,
        return_counts=True,
    )

    if len(unique_samples) != EXPECTED_TEACHER_FORCING_ROWS:
        raise ValueError(
            "Unexpected number of unique final teacher-forcing samples."
        )

    if not np.all(sample_counts == len(NEURAL_SEEDS)):
        raise ValueError(
            "Every final teacher-forcing sample must have exactly "
            f"{len(NEURAL_SEEDS)} neural replicas."
        )

    if set(np.unique(seeds_array).tolist()) != set(NEURAL_SEEDS):
        raise ValueError("Final neural seed set changed.")

    # ========================================================
    # Save final features
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_FILE,
        X=X,
        targets=targets_array,
        sample_ids=sample_ids_array,
        example_ids=example_ids_array,
        intents=intents_array,
        target_positions=positions_array,
        feedback_tokens=feedback_array,
        seeds=seeds_array,
        input_neurons=input_neurons,
        readout_population=readout_population,
        feature_block_names=np.asarray(FEATURE_BLOCK_NAMES),
        block_size=np.asarray(
            len(readout_population),
            dtype=np.int32,
        ),
        trace_tau=np.asarray(
            TRACE_TAU,
            dtype=np.float32,
        ),
        amplitude=np.asarray(
            AMPLITUDE,
            dtype=np.float32,
        ),
        mapping_seed=np.asarray(
            MAP_SEED,
            dtype=np.int32,
        ),
        final_holdout_sha256=np.asarray(
            sha256_file(FINAL_HOLDOUT_FILE)
        ),
        final_generator_sha256=np.asarray(
            sha256_file(FINAL_GENERATOR_FILE)
        ),
        frozen_mapping_sha256=np.asarray(
            sha256_file(FROZEN_MAP_FILE)
        ),
        protocol_sha256=np.asarray(
            sha256_file(PROTOCOL_FILE)
        ),
    )

    output_size_mb = (
        OUTPUT_FILE.stat().st_size
        / 1024
        / 1024
    )

    print()
    print("====================================================")
    print("                 STEP 77C COMPLETE")
    print("====================================================")
    print()
    print(f"Logical prompts:             {len(logical_prompts)}")
    print(f"Neural episodes:             {total_episodes}")
    print(f"Teacher-forcing rows:        {len(samples)}")
    print(f"Final neural rows:           {X.shape[0]}")
    print(f"Feature dimensions:          {X.shape[1]:,}")
    print(f"Finite:                      {np.isfinite(X).all()}")
    print(f"Direct input neurons:        {len(input_neurons):,}")
    print(f"Indirect readout neurons:    {len(readout_population):,}")
    print(f"Output:                      {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Output size:                 {output_size_mb:.2f} MB")
    print()
    print(
        "No architecture search, decoder fitting, or final-holdout "
        "model selection was performed."
    )
    print()
    print(
        "Next: Step 77D - score the already-refit frozen decoder on "
        "these final holdout features."
    )


if __name__ == "__main__":
    main()
