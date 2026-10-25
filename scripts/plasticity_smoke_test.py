from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import DualChannelMap


# ============================================================
# Phase 2 — Plastic PyFly
# Step 79A — Synaptic Plasticity Smoke Test
#
# This is the FIRST step that actually changes recurrent
# MaleCNS synaptic weights.
#
# What it does:
#   1. Load the real MaleCNS on CUDA.
#   2. Load the frozen PyFly input mapping.
#   3. Select a reproducible subset of REAL recurrent synapses
#      inside the visual_projection pathway.
#   4. Save their exact original weights.
#   5. Modify those weights IN GPU MEMORY.
#   6. Prove the neural response changes.
#   7. Restore the exact original weights.
#   8. Prove the original neural response returns.
#
# IMPORTANT:
#   * ~/fly-data/weights.npz is NEVER modified.
#   * This script is a safety/infrastructure test.
#   * Actual task learning starts in Step 79B after this passes.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda"

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sequence_challenge.jsonl"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)

BASELINE_CONTROL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "shuffled_label_final_control.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "plasticity"
)

SYNAPSE_MAP_FILE = (
    OUTPUT_DIR
    / "plastic_synapse_subset.npz"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "plasticity_smoke_test.json"
)


# ============================================================
# Frozen PyFly stimulation settings
# ============================================================

MAX_PROMPT_POSITIONS = 6

PROMPT_NEURONS_PER_CHANNEL = 16
FEEDBACK_NEURONS_PER_TOKEN = 32

PROMPT_TOKEN_STEPS = 8
PROMPT_GAP_STEPS = 2

WARMUP_STEPS = 10

AMPLITUDE = 0.8
TRACE_TAU = 0.25

MAP_SEED = 123


# ============================================================
# Plasticity smoke-test settings
#
# These values are PRECOMMITTED for this test.
# ============================================================

PLASTICITY_SEED = 79001

MAX_PLASTIC_SYNAPSES = 100_000

# Increase absolute magnitude by 10%, preserving sign.
SMOKE_TEST_SCALE = 1.10

PROBE_BRAIN_SEED = 79002

RESTORE_ATOL = 1e-6
RESTORE_RTOL = 1e-6


# ============================================================
# Expected project sizes
# ============================================================

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873


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


# ============================================================
# Load exact frozen PyFly dual-channel map
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

    result = DualChannelMap(
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
        result.prompt_neurons_per_channel
        !=
        PROMPT_NEURONS_PER_CHANNEL
    ):

        raise ValueError(
            "Frozen prompt mapping width changed."
        )

    if (
        result.feedback_neurons_per_token
        !=
        FEEDBACK_NEURONS_PER_TOKEN
    ):

        raise ValueError(
            "Frozen feedback mapping width changed."
        )

    if (
        result.max_prompt_positions
        !=
        MAX_PROMPT_POSITIONS
    ):

        raise ValueError(
            "Frozen prompt-position limit changed."
        )

    if (
        result.seed
        !=
        MAP_SEED
    ):

        raise ValueError(
            "Frozen neural map seed changed."
        )

    return result


# ============================================================
# Select REAL recurrent synapses for plasticity
#
# CUDA FlyBrain stores _W as CSR:
#
#   row    = postsynaptic neuron
#   column = presynaptic neuron
#
# We select synapses:
#
#   presynaptic  ∈ visual_projection
#   postsynaptic ∈ INDIRECT visual_projection readout
#
# This gives us a real recurrent visual subnetwork while
# preserving the old rule that directly stimulated neurons
# are not themselves used as the readout population.
# ============================================================

def select_plastic_synapses(
    brain: FlyBrain,
    visual_pool: np.ndarray,
    readout_population: np.ndarray,
):

    if brain.device != "cuda":

        raise RuntimeError(
            "Step 79A requires CUDA."
        )

    W = brain._W

    if str(
        W.format
    ).lower() != "csr":

        raise RuntimeError(
            f"Expected CUDA CSR matrix, "
            f"got {W.format!r}"
        )

    indptr = (
        W.indptr.get().astype(
            np.int64,
            copy=False,
        )
    )

    indices = (
        W.indices.get().astype(
            np.int64,
            copy=False,
        )
    )

    data = (
        W.data.get().astype(
            np.float32,
            copy=False,
        )
    )

    if (
        len(
            data
        )
        !=
        EXPECTED_CONNECTIONS
    ):

        raise ValueError(
            "Unexpected CUDA sparse-matrix "
            f"edge count: {len(data):,}"
        )

    is_visual = np.zeros(
        brain.n,
        dtype=bool,
    )

    is_visual[
        visual_pool
    ] = True

    edge_offsets = []
    pre_neurons = []
    post_neurons = []

    # --------------------------------------------------------
    # Iterate only 5,873 readout rows, not all 166k rows.
    # --------------------------------------------------------

    for post in readout_population:

        start = int(
            indptr[
                post
            ]
        )

        end = int(
            indptr[
                post + 1
            ]
        )

        if start == end:
            continue

        row_pre = indices[
            start:end
        ]

        mask = is_visual[
            row_pre
        ]

        if not np.any(
            mask
        ):
            continue

        local_offsets = np.flatnonzero(
            mask
        )

        selected_offsets = (
            start
            +
            local_offsets
        )

        edge_offsets.append(
            selected_offsets.astype(
                np.int64,
                copy=False,
            )
        )

        pre_neurons.append(
            row_pre[
                local_offsets
            ].astype(
                np.int64,
                copy=False,
            )
        )

        post_neurons.append(
            np.full(
                len(
                    local_offsets
                ),
                int(
                    post
                ),
                dtype=np.int64,
            )
        )

    if not edge_offsets:

        raise RuntimeError(
            "No eligible visual_projection "
            "synapses were found."
        )

    edge_offsets = np.concatenate(
        edge_offsets
    )

    pre_neurons = np.concatenate(
        pre_neurons
    )

    post_neurons = np.concatenate(
        post_neurons
    )

    eligible_count = len(
        edge_offsets
    )

    # --------------------------------------------------------
    # Reproducible subset
    # --------------------------------------------------------

    rng = np.random.default_rng(
        PLASTICITY_SEED
    )

    if (
        eligible_count
        >
        MAX_PLASTIC_SYNAPSES
    ):

        chosen = rng.choice(
            eligible_count,
            size=MAX_PLASTIC_SYNAPSES,
            replace=False,
        )

        chosen.sort()

        edge_offsets = edge_offsets[
            chosen
        ]

        pre_neurons = pre_neurons[
            chosen
        ]

        post_neurons = post_neurons[
            chosen
        ]

    original_weights = data[
        edge_offsets
    ].copy()

    if np.any(
        original_weights
        ==
        0
    ):

        raise RuntimeError(
            "Plastic subset unexpectedly "
            "contains zero-weight edges."
        )

    return {
        "eligible_count": int(
            eligible_count
        ),

        "edge_offsets": (
            edge_offsets
        ),

        "pre_neurons": (
            pre_neurons
        ),

        "post_neurons": (
            post_neurons
        ),

        "original_weights": (
            original_weights
        ),
    }


# ============================================================
# Run one deterministic prompt probe
# ============================================================

def run_prompt_probe(
    brain: FlyBrain,
    trace: Trace,
    dual_map: DualChannelMap,
    prompt_tokens: list[str],
) -> np.ndarray:

    brain.reset(
        seed=PROBE_BRAIN_SEED
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

    states = []

    # --------------------------------------------------------
    # Same frozen position-aware prompt stimulation
    # --------------------------------------------------------

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

            states.append(
                np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()
            )

        for _ in range(
            PROMPT_GAP_STEPS
        ):

            fired = brain.step()

            states.append(
                np.asarray(
                    trace.observe(
                        fired
                    )
                ).copy()
            )

    states = np.stack(
        states
    )

    # --------------------------------------------------------
    # Compact deterministic probe vector
    # --------------------------------------------------------

    return np.concatenate(
        [
            states.mean(
                axis=0
            ),

            states.max(
                axis=0
            ),

            states[
                -1
            ],
        ]
    ).astype(
        np.float32,
        copy=False,
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
        "       PHASE 2 - PLASTIC PYFLY"
    )

    print(
        "       STEP 79A - SYNAPTIC SMOKE TEST"
    )

    print(
        "===================================================="
    )

    print()

    # --------------------------------------------------------
    # Require completed baseline/control phase.
    # --------------------------------------------------------

    if not BASELINE_CONTROL_FILE.exists():

        raise FileNotFoundError(
            "Step 78C must be complete before "
            "starting Phase 2."
        )

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset missing: {DATASET_FILE}"
        )

    if not MAPPING_FILE.exists():

        raise FileNotFoundError(
            f"Frozen neural map missing: "
            f"{MAPPING_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Baseline/control phase:      COMPLETE"
    )

    print(
        f"Mapping SHA-256:             "
        f"{sha256_file(MAPPING_FILE)}"
    )

    # ========================================================
    # Load prompt + frozen mapping
    # ========================================================

    examples = load_dataset(
        DATASET_FILE
    )

    if not examples:

        raise RuntimeError(
            "Development dataset is empty."
        )

    # Use a fixed development prompt only as a deterministic
    # engineering probe, NOT as a performance metric.
    probe_example = examples[
        0
    ]

    prompt_tokens = list(
        probe_example.prompt_tokens
    )

    dual_map = (
        load_frozen_dual_map(
            MAPPING_FILE
        )
    )

    print()

    print(
        f"Engineering probe ID:       "
        f"{probe_example.id}"
    )

    print(
        f"Probe prompt:               "
        f"{' '.join(prompt_tokens)}"
    )

    # ========================================================
    # Load REAL MaleCNS
    # ========================================================

    print()

    print(
        "Loading real MaleCNS on CUDA..."
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
            f"Expected "
            f"{EXPECTED_NEURONS:,} neurons, "
            f"got {brain.n:,}"
        )

    if (
        brain._W.nnz
        !=
        EXPECTED_CONNECTIONS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_CONNECTIONS:,} "
            "recurrent edges, got "
            f"{brain._W.nnz:,}"
        )

    print(
        f"Neurons:                    "
        f"{brain.n:,}"
    )

    print(
        f"Recurrent edges:            "
        f"{brain._W.nnz:,}"
    )

    # ========================================================
    # Same PyFly populations
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
            "Indirect-readout population changed."
        )

    print(
        f"Visual-projection neurons:  "
        f"{len(visual_pool):,}"
    )

    print(
        f"Direct input neurons:       "
        f"{len(input_neurons):,}"
    )

    print(
        f"Indirect readout neurons:   "
        f"{len(readout_population):,}"
    )

    # ========================================================
    # Select real recurrent synapses
    # ========================================================

    print()

    print(
        "Selecting real MaleCNS synapses "
        "for plasticity..."
    )

    plastic = select_plastic_synapses(
        brain=brain,
        visual_pool=visual_pool,
        readout_population=(
            readout_population
        ),
    )

    offsets = plastic[
        "edge_offsets"
    ]

    pre_neurons = plastic[
        "pre_neurons"
    ]

    post_neurons = plastic[
        "post_neurons"
    ]

    original_weights = plastic[
        "original_weights"
    ]

    print(
        f"Eligible visual recurrent:  "
        f"{plastic['eligible_count']:,}"
    )

    print(
        f"Selected plastic synapses:  "
        f"{len(offsets):,}"
    )

    print(
        f"Excitatory selected:        "
        f"{np.sum(original_weights > 0):,}"
    )

    print(
        f"Inhibitory selected:        "
        f"{np.sum(original_weights < 0):,}"
    )

    print(
        f"Mean |weight|:              "
        f"{np.mean(np.abs(original_weights)):.6f}"
    )

    # ========================================================
    # Save exact original selected weights BEFORE mutation
    # ========================================================

    np.savez(
        SYNAPSE_MAP_FILE,

        edge_offsets=(
            offsets
        ),

        pre_neurons=(
            pre_neurons
        ),

        post_neurons=(
            post_neurons
        ),

        original_weights=(
            original_weights
        ),

        plasticity_seed=np.asarray(
            PLASTICITY_SEED,
            dtype=np.int64,
        ),

        mapping_sha256=np.asarray(
            sha256_file(
                MAPPING_FILE
            )
        ),
    )

    checkpoint_hash = (
        sha256_file(
            SYNAPSE_MAP_FILE
        )
    )

    print()

    print(
        "Original plastic subset saved:"
    )

    print(
        "  "
        f"{normalized_relative(SYNAPSE_MAP_FILE)}"
    )

    print(
        f"Checkpoint SHA-256:         "
        f"{checkpoint_hash}"
    )

    # ========================================================
    # Probe BEFORE modification
    # ========================================================

    trace = Trace(
        brain,
        idx=readout_population,
        tau=TRACE_TAU,
    )

    print()

    print(
        "Running baseline neural probe..."
    )

    before = run_prompt_probe(
        brain=brain,
        trace=trace,
        dual_map=dual_map,
        prompt_tokens=(
            prompt_tokens
        ),
    )

    # ========================================================
    # ACTUALLY MODIFY REAL RECURRENT WEIGHTS ON GPU
    # ========================================================

    import cupy as cp

    gpu_offsets = cp.asarray(
        offsets
    )

    gpu_original = (
        brain._W.data[
            gpu_offsets
        ].copy()
    )

    # Sign-preserving multiplicative change.
    brain._W.data[
        gpu_offsets
    ] = (
        gpu_original
        *
        np.float32(
            SMOKE_TEST_SCALE
        )
    )

    modified_cpu = (
        brain._W.data[
            gpu_offsets
        ].get()
    )

    changed_weights = int(
        np.sum(
            modified_cpu
            !=
            original_weights
        )
    )

    if (
        changed_weights
        !=
        len(
            offsets
        )
    ):

        raise RuntimeError(
            "Not every selected recurrent "
            "synapse changed on GPU."
        )

    sign_preserved = bool(
        np.all(
            np.sign(
                modified_cpu
            )
            ==
            np.sign(
                original_weights
            )
        )
    )

    if not sign_preserved:

        raise RuntimeError(
            "Smoke-test update changed "
            "synapse signs."
        )

    print()

    print(
        "REAL MaleCNS synapses changed "
        "in GPU memory."
    )

    print(
        f"Changed recurrent weights:  "
        f"{changed_weights:,}"
    )

    print(
        f"Magnitude scale:            "
        f"{SMOKE_TEST_SCALE:.3f}x"
    )

    print(
        f"Synapse signs preserved:    "
        f"{sign_preserved}"
    )

    # ========================================================
    # Probe AFTER modification
    # ========================================================

    print()

    print(
        "Running probe with modified synapses..."
    )

    after = run_prompt_probe(
        brain=brain,
        trace=trace,
        dual_map=dual_map,
        prompt_tokens=(
            prompt_tokens
        ),
    )

    delta = (
        after
        -
        before
    )

    l2_change = float(
        np.linalg.norm(
            delta
        )
    )

    max_change = float(
        np.max(
            np.abs(
                delta
            )
        )
    )

    changed_probe_values = int(
        np.sum(
            np.abs(
                delta
            )
            >
            1e-8
        )
    )

    print(
        f"Probe L2 change:            "
        f"{l2_change:.6f}"
    )

    print(
        f"Probe max |change|:         "
        f"{max_change:.6f}"
    )

    print(
        f"Changed probe values:       "
        f"{changed_probe_values:,}"
    )

    if (
        l2_change
        <=
        0.0
    ):

        raise RuntimeError(
            "Changing recurrent synapses "
            "did not alter the neural probe."
        )

    # ========================================================
    # RESTORE exact original GPU weights
    # ========================================================

    print()

    print(
        "Restoring original recurrent weights..."
    )

    brain._W.data[
        gpu_offsets
    ] = (
        gpu_original
    )

    restored_weights = (
        brain._W.data[
            gpu_offsets
        ].get()
    )

    if not np.array_equal(
        restored_weights,
        original_weights,
    ):

        raise RuntimeError(
            "Selected synapses were not "
            "restored exactly."
        )

    # ========================================================
    # Probe AFTER restoration
    # ========================================================

    restored = run_prompt_probe(
        brain=brain,
        trace=trace,
        dual_map=dual_map,
        prompt_tokens=(
            prompt_tokens
        ),
    )

    restore_max_error = float(
        np.max(
            np.abs(
                restored
                -
                before
            )
        )
    )

    restore_match = bool(
        np.allclose(
            restored,
            before,
            rtol=RESTORE_RTOL,
            atol=RESTORE_ATOL,
        )
    )

    if not restore_match:

        raise RuntimeError(
            "Neural response did not return "
            "to baseline after restoring "
            "the original synapses."
        )

    print(
        f"Exact selected weights back: True"
    )

    print(
        f"Baseline neural response back:"
        f" {restore_match}"
    )

    print(
        f"Restore max probe error:     "
        f"{restore_max_error:.9f}"
    )

    # ========================================================
    # Save manifest
    # ========================================================

    manifest = {
        "phase": 2,

        "step": "79A",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "PLASTICITY_INFRASTRUCTURE_VERIFIED"
        ),

        "device": DEVICE,

        "plasticity_seed": (
            PLASTICITY_SEED
        ),

        "probe_brain_seed": (
            PROBE_BRAIN_SEED
        ),

        "probe_example_id": (
            probe_example.id
        ),

        "probe_prompt_tokens": (
            prompt_tokens
        ),

        "network": {
            "neurons": int(
                brain.n
            ),

            "recurrent_edges": int(
                brain._W.nnz
            ),

            "visual_projection_neurons": int(
                len(
                    visual_pool
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
        },

        "plastic_subset": {
            "eligible_synapses": int(
                plastic[
                    "eligible_count"
                ]
            ),

            "selected_synapses": int(
                len(
                    offsets
                )
            ),

            "excitatory_synapses": int(
                np.sum(
                    original_weights
                    >
                    0
                )
            ),

            "inhibitory_synapses": int(
                np.sum(
                    original_weights
                    <
                    0
                )
            ),

            "mean_abs_original_weight": float(
                np.mean(
                    np.abs(
                        original_weights
                    )
                )
            ),

            "checkpoint_file": (
                normalized_relative(
                    SYNAPSE_MAP_FILE
                )
            ),

            "checkpoint_sha256": (
                checkpoint_hash
            ),
        },

        "smoke_update": {
            "scale": (
                SMOKE_TEST_SCALE
            ),

            "changed_synapses": (
                changed_weights
            ),

            "signs_preserved": (
                sign_preserved
            ),

            "probe_l2_change": (
                l2_change
            ),

            "probe_max_abs_change": (
                max_change
            ),

            "changed_probe_values": (
                changed_probe_values
            ),
        },

        "restore": {
            "selected_weights_restored_exactly": (
                True
            ),

            "baseline_probe_restored": (
                restore_match
            ),

            "max_probe_error": (
                restore_max_error
            ),
        },

        "disk_connectome_modified": (
            False
        ),

        "mapping_sha256": (
            sha256_file(
                MAPPING_FILE
            )
        ),

        "next_step": (
            "79B reward-modulated supervised "
            "plasticity training"
        ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
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
        f"{normalized_relative(SYNAPSE_MAP_FILE)}"
    )

    print(
        "  "
        f"{normalized_relative(MANIFEST_FILE)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 79A COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Plastic recurrent MaleCNS synapses are "
        "now verified to be writable and active."
    )

    print()

    print(
        "The original disk connectome is untouched."
    )

    print()

    print(
        "Next: Step 79B - actual task learning with "
        "reward-modulated synaptic updates."
    )


if __name__ == "__main__":
    main()
