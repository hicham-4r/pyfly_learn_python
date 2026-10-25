from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flybrain import FlyBrain, Trace

from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl
from pyfly.tokenizer import tokens_to_code


# ============================================================
# Phase 3 — Trainable PyFly
# Step 84 — Interactive PyFly Console
#
# This is an INTERACTIVE inference application.
#
# It loads the frozen Step-80 model and runs the exact
# Step-81/83 true closed-loop generation pipeline:
#
#   prompt tokens
#        ↓
#   real MaleCNS simulation
#        ↓
#   frozen PyTorch adapter
#        ↓
#   predicted Python token
#        ↓
#   predicted token fed back into MaleCNS
#        ↓
#   next token
#
# NO training.
# NO optimizer.
# NO weight updates.
#
# Input modes:
#
#   list
#       Show known logical prompts from project data.
#
#   show <example_id>
#       Show one prompt's tokens.
#
#   run <example_id>
#       Generate code for a known prompt.
#
#   tokens TOKEN_1 TOKEN_2 ...
#       Run an exact supported prompt-token sequence.
#
#   seed <integer>
#       Change the MaleCNS neural seed for interactive use.
#
#   help
#   quit
#
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_NAME = "cuda"


# ============================================================
# Frozen Phase-3 artifacts
# ============================================================

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "phase3_pyfly_adapter.pt"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "phase3_pyfly_adapter_scaler.npz"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "mappings"
    / "generator_dual_channel_map.npz"
)


# ============================================================
# Prompt catalog sources
#
# These are used ONLY to discover prompt examples.
# They are never used to train or modify the model.
# ============================================================

PROMPT_CATALOG_FILES = [
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_train.jsonl"
    ),

    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_validation.jsonl"
    ),

    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "generator_test.jsonl"
    ),
]


# ============================================================
# Frozen MaleCNS stimulation configuration
# ============================================================

PROMPT_TOKEN_STEPS = 8
PROMPT_GAP_STEPS = 2

FEEDBACK_TOKEN_STEPS = 8
FEEDBACK_GAP_STEPS = 1

WARMUP_STEPS = 10

AMPLITUDE = 0.8
TRACE_TAU = 0.25

EXPECTED_NEURONS = 166_700
EXPECTED_CONNECTIONS = 25_582_938

EXPECTED_DIRECT_INPUT_NEURONS = 3_328
EXPECTED_READOUT_NEURONS = 5_873
EXPECTED_FEATURE_DIM = 41_111

DEFAULT_NEURAL_SEED = 0


# ============================================================
# Exact Step-80 model architecture
# ============================================================

class PyFlyNeuralAdapter(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_position: int,
        hidden_dim: int,
        second_hidden_dim: int,
        position_embed_dim: int,
        dropout: float,
        input_dropout: float,
    ) -> None:

        super().__init__()

        self.input_norm = nn.LayerNorm(
            input_dim
        )

        self.input_dropout = nn.Dropout(
            input_dropout
        )

        self.neural_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.position_embedding = nn.Embedding(
            max_position + 1,
            position_embed_dim,
        )

        merged_dim = (
            hidden_dim
            +
            position_embed_dim
        )

        self.adapter = nn.Sequential(
            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                merged_dim,
                second_hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.LayerNorm(
                second_hidden_dim
            ),
        )

        self.classifier = nn.Linear(
            second_hidden_dim,
            num_classes,
        )

    def forward(
        self,
        X: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:

        X = self.input_norm(
            X
        )

        X = self.input_dropout(
            X
        )

        neural_hidden = (
            self.neural_projection(
                X
            )
        )

        position_hidden = (
            self.position_embedding(
                positions
            )
        )

        merged = torch.cat(
            [
                neural_hidden,
                position_hidden,
            ],
            dim=1,
        )

        hidden = self.adapter(
            merged
        )

        return self.classifier(
            hidden
        )


# ============================================================
# Prompt catalog
# ============================================================

@dataclass(
    frozen=True
)
class PromptEntry:

    example_id: str
    intent: str
    prompt_tokens: tuple[str, ...]


def load_prompt_catalog() -> dict[str, PromptEntry]:

    catalog: dict[
        str,
        PromptEntry,
    ] = {}

    for path in PROMPT_CATALOG_FILES:

        if not path.exists():

            continue

        samples = load_generator_jsonl(
            path
        )

        grouped = {}

        for sample in samples:

            grouped.setdefault(
                sample.example_id,
                [],
            ).append(
                sample
            )

        for example_id, rows in grouped.items():

            rows.sort(
                key=lambda row:
                    row.target_position
            )

            first = rows[
                0
            ]

            prompt_tokens = tuple(
                str(
                    token
                )

                for token
                in first.prompt_tokens
            )

            intent = str(
                first.intent
            )

            catalog[
                str(
                    example_id
                )
            ] = PromptEntry(
                example_id=str(
                    example_id
                ),

                intent=(
                    intent
                ),

                prompt_tokens=(
                    prompt_tokens
                ),
            )

    return catalog


# ============================================================
# Frozen neural mapping
# ============================================================

def load_dual_map(
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

    prompt_mapping = {}

    for index, (
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
                index
            ].copy()
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

    feedback_mapping = {}

    for index, token in enumerate(
        feedback_tokens
    ):

        feedback_mapping[
            str(
                token
            )
        ] = (
            feedback_neurons[
                index
            ].copy()
        )

    return DualChannelMap(
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


# ============================================================
# Interactive application
# ============================================================

class PyFlyConsole:

    def __init__(
        self,
    ) -> None:

        print()

        print(
            "===================================================="
        )

        print(
            "       PYFLY INTERACTIVE CONSOLE"
        )

        print(
            "===================================================="
        )

        print()

        # ----------------------------------------------------
        # Required files
        # ----------------------------------------------------

        for path in [
            MODEL_FILE,
            SCALER_FILE,
            MAPPING_FILE,
        ]:

            if not path.exists():

                raise FileNotFoundError(
                    f"Required file missing: {path}"
                )

        # ----------------------------------------------------
        # CUDA
        # ----------------------------------------------------

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is not available to PyTorch."
            )

        self.device = torch.device(
            DEVICE_NAME
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        checkpoint = torch.load(
            MODEL_FILE,
            map_location=self.device,
            weights_only=False,
        )

        self.token_classes = [
            str(
                token
            )

            for token
            in checkpoint[
                "token_classes"
            ]
        ]

        self.max_position = int(
            checkpoint[
                "max_position"
            ]
        )

        self.max_generation_steps = (
            self.max_position
            +
            1
        )

        self.model = PyFlyNeuralAdapter(
            input_dim=int(
                checkpoint[
                    "input_dim"
                ]
            ),

            num_classes=len(
                self.token_classes
            ),

            max_position=(
                self.max_position
            ),

            hidden_dim=int(
                checkpoint[
                    "hidden_dim"
                ]
            ),

            second_hidden_dim=int(
                checkpoint[
                    "second_hidden_dim"
                ]
            ),

            position_embed_dim=int(
                checkpoint[
                    "position_embed_dim"
                ]
            ),

            dropout=float(
                checkpoint[
                    "dropout"
                ]
            ),

            input_dropout=float(
                checkpoint[
                    "input_dropout"
                ]
            ),
        ).to(
            self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        print(
            f"Frozen model epoch: "
            f"{checkpoint.get('best_epoch')}"
        )

        print(
            f"Output classes: "
            f"{len(self.token_classes)}"
        )

        # ----------------------------------------------------
        # Scaler
        # ----------------------------------------------------

        scaler = np.load(
            SCALER_FILE,
            allow_pickle=False,
        )

        self.feature_mean = (
            scaler[
                "mean"
            ].astype(
                np.float32
            )
        )

        self.feature_std = (
            scaler[
                "std"
            ].astype(
                np.float32
            )
        )

        if (
            self.feature_mean.shape
            !=
            (
                EXPECTED_FEATURE_DIM,
            )
            or
            self.feature_std.shape
            !=
            (
                EXPECTED_FEATURE_DIM,
            )
        ):

            raise ValueError(
                "Frozen scaler dimensions changed."
            )

        # ----------------------------------------------------
        # Prompt catalog
        # ----------------------------------------------------

        self.catalog = (
            load_prompt_catalog()
        )

        # ----------------------------------------------------
        # Neural map
        # ----------------------------------------------------

        self.dual_map = load_dual_map(
            MAPPING_FILE
        )

        required_feedback = (
            set(
                self.token_classes
            )
            |
            {
                "BOS"
            }
        )

        missing_feedback = (
            required_feedback
            -
            set(
                self.dual_map.feedback_mapping.keys()
            )
        )

        if missing_feedback:

            raise ValueError(
                "Frozen feedback mapping missing: "
                f"{sorted(missing_feedback)}"
            )

        # ----------------------------------------------------
        # MaleCNS
        # ----------------------------------------------------

        print()

        print(
            "Loading real MaleCNS on CUDA..."
        )

        self.brain = FlyBrain(
            device="cuda",
            seed=0,
        )

        if (
            self.brain.n
            !=
            EXPECTED_NEURONS
        ):

            raise ValueError(
                "Unexpected MaleCNS neuron count."
            )

        if (
            self.brain._W.nnz
            !=
            EXPECTED_CONNECTIONS
        ):

            raise ValueError(
                "Unexpected MaleCNS recurrent edge count."
            )

        visual_pool = np.asarray(
            self.brain.cells(
                [
                    "visual_projection"
                ]
            ),
            dtype=np.int64,
        )

        input_neurons = (
            self.dual_map.all_input_neurons()
        )

        self.readout_population = np.setdiff1d(
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
                "Frozen direct-input population changed."
            )

        if (
            len(
                self.readout_population
            )
            !=
            EXPECTED_READOUT_NEURONS
        ):

            raise ValueError(
                "Frozen readout population changed."
            )

        self.trace = Trace(
            self.brain,
            idx=self.readout_population,
            tau=TRACE_TAU,
        )

        self.neural_seed = (
            DEFAULT_NEURAL_SEED
        )

        print(
            f"MaleCNS neurons: "
            f"{self.brain.n:,}"
        )

        print(
            f"Readout neurons: "
            f"{len(self.readout_population):,}"
        )

        print(
            f"Known prompt examples: "
            f"{len(self.catalog)}"
        )

        print()

        self.print_help()

    # ========================================================
    # UI
    # ========================================================

    def print_help(
        self,
    ) -> None:

        print(
            "Commands:"
        )

        print(
            "  list"
        )

        print(
            "      List known prompt examples."
        )

        print(
            "  list <intent>"
        )

        print(
            "      List prompts for one intent."
        )

        print(
            "  show <example_id>"
        )

        print(
            "      Show prompt tokens for an example."
        )

        print(
            "  run <example_id>"
        )

        print(
            "      Generate Python for a known example."
        )

        print(
            "  tokens TOKEN_1 TOKEN_2 ..."
        )

        print(
            "      Generate from an exact prompt-token sequence."
        )

        print(
            "  seed <integer>"
        )

        print(
            "      Change the interactive MaleCNS neural seed."
        )

        print(
            "  classes"
        )

        print(
            "      Show output token classes."
        )

        print(
            "  help"
        )

        print(
            "  quit"
        )

        print()

    def list_prompts(
        self,
        intent_filter: str | None = None,
    ) -> None:

        entries = sorted(
            self.catalog.values(),
            key=lambda item: (
                item.intent,
                item.example_id,
            ),
        )

        if intent_filter is not None:

            entries = [
                entry

                for entry
                in entries

                if entry.intent
                ==
                intent_filter
            ]

        if not entries:

            print(
                "No matching prompt examples."
            )

            return

        print()

        for entry in entries:

            print(
                f"{entry.example_id:22} "
                f"intent={entry.intent:10} "
                f"tokens="
                f"{' '.join(entry.prompt_tokens)}"
            )

        print()

    def show_prompt(
        self,
        example_id: str,
    ) -> None:

        entry = self.catalog.get(
            example_id
        )

        if entry is None:

            print(
                f"Unknown example ID: "
                f"{example_id}"
            )

            return

        print()

        print(
            f"Example ID: "
            f"{entry.example_id}"
        )

        print(
            f"Intent:     "
            f"{entry.intent}"
        )

        print(
            "Prompt:"
        )

        print(
            "  "
            +
            " ".join(
                entry.prompt_tokens
            )
        )

        print()

    # ========================================================
    # Frozen MaleCNS simulation
    # ========================================================

    def warmup(
        self,
    ) -> None:

        for _ in range(
            WARMUP_STEPS
        ):

            fired = self.brain.step()

            self.trace.observe(
                fired
            )

    def stimulate_prompt(
        self,
        prompt_tokens: list[str],
    ):

        states = []

        for position, token in enumerate(
            prompt_tokens
        ):

            try:

                neurons = (
                    self.dual_map.prompt_neurons_for(
                        token,
                        position,
                    )
                )

            except Exception as exc:

                raise ValueError(
                    f"Unsupported prompt token "
                    f"{token!r} at position "
                    f"{position}."
                ) from exc

            for _ in range(
                PROMPT_TOKEN_STEPS
            ):

                fired = self.brain.step(
                    inject=[
                        (
                            neurons,
                            AMPLITUDE,
                        )
                    ]
                )

                states.append(
                    np.asarray(
                        self.trace.observe(
                            fired
                        )
                    ).copy()
                )

            for _ in range(
                PROMPT_GAP_STEPS
            ):

                fired = self.brain.step()

                states.append(
                    np.asarray(
                        self.trace.observe(
                            fired
                        )
                    ).copy()
                )

        if not states:

            raise ValueError(
                "Prompt cannot be empty."
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

    def stimulate_feedback(
        self,
        token: str,
    ):

        neurons = (
            self.dual_map.feedback_neurons_for(
                token
            )
        )

        states = []

        for _ in range(
            FEEDBACK_TOKEN_STEPS
        ):

            fired = self.brain.step(
                inject=[
                    (
                        neurons,
                        AMPLITUDE,
                    )
                ]
            )

            states.append(
                np.asarray(
                    self.trace.observe(
                        fired
                    )
                ).copy()
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

            fired = self.brain.step()

            delayed_state = np.asarray(
                self.trace.observe(
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
    # Frozen adapter
    # ========================================================

    @torch.no_grad()
    def predict_token(
        self,
        feature: np.ndarray,
        position: int,
    ):

        standardized = (
            (
                feature
                -
                self.feature_mean
            )
            /
            self.feature_std
        ).astype(
            np.float32,
            copy=False,
        )

        X = torch.from_numpy(
            standardized
        ).reshape(
            1,
            -1,
        ).to(
            self.device
        )

        position_tensor = torch.tensor(
            [
                position
            ],
            dtype=torch.long,
            device=self.device,
        )

        logits = self.model(
            X,
            position_tensor,
        )[
            0
        ]

        probabilities = torch.softmax(
            logits,
            dim=0,
        )

        predicted_id = int(
            torch.argmax(
                probabilities
            ).item()
        )

        top_k = min(
            3,
            len(
                self.token_classes
            ),
        )

        top_probabilities, top_ids = (
            torch.topk(
                probabilities,
                k=top_k,
            )
        )

        top_predictions = [
            (
                self.token_classes[
                    int(
                        token_id
                    )
                ],
                float(
                    probability
                ),
            )

            for probability, token_id
            in zip(
                top_probabilities.cpu().tolist(),
                top_ids.cpu().tolist(),
            )
        ]

        return (
            self.token_classes[
                predicted_id
            ],
            top_predictions,
        )

    # ========================================================
    # True closed-loop generation
    # ========================================================

    def generate(
        self,
        prompt_tokens: list[str],
    ) -> None:

        self.brain.reset(
            seed=self.neural_seed
        )

        self.trace.reset()

        self.warmup()

        prompt_summary = (
            self.stimulate_prompt(
                prompt_tokens
            )
        )

        feedback_token = "BOS"

        generated_tokens = []

        print()

        print(
            "Generating closed-loop..."
        )

        print()

        for position in range(
            self.max_generation_steps
        ):

            feedback_summary = (
                self.stimulate_feedback(
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
                feature.shape
                !=
                (
                    EXPECTED_FEATURE_DIM,
                )
            ):

                raise RuntimeError(
                    f"Unexpected feature shape: "
                    f"{feature.shape}"
                )

            (
                predicted_token,
                top_predictions,
            ) = self.predict_token(
                feature,
                position,
            )

            generated_tokens.append(
                predicted_token
            )

            top3_text = " | ".join(
                f"{token}="
                f"{probability * 100:.1f}%"

                for token, probability
                in top_predictions
            )

            print(
                f"P{position}: "
                f"{predicted_token:10} "
                f"[{top3_text}]"
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

        print()

        print(
            "Generated tokens:"
        )

        print(
            "  "
            +
            " ".join(
                generated_tokens
            )
        )

        print()

        try:

            code = tokens_to_code(
                generated_tokens
            )

            print(
                "Generated Python:"
            )

            print()

            print(
                code
            )

        except Exception as exc:

            print(
                "Could not convert the generated "
                "token sequence into Python code."
            )

            print(
                f"Reason: {exc}"
            )

        print()

    # ========================================================
    # Command loop
    # ========================================================

    def run(
        self,
    ) -> None:

        while True:

            try:

                raw = input(
                    f"PyFly(seed={self.neural_seed})> "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):

                print()

                print(
                    "Exiting PyFly."
                )

                break

            if not raw:

                continue

            parts = raw.split()

            command = parts[
                0
            ].lower()

            args = parts[
                1:
            ]

            try:

                if command in {
                    "quit",
                    "exit",
                    "q",
                }:

                    print(
                        "Exiting PyFly."
                    )

                    break

                elif command == "help":

                    self.print_help()

                elif command == "classes":

                    print()

                    print(
                        " ".join(
                            self.token_classes
                        )
                    )

                    print()

                elif command == "list":

                    intent_filter = (
                        args[
                            0
                        ]
                        if args
                        else None
                    )

                    self.list_prompts(
                        intent_filter
                    )

                elif command == "show":

                    if not args:

                        print(
                            "Usage: "
                            "show <example_id>"
                        )

                        continue

                    self.show_prompt(
                        args[
                            0
                        ]
                    )

                elif command == "run":

                    if not args:

                        print(
                            "Usage: "
                            "run <example_id>"
                        )

                        continue

                    entry = self.catalog.get(
                        args[
                            0
                        ]
                    )

                    if entry is None:

                        print(
                            f"Unknown example ID: "
                            f"{args[0]}"
                        )

                        continue

                    print()

                    print(
                        f"Intent: "
                        f"{entry.intent}"
                    )

                    print(
                        "Prompt tokens:"
                    )

                    print(
                        "  "
                        +
                        " ".join(
                            entry.prompt_tokens
                        )
                    )

                    self.generate(
                        list(
                            entry.prompt_tokens
                        )
                    )

                elif command == "tokens":

                    if not args:

                        print(
                            "Usage: tokens "
                            "TOKEN_1 TOKEN_2 ..."
                        )

                        continue

                    self.generate(
                        [
                            str(
                                token
                            )

                            for token
                            in args
                        ]
                    )

                elif command == "seed":

                    if len(
                        args
                    ) != 1:

                        print(
                            "Usage: seed <integer>"
                        )

                        continue

                    self.neural_seed = int(
                        args[
                            0
                        ]
                    )

                    print(
                        f"Neural seed set to "
                        f"{self.neural_seed}."
                    )

                else:

                    print(
                        "Unknown command. "
                        "Type 'help'."
                    )

            except Exception as exc:

                print()

                print(
                    "PyFly error:"
                )

                print(
                    f"  {type(exc).__name__}: "
                    f"{exc}"
                )

                print()


# ============================================================
# Main
# ============================================================

def main() -> None:

    app = PyFlyConsole()

    app.run()


if __name__ == "__main__":
    main()
