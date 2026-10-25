from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QPainter,
    QPainterPath,
    QPen,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from flybrain import FlyBrain, Trace

from pyfly.dataset import load_dataset
from pyfly.dual_channel_map import DualChannelMap
from pyfly.generator_dataset import load_generator_jsonl
from pyfly.tokenizer import tokens_to_code


# ============================================================
# PyFly Studio V2
#
# A friendly PySide6 desktop application for the frozen
# Phase-3 PyFly system.
#
# Frozen inference path:
#
#   prompt tokens
#       ↓
#   real MaleCNS simulation
#       ↓
#   frozen Step-80 PyTorch adapter
#       ↓
#   predicted Python token
#       ↓
#   token fed back into MaleCNS
#       ↓
#   next token
#
# V2 additions:
#   * animated fly / neural activity display
#   * polished Generate / History / System tabs
#   * live progress bar and token chips
#   * cancel button
#   * copy code button
#   * save generated Python to .py
#   * random example button
#   * persistent local generation history
#   * dark / light themes
#   * final benchmark summary when available
#
# NO training.
# NO weight changes.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_NAME = "cuda"

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

FINAL_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "phase3_final_holdout_metrics.json"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "ui"
    / "pyfly_studio_v2_history.json"
)

PROCESSED_PROMPT_FILES = [
    PROJECT_ROOT / "data" / "processed" / "generator_train.jsonl",
    PROJECT_ROOT / "data" / "processed" / "generator_validation.jsonl",
    PROJECT_ROOT / "data" / "processed" / "generator_test.jsonl",
]

RAW_PROMPT_FILES = [
    PROJECT_ROOT / "data" / "raw" / "phase3_fresh_holdout.jsonl",
]


# ============================================================
# Frozen stimulation settings
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

DEFAULT_NEURAL_SEED = 3


# ============================================================
# Model
# ============================================================

class PyFlyNeuralAdapter(nn.Module):
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

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.neural_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(
            max_position + 1,
            position_embed_dim,
        )

        merged_dim = hidden_dim + position_embed_dim

        self.adapter = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(merged_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(second_hidden_dim),
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
        X = self.input_norm(X)
        X = self.input_dropout(X)

        neural_hidden = self.neural_projection(X)
        position_hidden = self.position_embedding(positions)

        merged = torch.cat(
            [neural_hidden, position_hidden],
            dim=1,
        )

        hidden = self.adapter(merged)
        return self.classifier(hidden)


# ============================================================
# Prompt catalog
# ============================================================

@dataclass(frozen=True)
class PromptEntry:
    example_id: str
    intent: str
    prompt_tokens: tuple[str, ...]
    source_name: str


def load_prompt_catalog() -> dict[str, PromptEntry]:
    catalog: dict[str, PromptEntry] = {}

    # Processed teacher-forcing rows.
    for path in PROCESSED_PROMPT_FILES:
        if not path.exists():
            continue

        samples = load_generator_jsonl(path)
        grouped: dict[str, list] = {}

        for sample in samples:
            grouped.setdefault(
                str(sample.example_id),
                [],
            ).append(sample)

        for example_id, rows in grouped.items():
            rows.sort(
                key=lambda row: row.target_position
            )

            first = rows[0]

            catalog[example_id] = PromptEntry(
                example_id=example_id,
                intent=str(first.intent),
                prompt_tokens=tuple(
                    str(token)
                    for token in first.prompt_tokens
                ),
                source_name=path.name,
            )

    # Raw logical prompts, such as the fresh Phase-3 holdout.
    for path in RAW_PROMPT_FILES:
        if not path.exists():
            continue

        try:
            examples = load_dataset(path)
        except Exception:
            # Catalog display should never prevent the app
            # from starting if an optional raw file is absent
            # or has an older format.
            continue

        for example in examples:
            catalog[str(example.id)] = PromptEntry(
                example_id=str(example.id),
                intent=str(example.intent),
                prompt_tokens=tuple(
                    str(token)
                    for token in example.prompt_tokens
                ),
                source_name=path.name,
            )

    return catalog


# ============================================================
# Dual channel map
# ============================================================

def load_dual_map(
    path: Path,
) -> DualChannelMap:
    data = np.load(
        path,
        allow_pickle=False,
    )

    prompt_tokens = data[
        "prompt_tokens"
    ].astype(str)

    prompt_positions = data[
        "prompt_positions"
    ].astype(np.int64)

    prompt_neurons = data[
        "prompt_neurons"
    ].astype(np.int64)

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
                str(token),
                int(position),
            )
        ] = prompt_neurons[
            index
        ].copy()

    feedback_tokens = data[
        "feedback_tokens"
    ].astype(str)

    feedback_neurons = data[
        "feedback_neurons"
    ].astype(np.int64)

    feedback_mapping = {}

    for index, token in enumerate(
        feedback_tokens
    ):
        feedback_mapping[
            str(token)
        ] = feedback_neurons[
            index
        ].copy()

    return DualChannelMap(
        prompt_mapping=prompt_mapping,
        feedback_mapping=feedback_mapping,
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
# Backend
# ============================================================

class GenerationCancelled(Exception):
    pass


class PyFlyEngine:
    def __init__(self) -> None:
        for path in [
            MODEL_FILE,
            SCALER_FILE,
            MAPPING_FILE,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file missing: {path}"
                )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available to PyTorch."
            )

        self.device = torch.device(
            DEVICE_NAME
        )

        self.catalog = load_prompt_catalog()
        self.dual_map = load_dual_map(
            MAPPING_FILE
        )

        checkpoint = torch.load(
            MODEL_FILE,
            map_location=self.device,
            weights_only=False,
        )

        self.token_classes = [
            str(token)
            for token in checkpoint[
                "token_classes"
            ]
        ]

        self.max_position = int(
            checkpoint[
                "max_position"
            ]
        )

        self.max_generation_steps = (
            self.max_position + 1
        )

        self.best_epoch = checkpoint.get(
            "best_epoch"
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
            max_position=self.max_position,
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

        scaler = np.load(
            SCALER_FILE,
            allow_pickle=False,
        )

        self.feature_mean = scaler[
            "mean"
        ].astype(
            np.float32
        )

        self.feature_std = scaler[
            "std"
        ].astype(
            np.float32
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
                "Frozen feedback mapping is missing "
                f"tokens: {sorted(missing_feedback)}"
            )

        self.brain = FlyBrain(
            device="cuda",
            seed=0,
        )

        if self.brain.n != EXPECTED_NEURONS:
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

    def supported_intents(
        self,
    ) -> list[str]:
        return sorted(
            {
                entry.intent
                for entry in self.catalog.values()
            }
        )

    def filtered_examples(
        self,
        intent: str | None = None,
    ) -> list[PromptEntry]:
        entries = sorted(
            self.catalog.values(),
            key=lambda entry: (
                entry.intent,
                entry.example_id,
            ),
        )

        if (
            intent
            and
            intent
            !=
            "ALL"
        ):
            entries = [
                entry
                for entry in entries
                if entry.intent == intent
            ]

        return entries

    def _check_cancel(
        self,
        should_cancel,
    ) -> None:
        if (
            should_cancel is not None
            and
            should_cancel()
        ):
            raise GenerationCancelled(
                "Generation cancelled by user."
            )

    def warmup(
        self,
        should_cancel=None,
    ) -> None:
        for _ in range(
            WARMUP_STEPS
        ):
            self._check_cancel(
                should_cancel
            )

            fired = self.brain.step()

            self.trace.observe(
                fired
            )

    def stimulate_prompt(
        self,
        prompt_tokens: list[str],
        should_cancel=None,
    ):
        states = []

        for position, token in enumerate(
            prompt_tokens
        ):
            self._check_cancel(
                should_cancel
            )

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
                    f"{token!r} at position {position}."
                ) from exc

            for _ in range(
                PROMPT_TOKEN_STEPS
            ):
                self._check_cancel(
                    should_cancel
                )

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
                self._check_cancel(
                    should_cancel
                )

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
        should_cancel=None,
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
            self._check_cancel(
                should_cancel
            )

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
            self._check_cancel(
                should_cancel
            )

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
            {
                "token": self.token_classes[
                    int(
                        token_id
                    )
                ],
                "probability": float(
                    probability
                ),
            }
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

    def generate(
        self,
        prompt_tokens: list[str],
        neural_seed: int,
        progress_callback=None,
        status_callback=None,
        should_cancel=None,
    ) -> dict:
        self._check_cancel(
            should_cancel
        )

        self.brain.reset(
            seed=int(
                neural_seed
            )
        )

        self.trace.reset()

        if status_callback is not None:
            status_callback(
                "Warming up MaleCNS..."
            )

        self.warmup(
            should_cancel
        )

        if status_callback is not None:
            status_callback(
                "Stimulating prompt channels..."
            )

        prompt_summary = (
            self.stimulate_prompt(
                prompt_tokens,
                should_cancel,
            )
        )

        feedback_token = "BOS"

        generated_tokens = []
        steps = []

        if status_callback is not None:
            status_callback(
                "Generating Python tokens closed-loop..."
            )

        for position in range(
            self.max_generation_steps
        ):
            self._check_cancel(
                should_cancel
            )

            feedback_summary = (
                self.stimulate_feedback(
                    feedback_token,
                    should_cancel,
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
                    "Unexpected live feature shape: "
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

            step_info = {
                "position": int(
                    position
                ),
                "feedback_token": (
                    feedback_token
                ),
                "prediction": (
                    predicted_token
                ),
                "confidence": float(
                    top_predictions[
                        0
                    ][
                        "probability"
                    ]
                ),
                "top3": (
                    top_predictions
                ),
            }

            steps.append(
                step_info
            )

            if progress_callback is not None:
                progress_callback(
                    step_info
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

        try:
            code_text = tokens_to_code(
                generated_tokens
            )

            code_error = None

        except Exception as exc:
            code_text = None
            code_error = str(
                exc
            )

        return {
            "prompt_tokens": list(
                prompt_tokens
            ),
            "neural_seed": int(
                neural_seed
            ),
            "generated_tokens": (
                generated_tokens
            ),
            "steps": (
                steps
            ),
            "code": (
                code_text
            ),
            "code_error": (
                code_error
            ),
        }


# ============================================================
# Worker
# ============================================================

class GenerationWorker(QObject):
    progress = Signal(dict)
    status = Signal(str)
    finished = Signal(dict)
    cancelled = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        engine: PyFlyEngine,
        prompt_tokens: list[str],
        neural_seed: int,
    ) -> None:
        super().__init__()

        self.engine = engine
        self.prompt_tokens = (
            prompt_tokens
        )
        self.neural_seed = (
            neural_seed
        )
        self.cancel_requested = False

    def request_cancel(
        self,
    ) -> None:
        self.cancel_requested = True

    def run(
        self,
    ) -> None:
        try:
            result = self.engine.generate(
                prompt_tokens=(
                    self.prompt_tokens
                ),
                neural_seed=(
                    self.neural_seed
                ),
                progress_callback=(
                    self.progress.emit
                ),
                status_callback=(
                    self.status.emit
                ),
                should_cancel=lambda:
                    self.cancel_requested,
            )

            self.finished.emit(
                result
            )

        except GenerationCancelled as exc:
            self.cancelled.emit(
                str(
                    exc
                )
            )

        except Exception as exc:
            self.error.emit(
                f"{type(exc).__name__}: "
                f"{exc}"
            )


# ============================================================
# Animated fly / neural display
# ============================================================

class FlyNeuralWidget(QWidget):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.phase = 0.0
        self.active = False
        self.current_token = ""
        self.activity_level = 0.0
        self.light_theme = False

        self.setMinimumHeight(
            165
        )

        self.setMaximumHeight(
            205
        )

        self.timer = QTimer(
            self
        )

        self.timer.timeout.connect(
            self.tick
        )

        self.timer.start(
            45
        )

    def set_active(
        self,
        active: bool,
    ) -> None:
        self.active = active

        if not active:
            self.activity_level = 0.0

        self.update()

    def set_token_activity(
        self,
        token: str,
        confidence: float,
    ) -> None:
        self.current_token = token
        self.activity_level = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        self.update()

    def set_light_theme(
        self,
        light: bool,
    ) -> None:
        self.light_theme = light
        self.update()

    def tick(
        self,
    ) -> None:
        self.phase += (
            0.32
            if self.active
            else
            0.08
        )

        if (
            not self.active
            and
            self.activity_level
            >
            0
        ):
            self.activity_level *= 0.96

        self.update()

    def paintEvent(
        self,
        event,
    ) -> None:
        _ = event

        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.Antialiasing
        )

        if self.light_theme:
            background = QColor(
                "#eef4ff"
            )
            body_color = QColor(
                "#475569"
            )
            body_highlight = QColor(
                "#94a3b8"
            )
            text_color = QColor(
                "#0f172a"
            )
            leg_color = QColor(
                "#64748b"
            )
        else:
            background = QColor(
                "#08111f"
            )
            body_color = QColor(
                "#475569"
            )
            body_highlight = QColor(
                "#cbd5e1"
            )
            text_color = QColor(
                "#f8fafc"
            )
            leg_color = QColor(
                "#94a3b8"
            )

        painter.fillRect(
            self.rect(),
            background,
        )

        w = self.width()
        h = self.height()

        cx = w / 2
        cy = h / 2 + 12

        wing_wave = (
            12
            +
            13
            *
            np.sin(
                self.phase
            )
        )

        pulse = (
            22
            +
            18
            *
            (
                0.5
                +
                0.5
                *
                np.sin(
                    self.phase
                    *
                    0.7
                )
            )
        )

        # Neural halo.
        halo_alpha = int(
            35
            +
            130
            *
            (
                self.activity_level
                if self.active
                else 0.15
            )
        )

        painter.setPen(
            Qt.NoPen
        )

        painter.setBrush(
            QColor(
                37,
                99,
                235,
                halo_alpha,
            )
        )

        painter.drawEllipse(
            int(
                cx
                -
                92
                -
                pulse
                /
                3
            ),
            int(
                cy
                -
                92
                -
                pulse
                /
                3
            ),
            int(
                184
                +
                2
                *
                pulse
                /
                3
            ),
            int(
                184
                +
                2
                *
                pulse
                /
                3
            ),
        )

        # Orbiting neural nodes.
        for index in range(
            14
        ):
            angle = (
                self.phase
                *
                0.35
                +
                index
                *
                (
                    2
                    *
                    np.pi
                    /
                    14
                )
            )

            radius = (
                112
                +
                8
                *
                np.sin(
                    self.phase
                    +
                    index
                )
            )

            x = (
                cx
                +
                np.cos(
                    angle
                )
                *
                radius
            )

            y = (
                cy
                +
                np.sin(
                    angle
                )
                *
                radius
                *
                0.63
            )

            alpha = int(
                90
                +
                120
                *
                (
                    self.activity_level
                    if self.active
                    else 0.2
                )
            )

            painter.setBrush(
                QColor(
                    34,
                    197,
                    94,
                    alpha,
                )
            )

            painter.drawEllipse(
                int(
                    x - 3
                ),
                int(
                    y - 3
                ),
                6,
                6,
            )

        # Wings.
        painter.setBrush(
            QColor(
                125,
                211,
                252,
                80
                if self.light_theme
                else 95,
            )
        )

        left_wing = QPainterPath()

        left_wing.moveTo(
            cx - 17,
            cy - 25,
        )

        left_wing.quadTo(
            cx - 118,
            cy - 110 - wing_wave,
            cx - 28,
            cy - 6,
        )

        left_wing.quadTo(
            cx - 95,
            cy - 38,
            cx - 17,
            cy - 25,
        )

        painter.drawPath(
            left_wing
        )

        right_wing = QPainterPath()

        right_wing.moveTo(
            cx + 17,
            cy - 25,
        )

        right_wing.quadTo(
            cx + 118,
            cy - 110 - wing_wave,
            cx + 28,
            cy - 6,
        )

        right_wing.quadTo(
            cx + 95,
            cy - 38,
            cx + 17,
            cy - 25,
        )

        painter.drawPath(
            right_wing
        )

        # Legs.
        painter.setPen(
            QPen(
                leg_color,
                3,
            )
        )

        for dx, dy1, dy2, side in [
            (-20, 0, 42, -1),
            (-10, 13, 54, -1),
            (10, 13, 54, 1),
            (20, 0, 42, 1),
        ]:
            painter.drawLine(
                int(
                    cx + dx
                ),
                int(
                    cy + dy1
                ),
                int(
                    cx
                    +
                    dx
                    +
                    side
                    *
                    38
                ),
                int(
                    cy + dy2
                ),
            )

        # Body.
        painter.setPen(
            Qt.NoPen
        )

        painter.setBrush(
            body_color
        )

        painter.drawEllipse(
            int(
                cx - 21
            ),
            int(
                cy - 49
            ),
            42,
            48,
        )

        painter.drawEllipse(
            int(
                cx - 24
            ),
            int(
                cy - 15
            ),
            48,
            70,
        )

        painter.drawEllipse(
            int(
                cx - 18
            ),
            int(
                cy + 35
            ),
            36,
            68,
        )

        painter.setBrush(
            body_highlight
        )

        painter.drawEllipse(
            int(
                cx - 7
            ),
            int(
                cy - 6
            ),
            14,
            30,
        )

        painter.drawEllipse(
            int(
                cx - 5
            ),
            int(
                cy + 54
            ),
            10,
            20,
        )

        # Eyes.
        painter.setBrush(
            QColor(
                "#ef4444"
            )
        )

        painter.drawEllipse(
            int(
                cx - 22
            ),
            int(
                cy - 38
            ),
            12,
            15,
        )

        painter.drawEllipse(
            int(
                cx + 10
            ),
            int(
                cy - 38
            ),
            12,
            15,
        )

        # Antennae.
        painter.setPen(
            QPen(
                body_highlight,
                3,
            )
        )

        painter.drawLine(
            int(
                cx - 8
            ),
            int(
                cy - 45
            ),
            int(
                cx - 28
            ),
            int(
                cy - 69
            ),
        )

        painter.drawLine(
            int(
                cx + 8
            ),
            int(
                cy - 45
            ),
            int(
                cx + 28
            ),
            int(
                cy - 69
            ),
        )

        # Header.
        painter.setPen(
            text_color
        )

        title_font = QFont()
        title_font.setPointSize(
            15
        )
        title_font.setBold(
            True
        )

        painter.setFont(
            title_font
        )

        painter.drawText(
            0,
            12,
            w,
            28,
            Qt.AlignCenter,
            "PyFly Neural Core",
        )

        sub_font = QFont()
        sub_font.setPointSize(
            9
        )

        painter.setFont(
            sub_font
        )

        token_text = (
            f"Thinking: {self.current_token}"
            if self.active
            and
            self.current_token
            else
            (
                "MaleCNS active"
                if self.active
                else
                "Ready"
            )
        )

        painter.drawText(
            0,
            h - 34,
            w,
            22,
            Qt.AlignCenter,
            token_text,
        )


# ============================================================
# Main window
# ============================================================

class MainWindow(QMainWindow):
    def __init__(
        self,
        engine: PyFlyEngine,
    ) -> None:
        super().__init__()

        self.engine = engine

        self.worker_thread: (
            QThread
            |
            None
        ) = None

        self.worker: (
            GenerationWorker
            |
            None
        ) = None

        self.current_result: (
            dict
            |
            None
        ) = None

        self.history = (
            self.load_history()
        )

        self.current_theme = "dark"

        self.setWindowTitle(
            "PyFly Studio V2"
        )

        # V2.2 target: 16-inch, 2560x1440, 16:9 laptop.
        # Qt uses logical pixels after Windows scaling, so on a
        # typical 125-150% scale this lands around 1600-1750 x
        # 880-980 usable pixels. We size from the actual screen
        # instead of assuming raw 2560x1440 coordinates.
        screen = QGuiApplication.primaryScreen()

        if screen is not None:
            available = screen.availableGeometry()

            target_width = min(
                1760,
                max(
                    1120,
                    int(
                        available.width()
                        *
                        0.94
                    ),
                ),
            )

            target_height = min(
                980,
                max(
                    720,
                    int(
                        available.height()
                        *
                        0.94
                    ),
                ),
            )

            self.resize(
                target_width,
                target_height,
            )
        else:
            self.resize(
                1600,
                900,
            )

        self.setMinimumSize(
            960,
            640,
        )

        self.build_ui()
        self.populate_intents()
        self.refresh_examples()
        self.refresh_history()
        self.populate_system_info()
        self.apply_theme(
            self.current_theme
        )

        self.status_label.setText(
            "Ready — frozen Phase-3 model loaded."
        )

        QTimer.singleShot(
            0,
            self.update_responsive_layout,
        )

    # ========================================================
    # Responsive layout
    # ========================================================

    def resizeEvent(
        self,
        event,
    ) -> None:
        super().resizeEvent(
            event
        )

        # Defer until Qt finishes the resize pass.
        QTimer.singleShot(
            0,
            self.update_responsive_layout,
        )

    def update_responsive_layout(
        self,
    ) -> None:
        if not hasattr(
            self,
            "generate_splitter",
        ):
            return

        width = self.width()
        height = self.height()

        # ----------------------------------------------------
        # TIER 1 — 1440p laptop / large desktop
        # Typical logical width after Windows scaling:
        # roughly 1450-1750 px.
        # ----------------------------------------------------
        if width >= 1380:
            self.generate_splitter.setOrientation(
                Qt.Horizontal
            )

            self.left_scroll.setMinimumWidth(
                370
            )

            self.left_scroll.setMaximumWidth(
                430
            )

            self.left_scroll.setMinimumHeight(
                0
            )

            self.left_scroll.setMaximumHeight(
                16_777_215
            )

            left_width = min(
                420,
                max(
                    385,
                    int(
                        width
                        *
                        0.255
                    ),
                ),
            )

            self.generate_splitter.setSizes(
                [
                    left_width,
                    max(
                        850,
                        width
                        -
                        left_width
                        -
                        36,
                    ),
                ]
            )

            self.subtitle_label.setVisible(
                True
            )

            self.gpu_badge.setVisible(
                True
            )

            self.theme_label.setVisible(
                True
            )

        # ----------------------------------------------------
        # TIER 2 — normal laptop / scaled window
        # Still horizontal, but a slimmer control rail.
        # ----------------------------------------------------
        elif width >= 1080:
            self.generate_splitter.setOrientation(
                Qt.Horizontal
            )

            self.left_scroll.setMinimumWidth(
                330
            )

            self.left_scroll.setMaximumWidth(
                375
            )

            self.left_scroll.setMinimumHeight(
                0
            )

            self.left_scroll.setMaximumHeight(
                16_777_215
            )

            self.generate_splitter.setSizes(
                [
                    350,
                    max(
                        680,
                        width
                        -
                        390,
                    ),
                ]
            )

            self.subtitle_label.setVisible(
                True
            )

            self.gpu_badge.setVisible(
                False
            )

            self.theme_label.setVisible(
                True
            )

        # ----------------------------------------------------
        # TIER 3 — snapped / narrow window
        # Stack controls over results so nothing is cut off.
        # ----------------------------------------------------
        else:
            self.generate_splitter.setOrientation(
                Qt.Vertical
            )

            self.left_scroll.setMinimumWidth(
                0
            )

            self.left_scroll.setMaximumWidth(
                16_777_215
            )

            control_height = min(
                315,
                max(
                    235,
                    int(
                        height
                        *
                        0.34
                    ),
                ),
            )

            self.left_scroll.setMinimumHeight(
                225
            )

            self.left_scroll.setMaximumHeight(
                330
            )

            self.generate_splitter.setSizes(
                [
                    control_height,
                    max(
                        360,
                        height
                        -
                        control_height
                        -
                        95,
                    ),
                ]
            )

            self.subtitle_label.setVisible(
                False
            )

            self.gpu_badge.setVisible(
                False
            )

            self.theme_label.setVisible(
                False
            )

    # ========================================================
    # History
    # ========================================================

    def load_history(
        self,
    ) -> list[dict]:
        if not HISTORY_FILE.exists():
            return []

        try:
            payload = json.loads(
                HISTORY_FILE.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                payload,
                list,
            ):
                return payload[
                    -100:
                ]

        except Exception:
            pass

        return []

    def save_history(
        self,
    ) -> None:
        HISTORY_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        HISTORY_FILE.write_text(
            json.dumps(
                self.history[
                    -100:
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # UI helpers
    # ========================================================

    def make_card(
        self,
        title: str,
        value: str,
    ) -> QFrame:
        card = QFrame()

        card.setObjectName(
            "metricCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            10,
            7,
            10,
            7,
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "metricTitle"
        )

        value_label = QLabel(
            value
        )

        value_label.setObjectName(
            "metricValue"
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            value_label
        )

        return card

    def build_ui(
        self,
    ) -> None:
        central = QWidget()
        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )

        root.setContentsMargins(
            12,
            12,
            12,
            12,
        )

        root.setSpacing(
            10
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = QFrame()
        header.setObjectName(
            "header"
        )

        header_layout = QHBoxLayout(
            header
        )

        header_layout.setContentsMargins(
            16,
            9,
            16,
            9,
        )

        title_box = QVBoxLayout()

        title = QLabel(
            "PyFly Studio V2"
        )

        title.setObjectName(
            "appTitle"
        )

        self.subtitle_label = QLabel(
            "MaleCNS-powered closed-loop Python generation"
        )

        self.subtitle_label.setObjectName(
            "appSubtitle"
        )

        title_box.addWidget(
            title
        )

        title_box.addWidget(
            self.subtitle_label
        )

        header_layout.addLayout(
            title_box
        )

        header_layout.addStretch(
            1
        )

        self.gpu_badge = QLabel(
            "● CUDA READY"
        )

        self.gpu_badge.setObjectName(
            "gpuBadge"
        )

        header_layout.addWidget(
            self.gpu_badge
        )

        self.theme_label = QLabel(
            "Theme"
        )

        header_layout.addWidget(
            self.theme_label
        )

        self.theme_combo = QComboBox()

        self.theme_combo.addItems(
            [
                "Dark",
                "Light",
            ]
        )

        self.theme_combo.currentTextChanged.connect(
            self.theme_changed
        )

        header_layout.addWidget(
            self.theme_combo
        )

        root.addWidget(
            header
        )

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(
            True
        )

        root.addWidget(
            self.tabs,
            1,
        )

        self.generate_tab = QWidget()
        self.history_tab = QWidget()
        self.system_tab = QWidget()

        self.tabs.addTab(
            self.generate_tab,
            "Generate"
        )

        self.tabs.addTab(
            self.history_tab,
            "History"
        )

        self.tabs.addTab(
            self.system_tab,
            "System"
        )

        self.build_generate_tab()
        self.build_history_tab()
        self.build_system_tab()

    def build_generate_tab(
        self,
    ) -> None:
        layout = QHBoxLayout(
            self.generate_tab
        )

        layout.setContentsMargins(
            0,
            8,
            0,
            0,
        )

        self.generate_splitter = QSplitter(
            Qt.Horizontal
        )

        self.generate_splitter.setChildrenCollapsible(
            False
        )

        layout.addWidget(
            self.generate_splitter
        )

        # ----------------------------------------------------
        # Left control panel
        # ----------------------------------------------------

        left = QWidget()

        left_layout = QVBoxLayout(
            left
        )

        left_layout.setSpacing(
            10
        )

        self.fly_widget = FlyNeuralWidget()

        left_layout.addWidget(
            self.fly_widget
        )

        stat_grid = QGridLayout()

        stat_grid.addWidget(
            self.make_card(
                "MaleCNS neurons",
                f"{EXPECTED_NEURONS:,}",
            ),
            0,
            0,
        )

        stat_grid.addWidget(
            self.make_card(
                "Readout neurons",
                f"{EXPECTED_READOUT_NEURONS:,}",
            ),
            0,
            1,
        )

        stat_grid.addWidget(
            self.make_card(
                "Model epoch",
                str(
                    self.engine.best_epoch
                ),
            ),
            1,
            0,
        )

        stat_grid.addWidget(
            self.make_card(
                "Token classes",
                str(
                    len(
                        self.engine.token_classes
                    )
                ),
            ),
            1,
            1,
        )

        left_layout.addLayout(
            stat_grid
        )

        prompt_box = QGroupBox(
            "Prompt"
        )

        prompt_layout = QVBoxLayout(
            prompt_box
        )

        mode_row = QHBoxLayout()

        self.example_radio = QRadioButton(
            "Example"
        )

        self.custom_radio = QRadioButton(
            "Custom tokens"
        )

        self.example_radio.setChecked(
            True
        )

        mode_row.addWidget(
            self.example_radio
        )

        mode_row.addWidget(
            self.custom_radio
        )

        mode_row.addStretch(
            1
        )

        prompt_layout.addLayout(
            mode_row
        )

        filter_row = QHBoxLayout()

        filter_row.addWidget(
            QLabel(
                "Intent"
            )
        )

        self.intent_combo = QComboBox()

        self.intent_combo.currentTextChanged.connect(
            self.refresh_examples
        )

        filter_row.addWidget(
            self.intent_combo,
            1,
        )

        prompt_layout.addLayout(
            filter_row
        )

        example_row = QHBoxLayout()

        self.example_combo = QComboBox()

        self.example_combo.currentIndexChanged.connect(
            self.example_changed
        )

        example_row.addWidget(
            self.example_combo,
            1,
        )

        self.random_button = QPushButton(
            "Surprise Me"
        )

        self.random_button.setObjectName(
            "secondaryButton"
        )

        self.random_button.clicked.connect(
            self.random_example
        )

        example_row.addWidget(
            self.random_button
        )

        prompt_layout.addLayout(
            example_row
        )

        self.example_preview = QPlainTextEdit()

        self.example_preview.setReadOnly(
            True
        )

        self.example_preview.setMaximumHeight(
            58
        )

        prompt_layout.addWidget(
            self.example_preview
        )

        custom_label = QLabel(
            "Custom prompt-token sequence"
        )

        prompt_layout.addWidget(
            custom_label
        )

        self.tokens_edit = QLineEdit()

        self.tokens_edit.setPlaceholderText(
            "Enter exact tokens separated by spaces"
        )

        prompt_layout.addWidget(
            self.tokens_edit
        )

        helper = QLabel(
            "Custom mode uses the project's existing token grammar; "
            "it does not yet understand arbitrary English sentences."
        )

        helper.setWordWrap(
            True
        )

        helper.setObjectName(
            "helperText"
        )

        prompt_layout.addWidget(
            helper
        )

        seed_row = QHBoxLayout()

        seed_row.addWidget(
            QLabel(
                "Neural seed"
            )
        )

        self.seed_spin = QSpinBox()

        self.seed_spin.setRange(
            0,
            999_999,
        )

        self.seed_spin.setValue(
            DEFAULT_NEURAL_SEED
        )

        seed_row.addWidget(
            self.seed_spin
        )

        seed_row.addStretch(
            1
        )

        prompt_layout.addLayout(
            seed_row
        )

        left_layout.addWidget(
            prompt_box
        )

        action_row = QHBoxLayout()

        self.generate_button = QPushButton(
            "Generate Python"
        )

        self.generate_button.setObjectName(
            "primaryButton"
        )

        self.generate_button.clicked.connect(
            self.start_generation
        )

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.cancel_button.setObjectName(
            "dangerButton"
        )

        self.cancel_button.setEnabled(
            False
        )

        self.cancel_button.clicked.connect(
            self.cancel_generation
        )

        action_row.addWidget(
            self.generate_button,
            2,
        )

        action_row.addWidget(
            self.cancel_button,
            1,
        )

        left_layout.addLayout(
            action_row
        )

        self.status_label = QLabel(
            "Ready"
        )

        self.status_label.setWordWrap(
            True
        )

        self.status_label.setObjectName(
            "statusLabel"
        )

        left_layout.addWidget(
            self.status_label
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            self.engine.max_generation_steps,
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setFormat(
            "Ready"
        )

        left_layout.addWidget(
            self.progress_bar
        )

        left_layout.addStretch(
            1
        )

        # ----------------------------------------------------
        # Right workspace
        # ----------------------------------------------------

        right = QWidget()

        right_layout = QVBoxLayout(
            right
        )

        right_layout.setSpacing(
            10
        )

        current_prompt_box = QGroupBox(
            "Current Prompt"
        )

        current_prompt_layout = QVBoxLayout(
            current_prompt_box
        )

        self.current_prompt_edit = QPlainTextEdit()

        self.current_prompt_edit.setReadOnly(
            True
        )

        self.current_prompt_edit.setMaximumHeight(
            54
        )

        current_prompt_layout.addWidget(
            self.current_prompt_edit
        )

        right_layout.addWidget(
            current_prompt_box
        )

        token_box = QGroupBox(
            "Generated Token Stream"
        )

        token_box_layout = QVBoxLayout(
            token_box
        )

        self.token_scroll = QScrollArea()

        self.token_scroll.setWidgetResizable(
            True
        )

        self.token_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        self.token_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.token_container = QWidget()

        self.token_layout = QHBoxLayout(
            self.token_container
        )

        self.token_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )

        self.token_layout.setSpacing(
            8
        )

        self.token_layout.addStretch(
            1
        )

        self.token_scroll.setWidget(
            self.token_container
        )

        token_box_layout.addWidget(
            self.token_scroll
        )

        right_layout.addWidget(
            token_box
        )

        steps_box = QGroupBox(
            "Live Neural Decisions"
        )

        steps_layout = QVBoxLayout(
            steps_box
        )

        self.steps_table = QTableWidget(
            0,
            5,
        )

        self.steps_table.setHorizontalHeaderLabels(
            [
                "Pos",
                "Feedback",
                "Prediction",
                "Confidence",
                "Alternatives",
            ]
        )

        self.steps_table.verticalHeader().setVisible(
            False
        )

        header = self.steps_table.horizontalHeader()

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            4,
            QHeaderView.Stretch,
        )

        self.steps_table.setAlternatingRowColors(
            True
        )

        steps_layout.addWidget(
            self.steps_table
        )

        right_layout.addWidget(
            steps_box,
            4,
        )

        code_box = QGroupBox(
            "Generated Python"
        )

        code_layout = QVBoxLayout(
            code_box
        )

        code_actions = QHBoxLayout()

        self.copy_button = QPushButton(
            "Copy Code"
        )

        self.copy_button.setObjectName(
            "secondaryButton"
        )

        self.copy_button.clicked.connect(
            self.copy_code
        )

        self.save_button = QPushButton(
            "Save .py"
        )

        self.save_button.setObjectName(
            "secondaryButton"
        )

        self.save_button.clicked.connect(
            self.save_code
        )

        self.clear_button = QPushButton(
            "Clear"
        )

        self.clear_button.setObjectName(
            "secondaryButton"
        )

        self.clear_button.clicked.connect(
            self.clear_output
        )

        code_actions.addWidget(
            self.copy_button
        )

        code_actions.addWidget(
            self.save_button
        )

        code_actions.addStretch(
            1
        )

        code_actions.addWidget(
            self.clear_button
        )

        code_layout.addLayout(
            code_actions
        )

        self.code_output = QPlainTextEdit()

        self.code_output.setReadOnly(
            True
        )

        code_font = QFont(
            "Consolas"
        )

        code_font.setPointSize(
            12
        )

        self.code_output.setFont(
            code_font
        )

        code_layout.addWidget(
            self.code_output
        )

        right_layout.addWidget(
            code_box,
            3,
        )

        log_box = QGroupBox(
            "Activity"
        )

        log_layout = QVBoxLayout(
            log_box
        )

        self.log_output = QPlainTextEdit()

        self.log_output.setReadOnly(
            True
        )

        self.log_output.setMaximumHeight(
            105
        )

        log_layout.addWidget(
            self.log_output
        )

        right_layout.addWidget(
            log_box
        )

        # ----------------------------------------------------
        # Responsive wrappers
        #
        # The control column is scrollable so it can remain narrow
        # without forcing the whole app wider than the screen.
        # ----------------------------------------------------

        self.left_scroll = QScrollArea()

        self.left_scroll.setWidgetResizable(
            True
        )

        self.left_scroll.setFrameShape(
            QFrame.NoFrame
        )

        self.left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.left_scroll.setWidget(
            left
        )

        self.left_scroll.setMinimumWidth(
            360
        )

        self.left_scroll.setMaximumWidth(
            430
        )

        left.setMinimumWidth(
            0
        )

        right.setMinimumWidth(
            620
        )

        self.generate_splitter.addWidget(
            self.left_scroll
        )

        self.generate_splitter.addWidget(
            right
        )

        self.generate_splitter.setStretchFactor(
            0,
            0,
        )

        self.generate_splitter.setStretchFactor(
            1,
            1,
        )

        self.generate_splitter.setSizes(
            [
                395,
                1080,
            ]
        )

    def build_history_tab(
        self,
    ) -> None:
        layout = QHBoxLayout(
            self.history_tab
        )

        layout.setContentsMargins(
            0,
            8,
            0,
            0,
        )

        left_box = QGroupBox(
            "Generation History"
        )

        left_layout = QVBoxLayout(
            left_box
        )

        self.history_list = QListWidget()

        self.history_list.currentRowChanged.connect(
            self.history_selected
        )

        left_layout.addWidget(
            self.history_list
        )

        clear_history_button = QPushButton(
            "Clear History"
        )

        clear_history_button.setObjectName(
            "dangerButton"
        )

        clear_history_button.clicked.connect(
            self.clear_history
        )

        left_layout.addWidget(
            clear_history_button
        )

        right_box = QGroupBox(
            "Selected Result"
        )

        right_layout = QVBoxLayout(
            right_box
        )

        self.history_detail = QPlainTextEdit()

        self.history_detail.setReadOnly(
            True
        )

        history_font = QFont(
            "Consolas"
        )

        history_font.setPointSize(
            11
        )

        self.history_detail.setFont(
            history_font
        )

        right_layout.addWidget(
            self.history_detail
        )

        reuse_button = QPushButton(
            "Reuse This Prompt"
        )

        reuse_button.setObjectName(
            "primaryButton"
        )

        reuse_button.clicked.connect(
            self.reuse_history_prompt
        )

        right_layout.addWidget(
            reuse_button
        )

        layout.addWidget(
            left_box,
            1,
        )

        layout.addWidget(
            right_box,
            2,
        )

    def build_system_tab(
        self,
    ) -> None:
        layout = QVBoxLayout(
            self.system_tab
        )

        layout.setContentsMargins(
            0,
            8,
            0,
            0,
        )

        metrics_box = QGroupBox(
            "Frozen Phase-3 Benchmark"
        )

        metrics_layout = QGridLayout(
            metrics_box
        )

        self.benchmark_labels = {}

        metric_names = [
            (
                "teacher_top1",
                "Teacher-forced Top-1",
            ),
            (
                "teacher_macro_f1",
                "Teacher Macro F1",
            ),
            (
                "closed_token",
                "Closed-loop token accuracy",
            ),
            (
                "closed_exact",
                "Exact programs",
            ),
            (
                "closed_valid",
                "Valid programs",
            ),
            (
                "closed_eos",
                "EOS completion",
            ),
        ]

        for index, (
            key,
            label_text,
        ) in enumerate(
            metric_names
        ):
            card = self.make_card(
                label_text,
                "—",
            )

            value_label = card.findChild(
                QLabel,
                "metricValue",
            )

            self.benchmark_labels[
                key
            ] = value_label

            row = index // 3
            column = index % 3

            metrics_layout.addWidget(
                card,
                row,
                column,
            )

        layout.addWidget(
            metrics_box
        )

        system_box = QGroupBox(
            "System Details"
        )

        system_layout = QGridLayout(
            system_box
        )

        self.system_gpu = QLabel()
        self.system_torch = QLabel()
        self.system_cuda = QLabel()
        self.system_epoch = QLabel()
        self.system_model = QLabel()
        self.system_mapping = QLabel()

        labels = [
            (
                "GPU",
                self.system_gpu,
            ),
            (
                "PyTorch",
                self.system_torch,
            ),
            (
                "CUDA runtime",
                self.system_cuda,
            ),
            (
                "Frozen model epoch",
                self.system_epoch,
            ),
            (
                "Model file",
                self.system_model,
            ),
            (
                "Mapping file",
                self.system_mapping,
            ),
        ]

        for row, (
            title,
            value,
        ) in enumerate(
            labels
        ):
            name_label = QLabel(
                title
            )

            name_label.setObjectName(
                "systemKey"
            )

            value.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )

            system_layout.addWidget(
                name_label,
                row,
                0,
            )

            system_layout.addWidget(
                value,
                row,
                1,
            )

        layout.addWidget(
            system_box
        )

        note = QLabel(
            "PyFly Studio V2 is inference-only. "
            "It does not retrain or modify the frozen model, "
            "MaleCNS recurrent weights, neural mapping, or "
            "the official Phase-3 benchmark."
        )

        note.setWordWrap(
            True
        )

        note.setObjectName(
            "helperText"
        )

        layout.addWidget(
            note
        )

        layout.addStretch(
            1
        )

    # ========================================================
    # Theme
    # ========================================================

    def theme_changed(
        self,
        text: str,
    ) -> None:
        self.current_theme = (
            "light"
            if text.lower()
            ==
            "light"
            else
            "dark"
        )

        self.apply_theme(
            self.current_theme
        )

    def apply_theme(
        self,
        theme: str,
    ) -> None:
        light = (
            theme
            ==
            "light"
        )

        self.fly_widget.set_light_theme(
            light
        )

        if light:
            self.setStyleSheet(
                """
                QWidget {
                    background-color: #f4f7fb;
                    color: #0f172a;
                    font-size: 13px;
                }
                QMainWindow {
                    background-color: #eaf0f8;
                }
                #header {
                    background-color: #ffffff;
                    border: 1px solid #dbe4ef;
                    border-radius: 14px;
                }
                #appTitle {
                    font-size: 20px;
                    font-weight: 800;
                    color: #0f172a;
                }
                #appSubtitle {
                    color: #64748b;
                }
                #gpuBadge {
                    background-color: #dcfce7;
                    color: #166534;
                    border-radius: 10px;
                    padding: 6px 10px;
                    font-weight: 700;
                }
                QGroupBox {
                    background-color: #ffffff;
                    border: 1px solid #dbe4ef;
                    border-radius: 12px;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-weight: 700;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #1d4ed8;
                }
                #metricCard {
                    background-color: #f8fafc;
                    border: 1px solid #dbe4ef;
                    border-radius: 10px;
                }
                #metricTitle {
                    color: #64748b;
                    font-size: 11px;
                }
                #metricValue {
                    color: #0f172a;
                    font-size: 15px;
                    font-weight: 800;
                }
                QLineEdit, QPlainTextEdit, QComboBox, QSpinBox,
                QTableWidget, QListWidget, QProgressBar {
                    background-color: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    selection-background-color: #bfdbfe;
                }
                QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
                    padding: 7px;
                }
                QHeaderView::section {
                    background-color: #e2e8f0;
                    color: #0f172a;
                    padding: 7px;
                    border: none;
                    font-weight: 700;
                }
                #primaryButton {
                    background-color: #2563eb;
                    color: white;
                    border: none;
                    border-radius: 9px;
                    padding: 10px 14px;
                    font-weight: 800;
                }
                #primaryButton:hover {
                    background-color: #1d4ed8;
                }
                #secondaryButton {
                    background-color: #e2e8f0;
                    color: #0f172a;
                    border: none;
                    border-radius: 9px;
                    padding: 9px 12px;
                    font-weight: 700;
                }
                #secondaryButton:hover {
                    background-color: #cbd5e1;
                }
                #dangerButton {
                    background-color: #fee2e2;
                    color: #991b1b;
                    border: none;
                    border-radius: 9px;
                    padding: 9px 12px;
                    font-weight: 700;
                }
                #dangerButton:hover {
                    background-color: #fecaca;
                }
                QPushButton:disabled {
                    background-color: #e2e8f0;
                    color: #94a3b8;
                }
                #helperText {
                    color: #64748b;
                    font-size: 11px;
                }
                #statusLabel {
                    color: #334155;
                    font-weight: 700;
                }
                #tokenChip {
                    background-color: #dbeafe;
                    color: #1e3a8a;
                    border: 1px solid #93c5fd;
                    border-radius: 10px;
                    padding: 7px 10px;
                    font-weight: 800;
                }
                QProgressBar {
                    text-align: center;
                    min-height: 18px;
                }
                QProgressBar::chunk {
                    background-color: #2563eb;
                    border-radius: 7px;
                }
                QTabBar::tab {
                    padding: 9px 20px;
                    margin-right: 4px;
                    border-radius: 8px;
                }
                QTabBar::tab:selected {
                    background-color: #dbeafe;
                    color: #1d4ed8;
                    font-weight: 800;
                }
                """
            )

        else:
            self.setStyleSheet(
                """
                QWidget {
                    background-color: #0b1220;
                    color: #e5e7eb;
                    font-size: 13px;
                }
                QMainWindow {
                    background-color: #070d18;
                }
                #header {
                    background-color: #101827;
                    border: 1px solid #263449;
                    border-radius: 14px;
                }
                #appTitle {
                    font-size: 20px;
                    font-weight: 800;
                    color: #f8fafc;
                }
                #appSubtitle {
                    color: #94a3b8;
                }
                #gpuBadge {
                    background-color: #123321;
                    color: #86efac;
                    border-radius: 10px;
                    padding: 6px 10px;
                    font-weight: 700;
                }
                QGroupBox {
                    background-color: #101827;
                    border: 1px solid #27364c;
                    border-radius: 12px;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-weight: 700;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #93c5fd;
                }
                #metricCard {
                    background-color: #0e1726;
                    border: 1px solid #28384e;
                    border-radius: 10px;
                }
                #metricTitle {
                    color: #94a3b8;
                    font-size: 11px;
                }
                #metricValue {
                    color: #f8fafc;
                    font-size: 15px;
                    font-weight: 800;
                }
                QLineEdit, QPlainTextEdit, QComboBox, QSpinBox,
                QTableWidget, QListWidget, QProgressBar {
                    background-color: #050b14;
                    color: #e5e7eb;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    selection-background-color: #1d4ed8;
                }
                QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
                    padding: 7px;
                }
                QHeaderView::section {
                    background-color: #1a2638;
                    color: #e2e8f0;
                    padding: 7px;
                    border: none;
                    font-weight: 700;
                }
                #primaryButton {
                    background-color: #2563eb;
                    color: white;
                    border: none;
                    border-radius: 9px;
                    padding: 10px 14px;
                    font-weight: 800;
                }
                #primaryButton:hover {
                    background-color: #3b82f6;
                }
                #secondaryButton {
                    background-color: #1e293b;
                    color: #e2e8f0;
                    border: none;
                    border-radius: 9px;
                    padding: 9px 12px;
                    font-weight: 700;
                }
                #secondaryButton:hover {
                    background-color: #334155;
                }
                #dangerButton {
                    background-color: #3b1820;
                    color: #fecaca;
                    border: none;
                    border-radius: 9px;
                    padding: 9px 12px;
                    font-weight: 700;
                }
                #dangerButton:hover {
                    background-color: #5b1e2b;
                }
                QPushButton:disabled {
                    background-color: #263449;
                    color: #64748b;
                }
                #helperText {
                    color: #94a3b8;
                    font-size: 11px;
                }
                #statusLabel {
                    color: #cbd5e1;
                    font-weight: 700;
                }
                #tokenChip {
                    background-color: #132b4f;
                    color: #bfdbfe;
                    border: 1px solid #2563eb;
                    border-radius: 10px;
                    padding: 7px 10px;
                    font-weight: 800;
                }
                QProgressBar {
                    text-align: center;
                    min-height: 18px;
                }
                QProgressBar::chunk {
                    background-color: #2563eb;
                    border-radius: 7px;
                }
                QTabBar::tab {
                    padding: 9px 20px;
                    margin-right: 4px;
                    border-radius: 8px;
                }
                QTabBar::tab:selected {
                    background-color: #172554;
                    color: #bfdbfe;
                    font-weight: 800;
                }
                """
            )

    # ========================================================
    # Catalog
    # ========================================================

    def populate_intents(
        self,
    ) -> None:
        self.intent_combo.blockSignals(
            True
        )

        self.intent_combo.clear()
        self.intent_combo.addItem(
            "ALL"
        )

        for intent in (
            self.engine.supported_intents()
        ):
            self.intent_combo.addItem(
                intent
            )

        self.intent_combo.blockSignals(
            False
        )

    def refresh_examples(
        self,
    ) -> None:
        intent = (
            self.intent_combo.currentText()
            or
            "ALL"
        )

        entries = (
            self.engine.filtered_examples(
                intent
            )
        )

        self.example_combo.blockSignals(
            True
        )

        self.example_combo.clear()

        for entry in entries:
            label = (
                f"{entry.example_id}  "
                f"[{entry.intent}]  "
                f"({entry.source_name})"
            )

            self.example_combo.addItem(
                label,
                entry.example_id,
            )

        self.example_combo.blockSignals(
            False
        )

        self.example_changed()

    def example_changed(
        self,
    ) -> None:
        example_id = (
            self.example_combo.currentData()
        )

        if not example_id:
            self.example_preview.clear()
            return

        entry = (
            self.engine.catalog.get(
                example_id
            )
        )

        if entry is None:
            self.example_preview.clear()
            return

        self.example_preview.setPlainText(
            " ".join(
                entry.prompt_tokens
            )
        )

    def random_example(
        self,
    ) -> None:
        if (
            self.example_combo.count()
            ==
            0
        ):
            return

        index = random.randrange(
            self.example_combo.count()
        )

        self.example_combo.setCurrentIndex(
            index
        )

        self.example_radio.setChecked(
            True
        )

    def get_prompt_tokens(
        self,
    ) -> list[str]:
        if (
            self.example_radio.isChecked()
        ):
            example_id = (
                self.example_combo.currentData()
            )

            if not example_id:
                raise ValueError(
                    "No example selected."
                )

            entry = (
                self.engine.catalog.get(
                    example_id
                )
            )

            if entry is None:
                raise ValueError(
                    "Selected example was not found."
                )

            return list(
                entry.prompt_tokens
            )

        raw = (
            self.tokens_edit.text().strip()
        )

        if not raw:
            raise ValueError(
                "Please enter custom prompt tokens."
            )

        return [
            token
            for token in raw.split()
            if token
        ]

    # ========================================================
    # Generation
    # ========================================================

    def append_log(
        self,
        text: str,
    ) -> None:
        self.log_output.appendPlainText(
            text
        )

        cursor = (
            self.log_output.textCursor()
        )

        cursor.movePosition(
            QTextCursor.End
        )

        self.log_output.setTextCursor(
            cursor
        )

    def clear_token_chips(
        self,
    ) -> None:
        while (
            self.token_layout.count()
            >
            1
        ):
            item = (
                self.token_layout.takeAt(
                    0
                )
            )

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def add_token_chip(
        self,
        token: str,
        confidence: float,
    ) -> None:
        chip = QLabel(
            f"{token}\n"
            f"{confidence * 100:.1f}%"
        )

        chip.setAlignment(
            Qt.AlignCenter
        )

        chip.setObjectName(
            "tokenChip"
        )

        self.token_layout.insertWidget(
            self.token_layout.count()
            -
            1,
            chip,
        )

        QTimer.singleShot(
            0,
            lambda:
                self.token_scroll.horizontalScrollBar().setValue(
                    self.token_scroll.horizontalScrollBar().maximum()
                ),
        )

    def start_generation(
        self,
    ) -> None:
        if (
            self.worker_thread
            is not None
        ):
            QMessageBox.information(
                self,
                "PyFly",
                "Generation is already running.",
            )

            return

        try:
            prompt_tokens = (
                self.get_prompt_tokens()
            )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "PyFly",
                str(
                    exc
                ),
            )

            return

        neural_seed = int(
            self.seed_spin.value()
        )

        self.current_result = None

        self.steps_table.setRowCount(
            0
        )

        self.code_output.clear()
        self.log_output.clear()
        self.clear_token_chips()

        self.progress_bar.setRange(
            0,
            self.engine.max_generation_steps,
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setFormat(
            "Starting…"
        )

        self.current_prompt_edit.setPlainText(
            " ".join(
                prompt_tokens
            )
        )

        self.status_label.setText(
            "Starting true closed-loop generation…"
        )

        self.fly_widget.set_active(
            True
        )

        self.generate_button.setEnabled(
            False
        )

        self.cancel_button.setEnabled(
            True
        )

        self.random_button.setEnabled(
            False
        )

        self.append_log(
            "PyFly generation started."
        )

        self.append_log(
            f"Neural seed: {neural_seed}"
        )

        self.append_log(
            "Prompt: "
            +
            " ".join(
                prompt_tokens
            )
        )

        self.worker_thread = QThread()

        self.worker = GenerationWorker(
            engine=self.engine,
            prompt_tokens=prompt_tokens,
            neural_seed=neural_seed,
        )

        self.worker.moveToThread(
            self.worker_thread
        )

        self.worker_thread.started.connect(
            self.worker.run
        )

        self.worker.progress.connect(
            self.handle_progress
        )

        self.worker.status.connect(
            self.handle_status
        )

        self.worker.finished.connect(
            self.handle_finished
        )

        self.worker.cancelled.connect(
            self.handle_cancelled
        )

        self.worker.error.connect(
            self.handle_error
        )

        self.worker.finished.connect(
            self.worker_thread.quit
        )

        self.worker.cancelled.connect(
            self.worker_thread.quit
        )

        self.worker.error.connect(
            self.worker_thread.quit
        )

        self.worker_thread.finished.connect(
            self.cleanup_worker
        )

        self.worker_thread.start()

    def cancel_generation(
        self,
    ) -> None:
        if self.worker is None:
            return

        self.worker.request_cancel()

        self.status_label.setText(
            "Cancelling generation…"
        )

        self.cancel_button.setEnabled(
            False
        )

    def handle_status(
        self,
        text: str,
    ) -> None:
        self.status_label.setText(
            text
        )

        self.append_log(
            text
        )

    def handle_progress(
        self,
        info: dict,
    ) -> None:
        position = int(
            info[
                "position"
            ]
        )

        confidence = float(
            info[
                "confidence"
            ]
        )

        self.progress_bar.setValue(
            position + 1
        )

        self.progress_bar.setFormat(
            f"Token {position + 1}/"
            f"{self.engine.max_generation_steps}"
        )

        self.fly_widget.set_token_activity(
            info[
                "prediction"
            ],
            confidence,
        )

        self.add_token_chip(
            info[
                "prediction"
            ],
            confidence,
        )

        row = (
            self.steps_table.rowCount()
        )

        self.steps_table.insertRow(
            row
        )

        top3 = info[
            "top3"
        ]

        alternatives = " | ".join(
            f"{item['token']} "
            f"{item['probability'] * 100:.1f}%"
            for item in top3[
                1:
            ]
        )

        self.steps_table.setItem(
            row,
            0,
            QTableWidgetItem(
                str(
                    position
                )
            ),
        )

        self.steps_table.setItem(
            row,
            1,
            QTableWidgetItem(
                info[
                    "feedback_token"
                ]
            ),
        )

        self.steps_table.setItem(
            row,
            2,
            QTableWidgetItem(
                info[
                    "prediction"
                ]
            ),
        )

        self.steps_table.setItem(
            row,
            3,
            QTableWidgetItem(
                f"{confidence * 100:.1f}%"
            ),
        )

        self.steps_table.setItem(
            row,
            4,
            QTableWidgetItem(
                alternatives
            ),
        )

        self.steps_table.scrollToBottom()

        self.append_log(
            f"P{position}: "
            f"{info['prediction']} "
            f"({confidence * 100:.1f}%)"
        )

    def handle_finished(
        self,
        result: dict,
    ) -> None:
        self.current_result = result

        generated_tokens = (
            result[
                "generated_tokens"
            ]
        )

        self.progress_bar.setValue(
            len(
                generated_tokens
            )
        )

        self.progress_bar.setFormat(
            "Generation complete"
        )

        if (
            result[
                "code"
            ]
            is not None
        ):
            self.code_output.setPlainText(
                result[
                    "code"
                ]
            )

            self.status_label.setText(
                "Done — Python generated successfully."
            )

        else:
            self.code_output.setPlainText(
                "Could not convert the generated token "
                "sequence to Python.\n\n"
                +
                (
                    result[
                        "code_error"
                    ]
                    or
                    "Unknown conversion error."
                )
            )

            self.status_label.setText(
                "Done — token sequence could not be rendered."
            )

        self.append_log(
            ""
        )

        self.append_log(
            "Generated tokens:"
        )

        self.append_log(
            "  "
            +
            " ".join(
                generated_tokens
            )
        )

        self.add_history_entry(
            result
        )

    def handle_cancelled(
        self,
        message: str,
    ) -> None:
        self.status_label.setText(
            "Generation cancelled."
        )

        self.progress_bar.setFormat(
            "Cancelled"
        )

        self.append_log(
            message
        )

    def handle_error(
        self,
        message: str,
    ) -> None:
        self.status_label.setText(
            "Generation failed."
        )

        self.progress_bar.setFormat(
            "Error"
        )

        self.append_log(
            ""
        )

        self.append_log(
            message
        )

        QMessageBox.critical(
            self,
            "PyFly Error",
            message,
        )

    def cleanup_worker(
        self,
    ) -> None:
        self.generate_button.setEnabled(
            True
        )

        self.cancel_button.setEnabled(
            False
        )

        self.random_button.setEnabled(
            True
        )

        self.fly_widget.set_active(
            False
        )

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if (
            self.worker_thread
            is not None
        ):
            self.worker_thread.deleteLater()
            self.worker_thread = None

    # ========================================================
    # Code actions
    # ========================================================

    def copy_code(
        self,
    ) -> None:
        text = (
            self.code_output.toPlainText().strip()
        )

        if not text:
            QMessageBox.information(
                self,
                "PyFly",
                "There is no generated code to copy.",
            )

            return

        QApplication.clipboard().setText(
            text
        )

        self.status_label.setText(
            "Generated code copied to clipboard."
        )

    def save_code(
        self,
    ) -> None:
        text = (
            self.code_output.toPlainText().strip()
        )

        if not text:
            QMessageBox.information(
                self,
                "PyFly",
                "There is no generated code to save.",
            )

            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save generated Python",
            str(
                PROJECT_ROOT
                /
                "pyfly_generated.py"
            ),
            "Python files (*.py);;All files (*)",
        )

        if not filename:
            return

        path = Path(
            filename
        )

        path.write_text(
            text
            +
            "\n",
            encoding="utf-8",
        )

        self.status_label.setText(
            f"Saved to {path}"
        )

    def clear_output(
        self,
    ) -> None:
        self.current_result = None

        self.steps_table.setRowCount(
            0
        )

        self.code_output.clear()
        self.log_output.clear()
        self.current_prompt_edit.clear()
        self.clear_token_chips()

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setFormat(
            "Ready"
        )

        self.status_label.setText(
            "Ready"
        )

        self.fly_widget.set_active(
            False
        )

        self.fly_widget.set_token_activity(
            "",
            0.0,
        )

    # ========================================================
    # History
    # ========================================================

    def add_history_entry(
        self,
        result: dict,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "seed": int(
                result[
                    "neural_seed"
                ]
            ),
            "prompt_tokens": (
                result[
                    "prompt_tokens"
                ]
            ),
            "generated_tokens": (
                result[
                    "generated_tokens"
                ]
            ),
            "code": (
                result[
                    "code"
                ]
            ),
        }

        self.history.append(
            entry
        )

        self.history = self.history[
            -100:
        ]

        self.save_history()
        self.refresh_history()

    def refresh_history(
        self,
    ) -> None:
        self.history_list.clear()

        for history_index in range(
            len(
                self.history
            )
            -
            1,
            -1,
            -1,
        ):
            entry = self.history[
                history_index
            ]

            prompt_preview = " ".join(
                entry.get(
                    "prompt_tokens",
                    [],
                )
            )

            item = QListWidgetItem(
                f"{entry.get('timestamp', '')}  "
                f"| seed={entry.get('seed', '?')}  "
                f"| {prompt_preview}"
            )

            item.setData(
                Qt.UserRole,
                history_index,
            )

            self.history_list.addItem(
                item
            )

        if (
            self.history_list.count()
            >
            0
        ):
            self.history_list.setCurrentRow(
                0
            )

        else:
            self.history_detail.clear()

    def history_selected(
        self,
        row: int,
    ) -> None:
        if row < 0:
            self.history_detail.clear()
            return

        item = (
            self.history_list.item(
                row
            )
        )

        if item is None:
            return

        index = item.data(
            Qt.UserRole
        )

        if (
            index is None
            or
            index
            >=
            len(
                self.history
            )
        ):
            return

        entry = self.history[
            index
        ]

        text = (
            f"Time: {entry.get('timestamp', '')}\n"
            f"Seed: {entry.get('seed', '')}\n\n"
            f"Prompt tokens:\n"
            f"{' '.join(entry.get('prompt_tokens', []))}\n\n"
            f"Generated tokens:\n"
            f"{' '.join(entry.get('generated_tokens', []))}\n\n"
            f"Generated Python:\n"
            f"{entry.get('code') or '<not renderable>'}"
        )

        self.history_detail.setPlainText(
            text
        )

    def reuse_history_prompt(
        self,
    ) -> None:
        row = (
            self.history_list.currentRow()
        )

        if row < 0:
            return

        item = self.history_list.item(
            row
        )

        if item is None:
            return

        index = item.data(
            Qt.UserRole
        )

        entry = self.history[
            index
        ]

        self.custom_radio.setChecked(
            True
        )

        self.tokens_edit.setText(
            " ".join(
                entry.get(
                    "prompt_tokens",
                    [],
                )
            )
        )

        self.seed_spin.setValue(
            int(
                entry.get(
                    "seed",
                    DEFAULT_NEURAL_SEED,
                )
            )
        )

        self.tabs.setCurrentWidget(
            self.generate_tab
        )

    def clear_history(
        self,
    ) -> None:
        answer = QMessageBox.question(
            self,
            "Clear history",
            "Delete the local PyFly Studio V2 history?",
        )

        if (
            answer
            !=
            QMessageBox.Yes
        ):
            return

        self.history.clear()

        try:
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()

        except OSError:
            pass

        self.refresh_history()

    # ========================================================
    # System page
    # ========================================================

    def populate_system_info(
        self,
    ) -> None:
        self.system_gpu.setText(
            torch.cuda.get_device_name(
                0
            )
        )

        self.system_torch.setText(
            torch.__version__
        )

        self.system_cuda.setText(
            str(
                torch.version.cuda
            )
        )

        self.system_epoch.setText(
            str(
                self.engine.best_epoch
            )
        )

        self.system_model.setText(
            str(
                MODEL_FILE
            )
        )

        self.system_mapping.setText(
            str(
                MAPPING_FILE
            )
        )

        if not FINAL_METRICS_FILE.exists():
            return

        try:
            metrics = json.loads(
                FINAL_METRICS_FILE.read_text(
                    encoding="utf-8"
                )
            )

            teacher = metrics.get(
                "teacher_forced_final",
                {},
            )

            closed = metrics.get(
                "closed_loop_final",
                {},
            )

            values = {
                "teacher_top1": (
                    teacher.get(
                        "top1_accuracy"
                    )
                ),
                "teacher_macro_f1": (
                    teacher.get(
                        "macro_f1"
                    )
                ),
                "closed_token": (
                    closed.get(
                        "mean_token_accuracy"
                    )
                ),
                "closed_exact": (
                    closed.get(
                        "exact_program_accuracy"
                    )
                ),
                "closed_valid": (
                    closed.get(
                        "valid_program_rate"
                    )
                ),
                "closed_eos": (
                    closed.get(
                        "eos_completion_rate"
                    )
                ),
            }

            for key, value in (
                values.items()
            ):
                label = (
                    self.benchmark_labels.get(
                        key
                    )
                )

                if (
                    label is not None
                    and
                    value is not None
                ):
                    label.setText(
                        f"{float(value) * 100:.2f}%"
                    )

        except Exception:
            pass

    # ========================================================
    # Close handling
    # ========================================================

    def closeEvent(
        self,
        event,
    ) -> None:
        if (
            self.worker_thread
            is not None
        ):
            answer = QMessageBox.question(
                self,
                "PyFly is running",
                "Generation is still running. "
                "Cancel and close the app?",
            )

            if (
                answer
                !=
                QMessageBox.Yes
            ):
                event.ignore()
                return

            if self.worker is not None:
                self.worker.request_cancel()

            self.worker_thread.quit()
            self.worker_thread.wait(
                1500
            )

        event.accept()


# ============================================================
# Startup
# ============================================================

def main() -> None:
    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "PyFly Studio V2"
    )

    app.setStyle(
        "Fusion"
    )

    try:
        engine = PyFlyEngine()

    except Exception as exc:
        QMessageBox.critical(
            None,
            "PyFly Startup Error",
            f"{type(exc).__name__}: {exc}",
        )

        raise

    window = MainWindow(
        engine
    )

    # V2.2: screen-aware 94% window rather than forced maximize.
    # This leaves a small desktop margin on a 16:9 laptop and
    # avoids the cramped feeling caused by Windows taskbar/DPI.
    screen = QGuiApplication.primaryScreen()

    if screen is not None:
        available = screen.availableGeometry()

        frame = window.frameGeometry()
        frame.moveCenter(
            available.center()
        )

        window.move(
            frame.topLeft()
        )

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
