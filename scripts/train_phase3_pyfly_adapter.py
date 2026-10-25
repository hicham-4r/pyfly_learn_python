from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "\nPyTorch is not installed in this virtual environment.\n"
        "Install the CUDA build of PyTorch, then run this script again.\n"
    ) from exc

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Phase 3 — Trainable PyFly
# Step 80 — GPU neural adapter on REAL MaleCNS features
#
# Architecture:
#
#     PyFly prompt / feedback stimulation
#                  ↓
#          real MaleCNS activity
#                  ↓
#       41,111 frozen neural features
#                  ↓
#        TRAINABLE PyTorch adapter
#                  ↓
#           9 Python tokens
#
# Data policy:
#   generator_train      -> gradient updates
#   generator_validation -> early stopping / checkpoint choice
#   generator_test       -> development test after training
#   old final holdout    -> NEVER opened here
#
# This is the start of the proper GPU training phase.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Input feature files
# ============================================================

TRAIN_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_train.npz"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_validation.npz"
)

DEV_TEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "generator_test.npz"
)


# ============================================================
# Outputs
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

BEST_MODEL_FILE = (
    MODEL_DIR
    / "phase3_pyfly_adapter.pt"
)

SCALER_FILE = (
    MODEL_DIR
    / "phase3_pyfly_adapter_scaler.npz"
)

TRAINING_METRICS_FILE = (
    METRICS_DIR
    / "phase3_pyfly_adapter_training.json"
)

FROZEN_CONFIG_FILE = (
    METRICS_DIR
    / "phase3_pyfly_adapter_config.json"
)


# ============================================================
# Expected frozen feature format
# ============================================================

EXPECTED_FEATURE_DIM = 41_111

EXPECTED_ROWS = {
    "train": 1_140,
    "validation": 190,
    "development_test": 190,
}

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# PRECOMMITTED Step-80 training settings
# ============================================================

RANDOM_SEED = 80_001

BATCH_SIZE = 64

MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 12

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
SECOND_HIDDEN_DIM = 256
POSITION_EMBED_DIM = 32

DROPOUT = 0.20
INPUT_DROPOUT = 0.05

GRADIENT_CLIP_NORM = 1.0

NUM_WORKERS = 0

# Conservative scheduler: reduce LR when validation Macro-F1
# stops improving.
LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 4
MIN_LR = 1e-6


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Load one MaleCNS feature split
# ============================================================

def load_split(
    path: Path,
    expected_rows: int,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing feature file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "X",
        "targets",
        "target_positions",
        "example_ids",
        "seeds",
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
            f"{path.name} is missing arrays: "
            f"{sorted(missing)}"
        )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    targets = data[
        "targets"
    ].astype(str)

    positions = data[
        "target_positions"
    ].astype(
        np.int64
    )

    example_ids = data[
        "example_ids"
    ].astype(str)

    neural_seeds = data[
        "seeds"
    ].astype(
        np.int64
    )

    if (
        X.shape
        !=
        (
            expected_rows,
            EXPECTED_FEATURE_DIM,
        )
    ):

        raise ValueError(
            f"{path.name}: expected "
            f"({expected_rows}, "
            f"{EXPECTED_FEATURE_DIM}), "
            f"got {X.shape}"
        )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            f"{path.name}: non-finite "
            "features found."
        )

    return {
        "X": X,
        "targets": targets,
        "positions": positions,
        "example_ids": example_ids,
        "neural_seeds": neural_seeds,
    }


# ============================================================
# Dataset
# ============================================================

class MaleCNSFeatureDataset(
    Dataset
):

    def __init__(
        self,
        X: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:

        # Standardize once on CPU to keep training loop simple.
        standardized = (
            (
                X
                -
                mean
            )
            /
            std
        ).astype(
            np.float32,
            copy=False,
        )

        self.X = torch.from_numpy(
            standardized
        )

        self.positions = torch.from_numpy(
            positions.astype(
                np.int64,
                copy=False,
            )
        )

        self.labels = torch.from_numpy(
            labels.astype(
                np.int64,
                copy=False,
            )
        )

    def __len__(
        self,
    ) -> int:

        return int(
            len(
                self.labels
            )
        )

    def __getitem__(
        self,
        index: int,
    ):

        return (
            self.X[
                index
            ],
            self.positions[
                index
            ],
            self.labels[
                index
            ],
        )


# ============================================================
# Trainable PyFly adapter
# ============================================================

class PyFlyNeuralAdapter(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_position: int,
    ) -> None:

        super().__init__()

        self.input_norm = nn.LayerNorm(
            input_dim
        )

        self.input_dropout = nn.Dropout(
            INPUT_DROPOUT
        )

        self.neural_projection = nn.Linear(
            input_dim,
            HIDDEN_DIM,
        )

        self.position_embedding = nn.Embedding(
            max_position
            +
            1,
            POSITION_EMBED_DIM,
        )

        merged_dim = (
            HIDDEN_DIM
            +
            POSITION_EMBED_DIM
        )

        self.adapter = nn.Sequential(
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                merged_dim,
                SECOND_HIDDEN_DIM,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.LayerNorm(
                SECOND_HIDDEN_DIM
            ),
        )

        self.classifier = nn.Linear(
            SECOND_HIDDEN_DIM,
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
# Metrics
# ============================================================

def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    positions: np.ndarray,
    logits: np.ndarray | None = None,
) -> dict:

    semantic_mask = np.isin(
        positions,
        SEMANTIC_POSITIONS,
    )

    result = {
        "top1_accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),

        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),

        "semantic_accuracy": float(
            accuracy_score(
                true_labels[
                    semantic_mask
                ],
                predicted_labels[
                    semantic_mask
                ],
            )
        ),

        "semantic_macro_f1": float(
            f1_score(
                true_labels[
                    semantic_mask
                ],
                predicted_labels[
                    semantic_mask
                ],
                average="macro",
                zero_division=0,
            )
        ),
    }

    position_accuracy = {}

    for position in sorted(
        np.unique(
            positions
        )
    ):

        mask = (
            positions
            ==
            position
        )

        position_accuracy[
            str(
                int(
                    position
                )
            )
        ] = float(
            accuracy_score(
                true_labels[
                    mask
                ],
                predicted_labels[
                    mask
                ],
            )
        )

    result[
        "accuracy_by_position"
    ] = (
        position_accuracy
    )

    if logits is not None:

        # top-k
        order = np.argsort(
            logits,
            axis=1,
        )

        for k in [
            2,
            3,
        ]:

            top_k = order[
                :,
                -k:
            ]

            correct = np.any(
                top_k
                ==
                true_labels[
                    :,
                    None,
                ],
                axis=1,
            )

            result[
                f"top{k}_accuracy"
            ] = float(
                np.mean(
                    correct
                )
            )

    return result


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    float,
    dict,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    model.eval()

    losses = []

    all_true = []
    all_pred = []
    all_positions = []
    all_logits = []

    criterion = nn.CrossEntropyLoss(
        reduction="mean"
    )

    for (
        X,
        positions,
        labels,
    ) in loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        positions = positions.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            X,
            positions,
        )

        loss = criterion(
            logits,
            labels,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        losses.append(
            float(
                loss.item()
            )
        )

        all_true.append(
            labels.cpu().numpy()
        )

        all_pred.append(
            predictions.cpu().numpy()
        )

        all_positions.append(
            positions.cpu().numpy()
        )

        all_logits.append(
            logits.cpu().numpy()
        )

    true_labels = np.concatenate(
        all_true
    )

    predicted_labels = np.concatenate(
        all_pred
    )

    positions = np.concatenate(
        all_positions
    )

    logits = np.concatenate(
        all_logits
    )

    metrics = calculate_metrics(
        true_labels=(
            true_labels
        ),
        predicted_labels=(
            predicted_labels
        ),
        positions=(
            positions
        ),
        logits=(
            logits
        ),
    )

    mean_loss = float(
        np.mean(
            losses
        )
    )

    return (
        mean_loss,
        metrics,
        true_labels,
        predicted_labels,
        logits,
    )


# ============================================================
# Model size
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    seed_everything(
        RANDOM_SEED
    )

    print()

    print(
        "===================================================="
    )

    print(
        "       PHASE 3 - TRAINABLE PYFLY"
    )

    print(
        "       STEP 80 - GPU NEURAL ADAPTER"
    )

    print(
        "===================================================="
    )

    print()

    # ========================================================
    # GPU requirement
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available to PyTorch.\n"
            "Step 80 is intended to train on your NVIDIA GPU."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"PyTorch version:           "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available:            "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"GPU:                       "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA runtime:              "
        f"{torch.version.cuda}"
    )

    print()

    print(
        "Old final holdout used:    NO"
    )

    # ========================================================
    # Load frozen MaleCNS features
    # ========================================================

    train = load_split(
        TRAIN_FILE,
        EXPECTED_ROWS[
            "train"
        ],
    )

    validation = load_split(
        VALIDATION_FILE,
        EXPECTED_ROWS[
            "validation"
        ],
    )

    dev_test = load_split(
        DEV_TEST_FILE,
        EXPECTED_ROWS[
            "development_test"
        ],
    )

    # Logical-prompt separation
    train_examples = set(
        train[
            "example_ids"
        ].tolist()
    )

    validation_examples = set(
        validation[
            "example_ids"
        ].tolist()
    )

    test_examples = set(
        dev_test[
            "example_ids"
        ].tolist()
    )

    if not train_examples.isdisjoint(
        validation_examples
    ):

        raise ValueError(
            "Train/validation prompt leakage."
        )

    if not train_examples.isdisjoint(
        test_examples
    ):

        raise ValueError(
            "Train/test prompt leakage."
        )

    if not validation_examples.isdisjoint(
        test_examples
    ):

        raise ValueError(
            "Validation/test prompt leakage."
        )

    print()

    print(
        f"Training rows:             "
        f"{len(train['X']):,}"
    )

    print(
        f"Validation rows:           "
        f"{len(validation['X']):,}"
    )

    print(
        f"Development-test rows:     "
        f"{len(dev_test['X']):,}"
    )

    print(
        f"MaleCNS feature dimension: "
        f"{train['X'].shape[1]:,}"
    )

    print()

    print(
        f"Training logical prompts:  "
        f"{len(train_examples)}"
    )

    print(
        f"Validation prompts:        "
        f"{len(validation_examples)}"
    )

    print(
        f"Development-test prompts:  "
        f"{len(test_examples)}"
    )

    # ========================================================
    # Token vocabulary
    # ========================================================

    token_classes = sorted(
        np.unique(
            train[
                "targets"
            ]
        ).tolist()
    )

    token_to_id = {
        token: index

        for index, token
        in enumerate(
            token_classes
        )
    }

    for split_name, split in [
        (
            "validation",
            validation,
        ),
        (
            "development_test",
            dev_test,
        ),
    ]:

        unknown = sorted(
            set(
                split[
                    "targets"
                ].tolist()
            )
            -
            set(
                token_classes
            )
        )

        if unknown:

            raise ValueError(
                f"{split_name} contains unseen "
                f"target tokens: {unknown}"
            )

    print()

    print(
        f"Target classes ({len(token_classes)}):"
    )

    print(
        "  "
        +
        ", ".join(
            token_classes
        )
    )

    # ========================================================
    # Train-only feature standardization
    # ========================================================

    print()

    print(
        "Computing train-only feature scaling..."
    )

    feature_mean = np.mean(
        train[
            "X"
        ],
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    feature_std = np.std(
        train[
            "X"
        ],
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    # Avoid divide-by-zero on inactive neurons/features.
    feature_std[
        feature_std
        <
        1e-6
    ] = np.float32(
        1.0
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        SCALER_FILE,

        mean=feature_mean,
        std=feature_std,
        token_classes=np.asarray(
            token_classes
        ),
    )

    # ========================================================
    # Integer labels
    # ========================================================

    def encode_targets(
        targets: np.ndarray,
    ) -> np.ndarray:

        return np.asarray(
            [
                token_to_id[
                    token
                ]

                for token
                in targets
            ],
            dtype=np.int64,
        )

    train_y = encode_targets(
        train[
            "targets"
        ]
    )

    validation_y = encode_targets(
        validation[
            "targets"
        ]
    )

    dev_test_y = encode_targets(
        dev_test[
            "targets"
        ]
    )

    # ========================================================
    # Dataset / loaders
    # ========================================================

    train_dataset = (
        MaleCNSFeatureDataset(
            X=train[
                "X"
            ],
            positions=train[
                "positions"
            ],
            labels=train_y,
            mean=feature_mean,
            std=feature_std,
        )
    )

    validation_dataset = (
        MaleCNSFeatureDataset(
            X=validation[
                "X"
            ],
            positions=validation[
                "positions"
            ],
            labels=validation_y,
            mean=feature_mean,
            std=feature_std,
        )
    )

    dev_test_dataset = (
        MaleCNSFeatureDataset(
            X=dev_test[
                "X"
            ],
            positions=dev_test[
                "positions"
            ],
            labels=dev_test_y,
            mean=feature_mean,
            std=feature_std,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    dev_test_loader = DataLoader(
        dev_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # ========================================================
    # Class-balanced loss
    # ========================================================

    class_counts = np.bincount(
        train_y,
        minlength=len(
            token_classes
        ),
    ).astype(
        np.float64
    )

    # sqrt inverse frequency is gentler than full inverse
    # frequency and works well for this small dataset.
    class_weights = np.sqrt(
        np.max(
            class_counts
        )
        /
        class_counts
    )

    class_weights = (
        class_weights
        /
        np.mean(
            class_weights
        )
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=(
            class_weights_tensor
        )
    )

    # ========================================================
    # Model
    # ========================================================

    max_position = int(
        max(
            np.max(
                train[
                    "positions"
                ]
            ),
            np.max(
                validation[
                    "positions"
                ]
            ),
            np.max(
                dev_test[
                    "positions"
                ]
            ),
        )
    )

    model = PyFlyNeuralAdapter(
        input_dim=(
            EXPECTED_FEATURE_DIM
        ),
        num_classes=len(
            token_classes
        ),
        max_position=(
            max_position
        ),
    ).to(
        device
    )

    parameter_count = count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_REDUCTION_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
        )
    )

    print()

    print(
        "Model:"
    )

    print(
        f"  Hidden dimension:        "
        f"{HIDDEN_DIM}"
    )

    print(
        f"  Second hidden:           "
        f"{SECOND_HIDDEN_DIM}"
    )

    print(
        f"  Position embedding:      "
        f"{POSITION_EMBED_DIM}"
    )

    print(
        f"  Dropout:                 "
        f"{DROPOUT}"
    )

    print(
        f"  Trainable parameters:    "
        f"{parameter_count:,}"
    )

    print(
        f"  Batch size:              "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Learning rate:           "
        f"{LEARNING_RATE}"
    )

    print(
        f"  Maximum epochs:          "
        f"{MAX_EPOCHS}"
    )

    print(
        f"  Early-stop patience:     "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    # ========================================================
    # Baseline validation before gradient training
    # ========================================================

    baseline_loss, (
        baseline_metrics
    ), _, _, _ = evaluate(
        model,
        validation_loader,
        device,
    )

    print()

    print(
        "===================================================="
    )

    print(
        "       VALIDATION BEFORE GPU TRAINING"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Loss:               "
        f"{baseline_loss:.6f}"
    )

    print(
        f"Top-1:              "
        f"{baseline_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:           "
        f"{baseline_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Semantic accuracy:  "
        f"{baseline_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    # ========================================================
    # Training
    # ========================================================

    best_validation_macro_f1 = (
        -math.inf
    )

    best_validation_semantic = (
        -math.inf
    )

    best_validation_top1 = (
        -math.inf
    )

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    print()

    print(
        "===================================================="
    )

    print(
        "              GPU TRAINING START"
    )

    print(
        "===================================================="
    )

    for epoch in range(
        1,
        MAX_EPOCHS
        +
        1,
    ):

        model.train()

        training_losses = []

        training_true = []
        training_pred = []
        training_positions = []

        for (
            X,
            positions,
            labels,
        ) in train_loader:

            X = X.to(
                device,
                non_blocking=True,
            )

            positions = positions.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X,
                positions,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM,
            )

            optimizer.step()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            training_losses.append(
                float(
                    loss.item()
                )
            )

            training_true.append(
                labels.detach().cpu().numpy()
            )

            training_pred.append(
                predictions.detach().cpu().numpy()
            )

            training_positions.append(
                positions.detach().cpu().numpy()
            )

        train_true = np.concatenate(
            training_true
        )

        train_pred = np.concatenate(
            training_pred
        )

        train_positions = np.concatenate(
            training_positions
        )

        train_metrics = calculate_metrics(
            true_labels=train_true,
            predicted_labels=train_pred,
            positions=train_positions,
        )

        train_loss = float(
            np.mean(
                training_losses
            )
        )

        (
            validation_loss,
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            device,
        )

        scheduler.step(
            validation_metrics[
                "macro_f1"
            ]
        )

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        epoch_row = {
            "epoch": int(
                epoch
            ),

            "learning_rate": (
                current_lr
            ),

            "train_loss": (
                train_loss
            ),

            "train": (
                train_metrics
            ),

            "validation_loss": (
                validation_loss
            ),

            "validation": (
                validation_metrics
            ),
        }

        history.append(
            epoch_row
        )

        print(
            f"Epoch {epoch:03} | "
            f"lr={current_lr:.2e} | "
            f"train loss={train_loss:.4f} "
            f"Top-1={train_metrics['top1_accuracy'] * 100:6.2f}% | "
            f"val loss={validation_loss:.4f} "
            f"Top-1={validation_metrics['top1_accuracy'] * 100:6.2f}% "
            f"MacroF1={validation_metrics['macro_f1'] * 100:6.2f}% "
            f"Semantic={validation_metrics['semantic_accuracy'] * 100:6.2f}%"
        )

        # ----------------------------------------------------
        # Selection priority:
        #   1. validation Macro F1
        #   2. validation semantic accuracy
        #   3. validation Top-1
        # ----------------------------------------------------

        candidate_key = (
            validation_metrics[
                "macro_f1"
            ],
            validation_metrics[
                "semantic_accuracy"
            ],
            validation_metrics[
                "top1_accuracy"
            ],
        )

        best_key = (
            best_validation_macro_f1,
            best_validation_semantic,
            best_validation_top1,
        )

        if candidate_key > best_key:

            best_validation_macro_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            best_validation_semantic = (
                validation_metrics[
                    "semantic_accuracy"
                ]
            )

            best_validation_top1 = (
                validation_metrics[
                    "top1_accuracy"
                ]
            )

            best_epoch = int(
                epoch
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "step": "80",
                    "phase": 3,
                    "model_class": (
                        "PyFlyNeuralAdapter"
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "token_classes": (
                        token_classes
                    ),
                    "input_dim": (
                        EXPECTED_FEATURE_DIM
                    ),
                    "hidden_dim": (
                        HIDDEN_DIM
                    ),
                    "second_hidden_dim": (
                        SECOND_HIDDEN_DIM
                    ),
                    "position_embed_dim": (
                        POSITION_EMBED_DIM
                    ),
                    "max_position": (
                        max_position
                    ),
                    "dropout": (
                        DROPOUT
                    ),
                    "input_dropout": (
                        INPUT_DROPOUT
                    ),
                    "best_epoch": (
                        best_epoch
                    ),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "random_seed": (
                        RANDOM_SEED
                    ),
                },
                BEST_MODEL_FILE,
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                f"Early stopping after epoch "
                f"{epoch}: no validation improvement "
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )

            break

    # ========================================================
    # Load best checkpoint
    # ========================================================

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # Best validation result
    # ========================================================

    (
        best_validation_loss,
        best_validation_metrics,
        _,
        _,
        _,
    ) = evaluate(
        model,
        validation_loader,
        device,
    )

    # ========================================================
    # DEVELOPMENT TEST — once after training
    # ========================================================

    (
        dev_test_loss,
        dev_test_metrics,
        _,
        _,
        _,
    ) = evaluate(
        model,
        dev_test_loader,
        device,
    )

    # ========================================================
    # Save config / metrics
    # ========================================================

    frozen_config = {
        "phase": 3,

        "step": "80",

        "status": (
            "PHASE3_ADAPTER_TRAINED"
        ),

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "architecture": {
            "input_dim": (
                EXPECTED_FEATURE_DIM
            ),

            "hidden_dim": (
                HIDDEN_DIM
            ),

            "second_hidden_dim": (
                SECOND_HIDDEN_DIM
            ),

            "position_embed_dim": (
                POSITION_EMBED_DIM
            ),

            "dropout": (
                DROPOUT
            ),

            "input_dropout": (
                INPUT_DROPOUT
            ),

            "num_classes": int(
                len(
                    token_classes
                )
            ),

            "token_classes": (
                token_classes
            ),

            "max_position": (
                max_position
            ),
        },

        "training": {
            "random_seed": (
                RANDOM_SEED
            ),

            "batch_size": (
                BATCH_SIZE
            ),

            "maximum_epochs": (
                MAX_EPOCHS
            ),

            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),

            "class_weighting": (
                "sqrt inverse train frequency"
            ),

            "optimizer": (
                "AdamW"
            ),

            "scheduler": (
                "ReduceLROnPlateau on validation Macro F1"
            ),
        },

        "selection_rule": (
            "validation Macro F1, then semantic accuracy, "
            "then Top-1"
        ),

        "best_epoch": (
            best_epoch
        ),

        "old_final_holdout_used": (
            False
        ),

        "model_file": str(
            BEST_MODEL_FILE.relative_to(
                PROJECT_ROOT
            )
        ).replace(
            "\\",
            "/",
        ),

        "scaler_file": str(
            SCALER_FILE.relative_to(
                PROJECT_ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
    }

    FROZEN_CONFIG_FILE.write_text(
        json.dumps(
            frozen_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    output = {
        "phase": 3,

        "step": "80",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "GPU_NEURAL_ADAPTER_TRAINING_COMPLETE"
        ),

        "gpu": (
            torch.cuda.get_device_name(
                0
            )
        ),

        "pytorch_version": (
            torch.__version__
        ),

        "cuda_version": (
            torch.version.cuda
        ),

        "trainable_parameters": (
            parameter_count
        ),

        "token_classes": (
            token_classes
        ),

        "baseline_validation": {
            "loss": (
                baseline_loss
            ),

            **baseline_metrics,
        },

        "best_epoch": (
            best_epoch
        ),

        "best_validation": {
            "loss": (
                best_validation_loss
            ),

            **best_validation_metrics,
        },

        "development_test": {
            "loss": (
                dev_test_loss
            ),

            **dev_test_metrics,
        },

        "history": (
            history
        ),

        "old_final_holdout_used": (
            False
        ),

        "artifacts": {
            "model": str(
                BEST_MODEL_FILE.relative_to(
                    PROJECT_ROOT
                )
            ).replace(
                "\\",
                "/",
            ),

            "scaler": str(
                SCALER_FILE.relative_to(
                    PROJECT_ROOT
                )
            ).replace(
                "\\",
                "/",
            ),

            "config": str(
                FROZEN_CONFIG_FILE.relative_to(
                    PROJECT_ROOT
                )
            ).replace(
                "\\",
                "/",
            ),
        },
    }

    TRAINING_METRICS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final report
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "             STEP 80 TRAINING RESULT"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Best epoch:          "
        f"{best_epoch}"
    )

    print()

    print(
        "Best validation:"
    )

    print(
        f"  Loss:              "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"  Top-1:             "
        f"{best_validation_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Top-2:             "
        f"{best_validation_metrics['top2_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Top-3:             "
        f"{best_validation_metrics['top3_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{best_validation_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy: "
        f"{best_validation_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Development test:"
    )

    print(
        f"  Loss:              "
        f"{dev_test_loss:.6f}"
    )

    print(
        f"  Top-1:             "
        f"{dev_test_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Top-2:             "
        f"{dev_test_metrics['top2_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Top-3:             "
        f"{dev_test_metrics['top3_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{dev_test_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Semantic accuracy: "
        f"{dev_test_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  "
        +
        str(
            BEST_MODEL_FILE.relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  "
        +
        str(
            SCALER_FILE.relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  "
        +
        str(
            TRAINING_METRICS_FILE.relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  "
        +
        str(
            FROZEN_CONFIG_FILE.relative_to(
                PROJECT_ROOT
            )
        )
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 80 COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "The Phase-3 PyFly adapter has now been "
        "trained with backpropagation on the GPU."
    )

    print()

    print(
        "Next: Step 81 - plug this trained adapter "
        "into true closed-loop Python generation."
    )


if __name__ == "__main__":
    main()
