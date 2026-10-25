from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pyfly.dataset import load_dataset
from pyfly.generator_dataset import load_generator_jsonl
from pyfly.tokenizer import build_code_vocabulary, code_to_tokens, tokens_to_code


# ============================================================
# Step 78A
# ENCODER-ONLY FINAL CONTROL
#
# Scientific question:
#
#     Does MaleCNS add useful computation beyond a simple
#     position-aware representation of the exact stimulus
#     history that was presented to the reservoir?
#
# This control uses:
#
#     prompt token + prompt position
#     full feedback-token history + feedback position
#
# It uses NO FlyBrain neurons and NO recurrent dynamics.
#
# Hyperparameters are selected ONLY on the old development
# validation split. After selection, the winning control is
# refit on all 48 development prompts and scored once on the
# already-frozen final holdout.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Development generator data
# ============================================================

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_train.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_validation.jsonl"
)

DEV_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generator_test.jsonl"
)


# ============================================================
# Frozen final holdout
# ============================================================

FINAL_LOGICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_holdout.jsonl"
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


# ============================================================
# MaleCNS final results, for comparison only
# ============================================================

MALECNS_TEACHER_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_teacher_forced_metrics.json"
)

MALECNS_CLOSED_LOOP_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "final_holdout_closed_loop_metrics.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "encoder_only_final_control.joblib"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "encoder_only_final_control.json"
)


# ============================================================
# Frozen task constants
# ============================================================

MAX_PROMPT_POSITIONS = 6

SEMANTIC_POSITIONS = [
    0,
    2,
    3,
    4,
]


# ============================================================
# Candidate grid
#
# Same six linear decoder families used in PyFly.
#
# For encoder-only features, "no PCA" is ALSO included.
# This intentionally makes the control strong rather than
# artificially handicapping it.
# ============================================================

PCA_OPTIONS = [
    None,
    10,
    20,
    40,
    60,
    100,
]

RIDGE_ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

LOGISTIC_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

SVM_C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]


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


def find_frozen_hash(
    protocol: dict,
    suffix: str,
) -> str:

    suffix = (
        suffix.replace(
            "\\",
            "/",
        )
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

        if normalized.endswith(
            suffix
        ):

            value = info.get(
                "sha256"
            )

            if value:
                return str(
                    value
                )

    raise ValueError(
        "Frozen protocol is missing "
        f"SHA-256 for {suffix}"
    )


# ============================================================
# Verify final-holdout integrity
# ============================================================

def verify_final_protocol() -> dict:

    required = [
        PROTOCOL_FILE,
        FINAL_LOGICAL_FILE,
        FINAL_GENERATOR_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{path}"
            )

    protocol = load_json(
        PROTOCOL_FILE
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_FINAL_EVALUATION"
    ):

        raise ValueError(
            "Final evaluation protocol "
            "is no longer marked frozen."
        )

    expected_hash = (
        find_frozen_hash(
            protocol,
            "data/raw/final_holdout.jsonl",
        )
    )

    current_hash = (
        sha256_file(
            FINAL_LOGICAL_FILE
        )
    )

    if (
        current_hash
        !=
        expected_hash
    ):

        raise ValueError(
            "Final holdout hash changed "
            "after Step 77A."
        )

    return protocol


# ============================================================
# Load samples
# ============================================================

def load_all_data():

    train = load_generator_jsonl(
        TRAIN_FILE
    )

    validation = load_generator_jsonl(
        VALIDATION_FILE
    )

    dev_test = load_generator_jsonl(
        DEV_TEST_FILE
    )

    final_generator = load_generator_jsonl(
        FINAL_GENERATOR_FILE
    )

    final_logical = load_dataset(
        FINAL_LOGICAL_FILE
    )

    if len(
        final_logical
    ) != 24:

        raise ValueError(
            "Expected 24 frozen final "
            f"logical prompts, got "
            f"{len(final_logical)}"
        )

    if len(
        final_generator
    ) != 152:

        raise ValueError(
            "Expected 152 frozen final "
            f"teacher-forcing rows, got "
            f"{len(final_generator)}"
        )

    return (
        train,
        validation,
        dev_test,
        final_generator,
        final_logical,
    )


# ============================================================
# Build encoder-only feature schema
# ============================================================

def build_feature_schema(
    development_samples,
):

    prompt_tokens = sorted(
        {
            token

            for sample
            in development_samples

            for token
            in sample.prompt_tokens
        }
    )

    prompt_to_id = {
        token: index

        for index, token
        in enumerate(
            prompt_tokens
        )
    }

    code_vocab = (
        build_code_vocabulary()
    )

    feedback_tokens = list(
        code_vocab.token_to_id.keys()
    )

    feedback_to_id = {
        token: index

        for index, token
        in enumerate(
            feedback_tokens
        )
    }

    max_feedback_positions = max(
        len(
            sample.context_tokens
        )

        for sample
        in development_samples
    )

    prompt_dimensions = (
        MAX_PROMPT_POSITIONS
        *
        len(
            prompt_tokens
        )
    )

    feedback_dimensions = (
        max_feedback_positions
        *
        len(
            feedback_tokens
        )
    )

    total_dimensions = (
        prompt_dimensions
        +
        feedback_dimensions
    )

    return {
        "prompt_tokens": (
            prompt_tokens
        ),

        "prompt_to_id": (
            prompt_to_id
        ),

        "feedback_tokens": (
            feedback_tokens
        ),

        "feedback_to_id": (
            feedback_to_id
        ),

        "max_feedback_positions": int(
            max_feedback_positions
        ),

        "prompt_dimensions": int(
            prompt_dimensions
        ),

        "feedback_dimensions": int(
            feedback_dimensions
        ),

        "total_dimensions": int(
            total_dimensions
        ),
    }


# ============================================================
# Encode one direct-stimulus history
# ============================================================

def encode_stimulus_history(
    prompt_tokens: list[str],
    feedback_history: list[str],
    schema: dict,
) -> np.ndarray:

    if len(
        prompt_tokens
    ) > MAX_PROMPT_POSITIONS:

        raise ValueError(
            "Prompt exceeds frozen "
            "maximum length."
        )

    if len(
        feedback_history
    ) > schema[
        "max_feedback_positions"
    ]:

        raise ValueError(
            "Feedback history exceeds "
            "development-supported length."
        )

    vector = np.zeros(
        schema[
            "total_dimensions"
        ],
        dtype=np.float32,
    )

    prompt_vocab_size = len(
        schema[
            "prompt_tokens"
        ]
    )

    feedback_vocab_size = len(
        schema[
            "feedback_tokens"
        ]
    )

    feedback_offset = schema[
        "prompt_dimensions"
    ]

    # --------------------------------------------------------
    # Position-aware prompt
    # --------------------------------------------------------

    for position, token in enumerate(
        prompt_tokens
    ):

        if token not in schema[
            "prompt_to_id"
        ]:

            raise KeyError(
                "Final prompt contains "
                f"unknown token: {token}"
            )

        token_index = schema[
            "prompt_to_id"
        ][
            token
        ]

        index = (
            position
            *
            prompt_vocab_size
            +
            token_index
        )

        vector[
            index
        ] = 1.0

    # --------------------------------------------------------
    # Position-aware full feedback history
    # --------------------------------------------------------

    for position, token in enumerate(
        feedback_history
    ):

        if token not in schema[
            "feedback_to_id"
        ]:

            raise KeyError(
                "Unknown feedback token: "
                f"{token}"
            )

        token_index = schema[
            "feedback_to_id"
        ][
            token
        ]

        index = (
            feedback_offset
            +
            position
            *
            feedback_vocab_size
            +
            token_index
        )

        vector[
            index
        ] = 1.0

    return vector


# ============================================================
# Convert GeneratorExample list -> matrix
# ============================================================

def samples_to_matrix(
    samples,
    schema: dict,
):

    X = np.stack(
        [
            encode_stimulus_history(
                sample.prompt_tokens,
                sample.context_tokens,
                schema,
            )

            for sample
            in samples
        ]
    )

    y = np.asarray(
        [
            sample.target_token

            for sample
            in samples
        ]
    )

    positions = np.asarray(
        [
            sample.target_position

            for sample
            in samples
        ],
        dtype=np.int32,
    )

    intents = np.asarray(
        [
            sample.intent

            for sample
            in samples
        ]
    )

    example_ids = np.asarray(
        [
            sample.example_id

            for sample
            in samples
        ]
    )

    return {
        "X": X,
        "y": y,
        "positions": positions,
        "intents": intents,
        "example_ids": example_ids,
    }


# ============================================================
# Pipeline builder
# ============================================================

def make_pipeline(
    pca_components,
    classifier,
) -> Pipeline:

    steps = [
        (
            "scale",
            StandardScaler(),
        )
    ]

    if pca_components is not None:

        steps.append(
            (
                "pca",

                PCA(
                    n_components=(
                        pca_components
                    ),

                    svd_solver=(
                        "randomized"
                    ),

                    random_state=42,
                ),
            )
        )

    steps.append(
        (
            "classifier",
            classifier,
        )
    )

    return Pipeline(
        steps
    )


# ============================================================
# Candidate builder
# ============================================================

def build_candidates():

    candidates = []

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    for pca_components in PCA_OPTIONS:

        for alpha in RIDGE_ALPHAS:

            candidates.append(
                {
                    "family": "Ridge",

                    "name": (
                        f"Ridge "
                        f"PCA={pca_components} "
                        f"alpha={alpha}"
                    ),

                    "pca": (
                        pca_components
                    ),

                    "parameter": float(
                        alpha
                    ),

                    "model": (
                        make_pipeline(
                            pca_components,

                            RidgeClassifier(
                                alpha=alpha,
                            ),
                        )
                    ),
                }
            )

    # --------------------------------------------------------
    # Balanced Ridge
    # --------------------------------------------------------

    for pca_components in PCA_OPTIONS:

        for alpha in RIDGE_ALPHAS:

            candidates.append(
                {
                    "family": (
                        "Balanced Ridge"
                    ),

                    "name": (
                        f"Balanced Ridge "
                        f"PCA={pca_components} "
                        f"alpha={alpha}"
                    ),

                    "pca": (
                        pca_components
                    ),

                    "parameter": float(
                        alpha
                    ),

                    "model": (
                        make_pipeline(
                            pca_components,

                            RidgeClassifier(
                                alpha=alpha,

                                class_weight=(
                                    "balanced"
                                ),
                            ),
                        )
                    ),
                }
            )

    # --------------------------------------------------------
    # Logistic
    # --------------------------------------------------------

    for pca_components in PCA_OPTIONS:

        for c_value in LOGISTIC_C_VALUES:

            candidates.append(
                {
                    "family": "Logistic",

                    "name": (
                        f"Logistic "
                        f"PCA={pca_components} "
                        f"C={c_value}"
                    ),

                    "pca": (
                        pca_components
                    ),

                    "parameter": float(
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            pca_components,

                            LogisticRegression(
                                C=c_value,

                                solver="lbfgs",

                                max_iter=5000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )

    # --------------------------------------------------------
    # Balanced Logistic
    # --------------------------------------------------------

    for pca_components in PCA_OPTIONS:

        for c_value in LOGISTIC_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Balanced Logistic"
                    ),

                    "name": (
                        f"Balanced Logistic "
                        f"PCA={pca_components} "
                        f"C={c_value}"
                    ),

                    "pca": (
                        pca_components
                    ),

                    "parameter": float(
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            pca_components,

                            LogisticRegression(
                                C=c_value,

                                solver="lbfgs",

                                class_weight=(
                                    "balanced"
                                ),

                                max_iter=5000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    for pca_components in PCA_OPTIONS:

        for c_value in SVM_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Linear SVM"
                    ),

                    "name": (
                        f"Linear SVM "
                        f"PCA={pca_components} "
                        f"C={c_value}"
                    ),

                    "pca": (
                        pca_components
                    ),

                    "parameter": float(
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            pca_components,

                            LinearSVC(
                                C=c_value,

                                max_iter=10000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )

    # --------------------------------------------------------
    # Balanced Linear SVM
    # --------------------------------------------------------

    for pca_components in PCA_OPTIONS:

        for c_value in SVM_C_VALUES:

            candidates.append(
                {
                    "family": (
                        "Balanced Linear SVM"
                    ),

                    "name": (
                        f"Balanced Linear SVM "
                        f"PCA={pca_components} "
                        f"C={c_value}"
                    ),

                    "pca": (
                        pca_components
                    ),

                    "parameter": float(
                        c_value
                    ),

                    "model": (
                        make_pipeline(
                            pca_components,

                            LinearSVC(
                                C=c_value,

                                class_weight=(
                                    "balanced"
                                ),

                                max_iter=10000,

                                random_state=42,
                            ),
                        )
                    ),
                }
            )

    return candidates


# ============================================================
# Build a fresh model from selected candidate specification
# ============================================================

def rebuild_selected_model(
    selected: dict,
) -> Pipeline:

    family = selected[
        "family"
    ]

    pca_components = selected[
        "pca"
    ]

    parameter = selected[
        "parameter"
    ]

    if family == "Ridge":

        classifier = RidgeClassifier(
            alpha=parameter,
        )

    elif family == "Balanced Ridge":

        classifier = RidgeClassifier(
            alpha=parameter,
            class_weight="balanced",
        )

    elif family == "Logistic":

        classifier = LogisticRegression(
            C=parameter,
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

    elif family == "Balanced Logistic":

        classifier = LogisticRegression(
            C=parameter,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

    elif family == "Linear SVM":

        classifier = LinearSVC(
            C=parameter,
            max_iter=10000,
            random_state=42,
        )

    elif family == "Balanced Linear SVM":

        classifier = LinearSVC(
            C=parameter,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        )

    else:

        raise ValueError(
            f"Unknown selected family: "
            f"{family}"
        )

    return make_pipeline(
        pca_components,
        classifier,
    )


# ============================================================
# Score helpers
# ============================================================

def get_scores(
    model: Pipeline,
    X: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    classifier = model.named_steps[
        "classifier"
    ]

    classes = np.asarray(
        classifier.classes_
    ).astype(str)

    scores = np.asarray(
        model.decision_function(
            X
        )
    )

    if scores.ndim == 1:

        scores = scores.reshape(
            -1,
            1,
        )

    return (
        scores,
        classes,
    )


def top_k_accuracy(
    scores: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> float:

    k = min(
        k,
        len(
            classes
        ),
    )

    top_indices = np.argsort(
        scores,
        axis=1,
    )[
        :,
        -k:
    ]

    top_classes = classes[
        top_indices
    ]

    correct = np.asarray(
        [
            true_label
            in
            row.tolist()

            for true_label, row
            in zip(
                y_true,
                top_classes,
            )
        ],
        dtype=bool,
    )

    return float(
        np.mean(
            correct
        )
    )


# ============================================================
# Teacher-forced metrics
# ============================================================

def teacher_forced_metrics(
    model: Pipeline,
    data: dict,
) -> dict:

    X = data[
        "X"
    ]

    y = data[
        "y"
    ]

    positions = data[
        "positions"
    ]

    intents = data[
        "intents"
    ]

    predictions = model.predict(
        X
    ).astype(str)

    (
        scores,
        classes,
    ) = get_scores(
        model,
        X,
    )

    semantic_mask = np.isin(
        positions,
        SEMANTIC_POSITIONS,
    )

    position_metrics = {}

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

        position_metrics[
            str(
                int(
                    position
                )
            )
        ] = float(
            accuracy_score(
                y[
                    mask
                ],
                predictions[
                    mask
                ],
            )
        )

    token_metrics = {}

    for token in sorted(
        np.unique(
            y
        )
    ):

        mask = (
            y
            ==
            token
        )

        token_metrics[
            str(
                token
            )
        ] = float(
            accuracy_score(
                y[
                    mask
                ],
                predictions[
                    mask
                ],
            )
        )

    intent_metrics = {}

    for intent in sorted(
        np.unique(
            intents
        )
    ):

        mask = (
            intents
            ==
            intent
        )

        intent_metrics[
            str(
                intent
            )
        ] = float(
            accuracy_score(
                y[
                    mask
                ],
                predictions[
                    mask
                ],
            )
        )

    return {
        "top1_accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),

        "top2_accuracy": (
            top_k_accuracy(
                scores,
                y,
                classes,
                2,
            )
        ),

        "top3_accuracy": (
            top_k_accuracy(
                scores,
                y,
                classes,
                3,
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "semantic_accuracy": float(
            accuracy_score(
                y[
                    semantic_mask
                ],
                predictions[
                    semantic_mask
                ],
            )
        ),

        "semantic_macro_f1": float(
            f1_score(
                y[
                    semantic_mask
                ],
                predictions[
                    semantic_mask
                ],
                average="macro",
                zero_division=0,
            )
        ),

        "accuracy_by_position": (
            position_metrics
        ),

        "accuracy_by_token": (
            token_metrics
        ),

        "accuracy_by_intent": (
            intent_metrics
        ),

        "class_order": (
            classes.tolist()
        ),
    }


# ============================================================
# Closed-loop helpers
# ============================================================

def sequence_token_accuracy(
    target_tokens: list[str],
    generated_tokens: list[str],
) -> float:

    denominator = max(
        len(
            target_tokens
        ),
        len(
            generated_tokens
        ),
    )

    if denominator == 0:
        return 1.0

    correct = 0

    for index in range(
        denominator
    ):

        target = (
            target_tokens[
                index
            ]
            if index
            <
            len(
                target_tokens
            )
            else None
        )

        predicted = (
            generated_tokens[
                index
            ]
            if index
            <
            len(
                generated_tokens
            )
            else None
        )

        if target == predicted:
            correct += 1

    return float(
        correct
        /
        denominator
    )


def decode_generated(
    generated_tokens: list[str],
):

    try:

        generated_code = (
            tokens_to_code(
                generated_tokens
            )
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
            generated_code
        )

        syntax_valid = True

    except SyntaxError:

        syntax_valid = False

    return (
        generated_code,
        template_valid,
        syntax_valid,
    )


# ============================================================
# Closed-loop encoder-only generation
# ============================================================

def run_closed_loop(
    model: Pipeline,
    final_logical,
    schema: dict,
) -> dict:

    rows = []

    max_generation_steps = schema[
        "max_feedback_positions"
    ]

    for example in final_logical:

        target_tokens = (
            code_to_tokens(
                example.code
            )
        )

        context = [
            "BOS"
        ]

        generated_tokens = []

        for _ in range(
            max_generation_steps
        ):

            feature = (
                encode_stimulus_history(
                    example.prompt_tokens,
                    context,
                    schema,
                )
            )

            predicted = str(
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
                predicted
            )

            if predicted == "EOS":
                break

            if (
                len(
                    context
                )
                <
                schema[
                    "max_feedback_positions"
                ]
            ):

                context.append(
                    predicted
                )

        (
            generated_code,
            template_valid,
            syntax_valid,
        ) = decode_generated(
            generated_tokens
        )

        token_accuracy = (
            sequence_token_accuracy(
                target_tokens,
                generated_tokens,
            )
        )

        rows.append(
            {
                "example_id": (
                    example.id
                ),

                "intent": (
                    example.intent
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
                    generated_tokens
                    ==
                    target_tokens
                ),

                "valid_program": bool(
                    template_valid
                ),

                "syntax_valid": bool(
                    syntax_valid
                ),

                "eos_completed": bool(
                    generated_tokens
                    and
                    generated_tokens[
                        -1
                    ]
                    ==
                    "EOS"
                ),
            }
        )

    aggregate = {
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

        "syntax_valid_rate": float(
            np.mean(
                [
                    row[
                        "syntax_valid"
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

    per_intent = {}

    for intent in sorted(
        {
            row[
                "intent"
            ]
            for row
            in rows
        }
    ):

        intent_rows = [
            row

            for row
            in rows

            if row[
                "intent"
            ]
            ==
            intent
        ]

        per_intent[
            intent
        ] = {
            "prompts": int(
                len(
                    intent_rows
                )
            ),

            "mean_token_accuracy": float(
                np.mean(
                    [
                        row[
                            "token_accuracy"
                        ]
                        for row
                        in intent_rows
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
                        in intent_rows
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
                        in intent_rows
                    ]
                )
            ),
        }

    return {
        "aggregate": aggregate,
        "per_intent": per_intent,
        "predictions": rows,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()

    print(
        "===================================================="
    )

    print(
        "      PyFly STEP 78A - ENCODER-ONLY CONTROL"
    )

    print(
        "===================================================="
    )

    print()

    protocol = verify_final_protocol()

    print(
        "Frozen final-holdout integrity: PASS"
    )

    (
        train_samples,
        validation_samples,
        dev_test_samples,
        final_generator_samples,
        final_logical,
    ) = load_all_data()

    development_samples = (
        train_samples
        +
        validation_samples
        +
        dev_test_samples
    )

    schema = build_feature_schema(
        development_samples
    )

    print()

    print(
        f"Development logical prompts: "
        f"{len(set(sample.example_id for sample in development_samples))}"
    )

    print(
        f"Development teacher rows:    "
        f"{len(development_samples)}"
    )

    print(
        f"Prompt vocabulary:           "
        f"{len(schema['prompt_tokens'])}"
    )

    print(
        f"Feedback vocabulary:         "
        f"{len(schema['feedback_tokens'])}"
    )

    print(
        f"Prompt positions:            "
        f"{MAX_PROMPT_POSITIONS}"
    )

    print(
        f"Feedback-history positions:  "
        f"{schema['max_feedback_positions']}"
    )

    print(
        f"Encoder-only dimensions:     "
        f"{schema['total_dimensions']}"
    )

    # --------------------------------------------------------
    # Matrices
    # --------------------------------------------------------

    train = samples_to_matrix(
        train_samples,
        schema,
    )

    validation = samples_to_matrix(
        validation_samples,
        schema,
    )

    dev_test = samples_to_matrix(
        dev_test_samples,
        schema,
    )

    final_teacher = samples_to_matrix(
        final_generator_samples,
        schema,
    )

    # --------------------------------------------------------
    # Validation-only model selection
    # --------------------------------------------------------

    print()

    print(
        "===================================================="
    )

    print(
        "       ENCODER-ONLY VALIDATION MODEL SEARCH"
    )

    print(
        "===================================================="
    )

    print()

    candidates = build_candidates()

    best = None
    family_best = {}
    search_results = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        pca_components = candidate[
            "pca"
        ]

        if (
            pca_components is not None
            and
            pca_components
            >=
            min(
                train[
                    "X"
                ].shape
            )
        ):

            continue

        model = candidate[
            "model"
        ]

        model.fit(
            train[
                "X"
            ],
            train[
                "y"
            ],
        )

        predictions = model.predict(
            validation[
                "X"
            ]
        )

        validation_accuracy = float(
            accuracy_score(
                validation[
                    "y"
                ],
                predictions,
            )
        )

        validation_macro_f1 = float(
            f1_score(
                validation[
                    "y"
                ],
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        result = {
            "family": candidate[
                "family"
            ],

            "name": candidate[
                "name"
            ],

            "pca": candidate[
                "pca"
            ],

            "parameter": candidate[
                "parameter"
            ],

            "validation_accuracy": (
                validation_accuracy
            ),

            "validation_macro_f1": (
                validation_macro_f1
            ),
        }

        search_results.append(
            result
        )

        current_family_best = (
            family_best.get(
                candidate[
                    "family"
                ]
            )
        )

        if (
            current_family_best is None
            or
            validation_macro_f1
            >
            current_family_best[
                "validation_macro_f1"
            ]
            or (
                validation_macro_f1
                ==
                current_family_best[
                    "validation_macro_f1"
                ]
                and
                validation_accuracy
                >
                current_family_best[
                    "validation_accuracy"
                ]
            )
        ):

            family_best[
                candidate[
                    "family"
                ]
            ] = result

        if (
            best is None
            or
            validation_macro_f1
            >
            best[
                "validation_macro_f1"
            ]
            or (
                validation_macro_f1
                ==
                best[
                    "validation_macro_f1"
                ]
                and
                validation_accuracy
                >
                best[
                    "validation_accuracy"
                ]
            )
        ):

            best = result

        print(
            f"[{index:03}/{len(candidates):03}] "
            f"{candidate['family']:20} "
            f"PCA={str(pca_components):>4} "
            f"param="
            f"{candidate['parameter']:8.4f} "
            f"acc="
            f"{validation_accuracy * 100:6.2f}% "
            f"F1="
            f"{validation_macro_f1 * 100:6.2f}%"
        )

    if best is None:

        raise RuntimeError(
            "No encoder-only candidate "
            "could be selected."
        )

    print()

    print(
        "===================================================="
    )

    print(
        "       SELECTED ENCODER-ONLY CONTROL"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Family:               "
        f"{best['family']}"
    )

    print(
        f"Configuration:        "
        f"{best['name']}"
    )

    print(
        f"Validation accuracy:  "
        f"{best['validation_accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation macro F1:  "
        f"{best['validation_macro_f1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Optional old development-test diagnostic
    # --------------------------------------------------------

    selected_for_dev_test = (
        rebuild_selected_model(
            best
        )
    )

    X_train_val = np.concatenate(
        [
            train[
                "X"
            ],
            validation[
                "X"
            ],
        ],
        axis=0,
    )

    y_train_val = np.concatenate(
        [
            train[
                "y"
            ],
            validation[
                "y"
            ],
        ],
        axis=0,
    )

    selected_for_dev_test.fit(
        X_train_val,
        y_train_val,
    )

    dev_test_metrics = (
        teacher_forced_metrics(
            selected_for_dev_test,
            dev_test,
        )
    )

    # --------------------------------------------------------
    # Refit selected control on ALL 48 development prompts
    # --------------------------------------------------------

    final_model = (
        rebuild_selected_model(
            best
        )
    )

    all_dev_X = np.concatenate(
        [
            train[
                "X"
            ],
            validation[
                "X"
            ],
            dev_test[
                "X"
            ],
        ],
        axis=0,
    )

    all_dev_y = np.concatenate(
        [
            train[
                "y"
            ],
            validation[
                "y"
            ],
            dev_test[
                "y"
            ],
        ],
        axis=0,
    )

    final_model.fit(
        all_dev_X,
        all_dev_y,
    )

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        OUTPUT_MODEL,
    )

    # --------------------------------------------------------
    # Frozen final teacher-forced score
    # --------------------------------------------------------

    final_teacher_metrics = (
        teacher_forced_metrics(
            final_model,
            final_teacher,
        )
    )

    # --------------------------------------------------------
    # Frozen final closed-loop score
    # --------------------------------------------------------

    final_closed_loop = (
        run_closed_loop(
            final_model,
            final_logical,
            schema,
        )
    )

    # --------------------------------------------------------
    # MaleCNS comparison
    # --------------------------------------------------------

    malecns_teacher = None
    malecns_closed_loop = None

    if (
        MALECNS_TEACHER_METRICS_FILE.exists()
    ):

        malecns_teacher = (
            load_json(
                MALECNS_TEACHER_METRICS_FILE
            ).get(
                "aggregate"
            )
        )

    if (
        MALECNS_CLOSED_LOOP_METRICS_FILE.exists()
    ):

        malecns_closed_loop = (
            load_json(
                MALECNS_CLOSED_LOOP_METRICS_FILE
            ).get(
                "aggregate"
            )
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()

    print(
        "===================================================="
    )

    print(
        " ENCODER-ONLY FINAL TEACHER-FORCED RESULTS"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Top-1:              "
        f"{final_teacher_metrics['top1_accuracy'] * 100:.2f}%"
    )

    print(
        f"Top-2:              "
        f"{final_teacher_metrics['top2_accuracy'] * 100:.2f}%"
    )

    print(
        f"Top-3:              "
        f"{final_teacher_metrics['top3_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1:           "
        f"{final_teacher_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Semantic accuracy:  "
        f"{final_teacher_metrics['semantic_accuracy'] * 100:.2f}%"
    )

    print(
        f"Semantic macro F1:  "
        f"{final_teacher_metrics['semantic_macro_f1'] * 100:.2f}%"
    )

    print()

    print(
        "Accuracy by output position:"
    )

    for position, value in (
        final_teacher_metrics[
            "accuracy_by_position"
        ].items()
    ):

        print(
            f"  P{position}: "
            f"{value * 100:6.2f}%"
        )

    print()

    print(
        "===================================================="
    )

    print(
        "  ENCODER-ONLY FINAL CLOSED-LOOP RESULTS"
    )

    print(
        "===================================================="
    )

    print()

    closed = final_closed_loop[
        "aggregate"
    ]

    print(
        f"Mean token accuracy:     "
        f"{closed['mean_token_accuracy'] * 100:.2f}%"
    )

    print(
        f"Exact-program accuracy:  "
        f"{closed['exact_program_accuracy'] * 100:.2f}%"
    )

    print(
        f"Valid-program rate:      "
        f"{closed['valid_program_rate'] * 100:.2f}%"
    )

    print(
        f"Syntax-valid rate:       "
        f"{closed['syntax_valid_rate'] * 100:.2f}%"
    )

    print(
        f"EOS completion rate:     "
        f"{closed['eos_completion_rate'] * 100:.2f}%"
    )

    print()

    print(
        "Closed-loop by intent:"
    )

    for intent, row in (
        final_closed_loop[
            "per_intent"
        ].items()
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

    # --------------------------------------------------------
    # Direct comparison with MaleCNS
    # --------------------------------------------------------

    if (
        malecns_teacher is not None
        and
        malecns_closed_loop is not None
    ):

        print()

        print(
            "===================================================="
        )

        print(
            "        ENCODER-ONLY vs MaleCNS FINAL"
        )

        print(
            "===================================================="
        )

        print()

        print(
            "Teacher-forced:"
        )

        print(
            f"  Top-1:      "
            f"Encoder={final_teacher_metrics['top1_accuracy'] * 100:6.2f}%  "
            f"MaleCNS={malecns_teacher['top1_accuracy'] * 100:6.2f}%  "
            f"Delta="
            f"{(final_teacher_metrics['top1_accuracy'] - malecns_teacher['top1_accuracy']) * 100:+.2f} pp"
        )

        print(
            f"  Macro F1:   "
            f"Encoder={final_teacher_metrics['macro_f1'] * 100:6.2f}%  "
            f"MaleCNS={malecns_teacher['macro_f1'] * 100:6.2f}%  "
            f"Delta="
            f"{(final_teacher_metrics['macro_f1'] - malecns_teacher['macro_f1']) * 100:+.2f} pp"
        )

        print(
            f"  Semantic:   "
            f"Encoder={final_teacher_metrics['semantic_accuracy'] * 100:6.2f}%  "
            f"MaleCNS={malecns_teacher['semantic_accuracy'] * 100:6.2f}%  "
            f"Delta="
            f"{(final_teacher_metrics['semantic_accuracy'] - malecns_teacher['semantic_accuracy']) * 100:+.2f} pp"
        )

        print()

        print(
            "Closed-loop:"
        )

        print(
            f"  Token acc:  "
            f"Encoder={closed['mean_token_accuracy'] * 100:6.2f}%  "
            f"MaleCNS={malecns_closed_loop['mean_token_accuracy'] * 100:6.2f}%  "
            f"Delta="
            f"{(closed['mean_token_accuracy'] - malecns_closed_loop['mean_token_accuracy']) * 100:+.2f} pp"
        )

        print(
            f"  Exact:      "
            f"Encoder={closed['exact_program_accuracy'] * 100:6.2f}%  "
            f"MaleCNS={malecns_closed_loop['exact_program_accuracy'] * 100:6.2f}%  "
            f"Delta="
            f"{(closed['exact_program_accuracy'] - malecns_closed_loop['exact_program_accuracy']) * 100:+.2f} pp"
        )

        print(
            f"  Valid:      "
            f"Encoder={closed['valid_program_rate'] * 100:6.2f}%  "
            f"MaleCNS={malecns_closed_loop['valid_program_rate'] * 100:6.2f}%  "
            f"Delta="
            f"{(closed['valid_program_rate'] - malecns_closed_loop['valid_program_rate']) * 100:+.2f} pp"
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    result = {
        "step": "78A",

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "control": (
            "encoder_only_full_stimulus_history"
        ),

        "description": (
            "Position-aware prompt plus "
            "position-aware full feedback history; "
            "no FlyBrain reservoir."
        ),

        "feature_schema": {
            "prompt_vocabulary": (
                schema[
                    "prompt_tokens"
                ]
            ),

            "feedback_vocabulary": (
                schema[
                    "feedback_tokens"
                ]
            ),

            "max_prompt_positions": (
                MAX_PROMPT_POSITIONS
            ),

            "max_feedback_positions": (
                schema[
                    "max_feedback_positions"
                ]
            ),

            "feature_dimensions": (
                schema[
                    "total_dimensions"
                ]
            ),
        },

        "selection": {
            "selection_split": (
                "development validation only"
            ),

            "selection_metric": (
                "macro F1, accuracy tie-break"
            ),

            "selected_family": (
                best[
                    "family"
                ]
            ),

            "selected_configuration": (
                best[
                    "name"
                ]
            ),

            "validation_accuracy": (
                best[
                    "validation_accuracy"
                ]
            ),

            "validation_macro_f1": (
                best[
                    "validation_macro_f1"
                ]
            ),

            "family_best": (
                family_best
            ),
        },

        "development_test_diagnostic": (
            dev_test_metrics
        ),

        "final_teacher_forced": (
            final_teacher_metrics
        ),

        "final_closed_loop": (
            {
                "aggregate": (
                    final_closed_loop[
                        "aggregate"
                    ]
                ),

                "per_intent": (
                    final_closed_loop[
                        "per_intent"
                    ]
                ),

                "predictions": (
                    final_closed_loop[
                        "predictions"
                    ]
                ),
            }
        ),

        "malecns_final_comparison": {
            "teacher_forced": (
                malecns_teacher
            ),

            "closed_loop": (
                malecns_closed_loop
            ),
        },

        "integrity": {
            "final_holdout_sha256": (
                sha256_file(
                    FINAL_LOGICAL_FILE
                )
            ),

            "protocol_sha256": (
                sha256_file(
                    PROTOCOL_FILE
                )
            ),

            "model_sha256": (
                sha256_file(
                    OUTPUT_MODEL
                )
            ),
        },

        "rules": [
            (
                "No final-holdout result was used "
                "for model or hyperparameter selection."
            ),

            (
                "The encoder-only control uses no "
                "FlyBrain state or recurrent dynamics."
            ),

            (
                "The full directly presented stimulus "
                "history is explicitly represented."
            ),

            (
                "The no-PCA option is included to make "
                "the encoder-only control strong."
            ),
        ],
    }

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_METRICS.write_text(
        json.dumps(
            result,
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
        f"{OUTPUT_MODEL.relative_to(PROJECT_ROOT)}"
    )

    print(
        "  "
        f"{OUTPUT_METRICS.relative_to(PROJECT_ROOT)}"
    )

    print()

    print(
        "===================================================="
    )

    print(
        "                 STEP 78A COMPLETE"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Next control: matched random recurrent reservoir."
    )


if __name__ == "__main__":
    main()
