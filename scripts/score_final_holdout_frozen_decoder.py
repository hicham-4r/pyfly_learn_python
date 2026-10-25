from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ============================================================
# STEP 77D — FINAL TEACHER-FORCED SCORING
#
# This script ONLY scores the already-frozen Step-62 decoder.
# It performs no fitting, search, or model selection.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = PROJECT_ROOT / "artifacts/features/generator_final_holdout.npz"
MODEL_FILE = PROJECT_ROOT / "artifacts/models/generator_final_frozen_decoder.joblib"
PROTOCOL_FILE = PROJECT_ROOT / "artifacts/metrics/final_evaluation_protocol.json"
REFIT_MANIFEST_FILE = PROJECT_ROOT / "artifacts/metrics/final_decoder_refit_manifest.json"
HOLDOUT_FILE = PROJECT_ROOT / "data/raw/final_holdout.jsonl"
GENERATOR_FILE = PROJECT_ROOT / "data/processed/generator_final_holdout.jsonl"
MAP_FILE = PROJECT_ROOT / "data/mappings/generator_dual_channel_map.npz"

OUTPUT_METRICS = PROJECT_ROOT / "artifacts/metrics/final_holdout_teacher_forced_metrics.json"
OUTPUT_PREDICTIONS = PROJECT_ROOT / "artifacts/metrics/final_holdout_teacher_forced_predictions.npz"

EXPECTED_FAMILY = "Balanced Logistic"
EXPECTED_CONFIG = "Balanced Logistic PCA=100 C=0.1"
EXPECTED_PCA = 100
EXPECTED_C = 0.1
EXPECTED_PROMPTS = 24
EXPECTED_TF_ROWS = 152
EXPECTED_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_ROWS = EXPECTED_TF_ROWS * len(EXPECTED_SEEDS)
EXPECTED_INPUT_NEURONS = 3328
EXPECTED_READOUT_NEURONS = 5873

FEATURE_BLOCKS = [
    "prompt_mean",
    "prompt_max",
    "prompt_final",
    "feedback_mean",
    "feedback_max",
    "feedback_final",
    "delayed_state",
]
EXPECTED_FEATURES = EXPECTED_READOUT_NEURONS * len(FEATURE_BLOCKS)
SEMANTIC_POSITIONS = [0, 2, 3, 4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar value, got shape {arr.shape}")
    return str(arr.reshape(-1)[0])


def top_k_accuracy(scores, y_true, classes, k: int) -> float:
    k = min(k, len(classes))
    top = np.argsort(scores, axis=1)[:, -k:]
    hits = 0
    for i, target in enumerate(y_true):
        candidates = {str(classes[j]) for j in top[i]}
        hits += str(target) in candidates
    return hits / len(y_true)


def mean_sd(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "values": [float(x) for x in arr],
    }


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def verify_protocol_and_model() -> tuple[dict, dict]:
    for path, label in [
        (PROTOCOL_FILE, "Step-77A protocol"),
        (REFIT_MANIFEST_FILE, "Step-77B refit manifest"),
        (MODEL_FILE, "frozen final decoder"),
        (FEATURES_FILE, "Step-77C final features"),
        (HOLDOUT_FILE, "final holdout"),
        (GENERATOR_FILE, "final generator dataset"),
        (MAP_FILE, "frozen neural map"),
    ]:
        require(path, label)

    protocol = load_json(PROTOCOL_FILE)
    manifest = load_json(REFIT_MANIFEST_FILE)

    if protocol.get("status") != "FROZEN_BEFORE_FINAL_EVALUATION":
        raise ValueError("Final evaluation protocol is not frozen.")
    if manifest.get("status") != "FROZEN_DECODER_REFIT_COMPLETE":
        raise ValueError("Step-77B refit manifest is not complete.")

    selection = protocol.get("decoder_selection", {})
    if selection.get("selected_family") != EXPECTED_FAMILY:
        raise ValueError("Frozen decoder family changed.")
    if selection.get("selected_configuration") != EXPECTED_CONFIG:
        raise ValueError("Frozen decoder configuration changed.")
    if manifest.get("selected_family") != EXPECTED_FAMILY:
        raise ValueError("Refit manifest decoder family changed.")
    if manifest.get("selected_configuration") != EXPECTED_CONFIG:
        raise ValueError("Refit manifest decoder configuration changed.")

    if sha256(MODEL_FILE) != manifest.get("model_sha256"):
        raise ValueError("Frozen model hash changed after Step 77B.")
    if sha256(PROTOCOL_FILE) != manifest.get("protocol_sha256"):
        raise ValueError("Protocol hash changed after Step 77B.")
    if sha256(HOLDOUT_FILE) != manifest.get("final_holdout_sha256"):
        raise ValueError("Final holdout hash changed after Step 77B.")

    return protocol, manifest


def load_final_features() -> dict:
    data = np.load(FEATURES_FILE, allow_pickle=True)
    required = {
        "X", "targets", "sample_ids", "example_ids", "intents",
        "target_positions", "feedback_tokens", "seeds", "input_neurons",
        "readout_population", "feature_block_names", "block_size",
        "trace_tau", "amplitude", "mapping_seed", "final_holdout_sha256",
        "final_generator_sha256", "frozen_mapping_sha256", "protocol_sha256",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Final feature file missing: {sorted(missing)}")

    out = {
        "X": data["X"].astype(np.float32),
        "targets": data["targets"].astype(str),
        "sample_ids": data["sample_ids"].astype(str),
        "example_ids": data["example_ids"].astype(str),
        "intents": data["intents"].astype(str),
        "positions": data["target_positions"].astype(np.int32),
        "feedback": data["feedback_tokens"].astype(str),
        "seeds": data["seeds"].astype(np.int32),
        "input_neurons": data["input_neurons"].astype(np.int64),
        "readout_population": data["readout_population"].astype(np.int64),
        "feature_blocks": data["feature_block_names"].astype(str),
        "block_size": int(np.asarray(data["block_size"]).item()),
        "holdout_hash": scalar_str(data["final_holdout_sha256"]),
        "generator_hash": scalar_str(data["final_generator_sha256"]),
        "mapping_hash": scalar_str(data["frozen_mapping_sha256"]),
        "protocol_hash": scalar_str(data["protocol_sha256"]),
    }

    X = out["X"]
    if X.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(
            f"Unexpected final feature shape {X.shape}; expected "
            f"({EXPECTED_ROWS}, {EXPECTED_FEATURES})."
        )
    if not np.isfinite(X).all():
        raise ValueError("Final features contain non-finite values.")
    if out["feature_blocks"].tolist() != FEATURE_BLOCKS:
        raise ValueError("Step-62 feature-block definition changed.")
    if out["block_size"] != EXPECTED_READOUT_NEURONS:
        raise ValueError("Frozen readout block size changed.")
    if len(out["input_neurons"]) != EXPECTED_INPUT_NEURONS:
        raise ValueError("Direct-input neuron count changed.")
    if len(out["readout_population"]) != EXPECTED_READOUT_NEURONS:
        raise ValueError("Indirect readout population changed.")
    if len(np.intersect1d(out["input_neurons"], out["readout_population"])):
        raise ValueError("Directly stimulated neurons leaked into readout.")

    if sorted(np.unique(out["seeds"]).tolist()) != EXPECTED_SEEDS:
        raise ValueError("Final neural seed set changed.")
    if len(np.unique(out["example_ids"])) != EXPECTED_PROMPTS:
        raise ValueError("Expected 24 final logical prompts.")

    samples, counts = np.unique(out["sample_ids"], return_counts=True)
    if len(samples) != EXPECTED_TF_ROWS:
        raise ValueError("Expected 152 final teacher-forcing samples.")
    if not np.all(counts == len(EXPECTED_SEEDS)):
        raise ValueError("Each teacher-forcing sample must have five replicas.")

    for key in ["targets", "sample_ids", "example_ids", "intents", "positions", "feedback", "seeds"]:
        if len(out[key]) != EXPECTED_ROWS:
            raise ValueError(f"Metadata row mismatch: {key}")

    if out["holdout_hash"] != sha256(HOLDOUT_FILE):
        raise ValueError("Final feature holdout hash mismatch.")
    if out["generator_hash"] != sha256(GENERATOR_FILE):
        raise ValueError("Final feature generator hash mismatch.")
    if out["mapping_hash"] != sha256(MAP_FILE):
        raise ValueError("Final feature neural-map hash mismatch.")
    if out["protocol_hash"] != sha256(PROTOCOL_FILE):
        raise ValueError("Final feature protocol hash mismatch.")

    return out


def load_frozen_model():
    model = joblib.load(MODEL_FILE)
    if not hasattr(model, "named_steps"):
        raise ValueError("Frozen decoder is not the expected sklearn Pipeline.")

    for step in ["scale", "pca", "classifier"]:
        if step not in model.named_steps:
            raise ValueError(f"Frozen decoder missing pipeline step: {step}")

    pca = model.named_steps["pca"]
    clf = model.named_steps["classifier"]

    if int(pca.n_components) != EXPECTED_PCA:
        raise ValueError("Frozen PCA component count changed.")
    if not np.isclose(float(clf.C), EXPECTED_C):
        raise ValueError("Frozen LogisticRegression C changed.")
    if clf.class_weight != "balanced":
        raise ValueError("Frozen classifier class_weight changed.")
    if clf.solver != "lbfgs":
        raise ValueError("Frozen classifier solver changed.")

    return model


def main() -> None:
    print()
    print("====================================================")
    print("   PyFly STEP 77D - FINAL TEACHER-FORCED SCORING")
    print("====================================================")
    print()

    _, manifest = verify_protocol_and_model()
    final = load_final_features()
    model = load_frozen_model()

    print("Frozen protocol integrity:     PASS")
    print("Frozen decoder integrity:      PASS")
    print("Final feature integrity:       PASS")
    print("Final holdout/map hashes:      PASS")
    print()
    print(f"Frozen configuration:          {EXPECTED_CONFIG}")
    print(f"Final logical prompts:         {len(np.unique(final['example_ids']))}")
    print(f"Teacher-forcing rows:          {len(np.unique(final['sample_ids']))}")
    print(f"Neural rows:                   {len(final['X'])}")
    print(f"Feature dimensions:            {final['X'].shape[1]:,}")
    print()
    print("Scoring frozen decoder...")

    X = final["X"]
    y_true = final["targets"]
    y_pred = model.predict(X).astype(str)

    clf = model.named_steps["classifier"]
    classes = np.asarray(clf.classes_).astype(str)
    scores = model.predict_proba(X)

    unknown = sorted(set(y_true.tolist()) - set(classes.tolist()))
    if unknown:
        raise ValueError(f"Unknown final target classes: {unknown}")

    top1 = float(accuracy_score(y_true, y_pred))
    top2 = float(top_k_accuracy(scores, y_true, classes, 2))
    top3 = float(top_k_accuracy(scores, y_true, classes, 3))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    semantic_mask = np.isin(final["positions"], SEMANTIC_POSITIONS)
    semantic_accuracy = float(accuracy_score(y_true[semantic_mask], y_pred[semantic_mask]))
    semantic_macro_f1 = float(
        f1_score(
            y_true[semantic_mask],
            y_pred[semantic_mask],
            average="macro",
            zero_division=0,
        )
    )

    position_metrics = {}
    for position in sorted(np.unique(final["positions"]).tolist()):
        mask = final["positions"] == position
        position_metrics[str(int(position))] = {
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "rows": int(mask.sum()),
        }

    token_metrics = {}
    for token in sorted(np.unique(y_true).tolist()):
        mask = y_true == token
        token_metrics[token] = {
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "rows": int(mask.sum()),
        }

    intent_metrics = {}
    for intent in sorted(np.unique(final["intents"]).tolist()):
        mask = final["intents"] == intent
        intent_metrics[intent] = {
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "macro_f1": float(
                f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            ),
            "rows": int(mask.sum()),
        }

    per_seed = {}
    seed_top1, seed_top2, seed_top3 = [], [], []
    seed_f1, seed_sem, seed_sem_f1 = [], [], []

    for seed in EXPECTED_SEEDS:
        mask = final["seeds"] == seed
        yt, yp, sc = y_true[mask], y_pred[mask], scores[mask]
        pos = final["positions"][mask]
        sem_mask = np.isin(pos, SEMANTIC_POSITIONS)

        if len(yt) != EXPECTED_TF_ROWS:
            raise ValueError(f"Seed {seed} should have 152 rows, got {len(yt)}.")

        metrics = {
            "rows": int(len(yt)),
            "top1_accuracy": float(accuracy_score(yt, yp)),
            "top2_accuracy": float(top_k_accuracy(sc, yt, classes, 2)),
            "top3_accuracy": float(top_k_accuracy(sc, yt, classes, 3)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "semantic_accuracy": float(accuracy_score(yt[sem_mask], yp[sem_mask])),
            "semantic_macro_f1": float(
                f1_score(yt[sem_mask], yp[sem_mask], average="macro", zero_division=0)
            ),
        }
        per_seed[str(seed)] = metrics
        seed_top1.append(metrics["top1_accuracy"])
        seed_top2.append(metrics["top2_accuracy"])
        seed_top3.append(metrics["top3_accuracy"])
        seed_f1.append(metrics["macro_f1"])
        seed_sem.append(metrics["semantic_accuracy"])
        seed_sem_f1.append(metrics["semantic_macro_f1"])

    uncertainty = {
        "top1_accuracy": mean_sd(seed_top1),
        "top2_accuracy": mean_sd(seed_top2),
        "top3_accuracy": mean_sd(seed_top3),
        "macro_f1": mean_sd(seed_f1),
        "semantic_accuracy": mean_sd(seed_sem),
        "semantic_macro_f1": mean_sd(seed_sem_f1),
    }

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    dev_distribution = manifest.get("development_refit", {}).get("target_distribution", {})
    if not dev_distribution:
        raise ValueError("Step-77B manifest lacks target distribution.")
    majority_token = max(dev_distribution, key=dev_distribution.get)
    majority_accuracy = float(
        accuracy_score(y_true, np.full(len(y_true), majority_token, dtype=object))
    )
    chance = 1.0 / len(classes)

    history = manifest.get("development_history", {})
    old_top1 = float(history["development_diagnostic_test_top1"])
    old_f1 = float(history["development_diagnostic_test_macro_f1"])
    old_sem = float(history["development_diagnostic_semantic_accuracy"])

    print()
    print("====================================================")
    print("          FINAL HOLDOUT - AGGREGATE RESULTS")
    print("====================================================")
    print()
    print(f"Uniform chance:               {chance * 100:6.2f}%")
    print(f"Development-majority ({majority_token}):      {majority_accuracy * 100:6.2f}%")
    print()
    print(f"FINAL Top-1:                  {top1 * 100:6.2f}%")
    print(f"FINAL Top-2:                  {top2 * 100:6.2f}%")
    print(f"FINAL Top-3:                  {top3 * 100:6.2f}%")
    print(f"FINAL Macro F1:               {macro_f1 * 100:6.2f}%")
    print(f"FINAL semantic accuracy:      {semantic_accuracy * 100:6.2f}%")
    print(f"FINAL semantic macro F1:      {semantic_macro_f1 * 100:6.2f}%")
    print()
    print("Mean +/- SD across neural seeds:")
    print(
        f"  Top-1:                    {uncertainty['top1_accuracy']['mean'] * 100:6.2f}% "
        f"+/- {uncertainty['top1_accuracy']['sd'] * 100:5.2f} pp"
    )
    print(
        f"  Macro F1:                 {uncertainty['macro_f1']['mean'] * 100:6.2f}% "
        f"+/- {uncertainty['macro_f1']['sd'] * 100:5.2f} pp"
    )
    print(
        f"  Semantic accuracy:        {uncertainty['semantic_accuracy']['mean'] * 100:6.2f}% "
        f"+/- {uncertainty['semantic_accuracy']['sd'] * 100:5.2f} pp"
    )

    print()
    print("Accuracy by output position:")
    for position, info in position_metrics.items():
        print(f"  P{position}: {info['accuracy'] * 100:6.2f}% ({info['rows']} rows)")

    print()
    print("Accuracy by target token:")
    for token, info in token_metrics.items():
        print(f"  {token:12} {info['accuracy'] * 100:6.2f}% ({info['rows']} rows)")

    print()
    print("Teacher-forced accuracy by intent:")
    for intent, info in intent_metrics.items():
        print(f"  {intent:12} {info['accuracy'] * 100:6.2f}% ({info['rows']} rows)")

    print()
    print("Per-seed Top-1 / semantic / Macro F1:")
    for seed in EXPECTED_SEEDS:
        m = per_seed[str(seed)]
        print(
            f"  seed={seed}: Top-1={m['top1_accuracy'] * 100:6.2f}%  "
            f"Semantic={m['semantic_accuracy'] * 100:6.2f}%  "
            f"MacroF1={m['macro_f1'] * 100:6.2f}%"
        )

    print()
    print("Development diagnostics (context only, NOT final):")
    print(f"  Top-1:                     {old_top1 * 100:6.2f}%")
    print(f"  Macro F1:                  {old_f1 * 100:6.2f}%")
    print(f"  Semantic accuracy:         {old_sem * 100:6.2f}%")
    print()
    print("Final minus development diagnostic:")
    print(f"  Top-1:                     {(top1 - old_top1) * 100:+6.2f} pp")
    print(f"  Macro F1:                  {(macro_f1 - old_f1) * 100:+6.2f} pp")
    print(f"  Semantic accuracy:         {(semantic_accuracy - old_sem) * 100:+6.2f} pp")

    print()
    print("Class order:")
    print(classes.tolist())
    print()
    print("Confusion matrix:")
    print(cm)

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PREDICTIONS,
        targets=y_true,
        predictions=y_pred,
        scores=scores.astype(np.float32),
        classes=classes,
        sample_ids=final["sample_ids"],
        example_ids=final["example_ids"],
        intents=final["intents"],
        target_positions=final["positions"],
        feedback_tokens=final["feedback"],
        seeds=final["seeds"],
    )

    metrics = {
        "step": "77D",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FINAL_TEACHER_FORCED_SCORE_COMPLETE",
        "evaluation_role": "untouched_final_holdout",
        "frozen_model": {
            "family": EXPECTED_FAMILY,
            "configuration": EXPECTED_CONFIG,
            "file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
            "sha256": sha256(MODEL_FILE),
        },
        "integrity": {
            "protocol_sha256": sha256(PROTOCOL_FILE),
            "final_features_sha256": sha256(FEATURES_FILE),
            "final_holdout_sha256": sha256(HOLDOUT_FILE),
            "final_generator_sha256": sha256(GENERATOR_FILE),
            "frozen_mapping_sha256": sha256(MAP_FILE),
        },
        "dataset": {
            "logical_prompts": int(len(np.unique(final["example_ids"]))),
            "teacher_forcing_rows": int(len(np.unique(final["sample_ids"]))),
            "neural_rows": int(len(y_true)),
            "feature_dimensions": int(X.shape[1]),
            "neural_seeds": EXPECTED_SEEDS,
        },
        "baselines": {
            "uniform_chance": chance,
            "development_majority_token": majority_token,
            "development_majority_accuracy_on_final": majority_accuracy,
        },
        "aggregate": {
            "top1_accuracy": top1,
            "top2_accuracy": top2,
            "top3_accuracy": top3,
            "macro_f1": macro_f1,
            "semantic_positions": SEMANTIC_POSITIONS,
            "semantic_accuracy": semantic_accuracy,
            "semantic_macro_f1": semantic_macro_f1,
        },
        "uncertainty_across_neural_seeds": uncertainty,
        "per_seed": per_seed,
        "accuracy_by_position": position_metrics,
        "accuracy_by_token": token_metrics,
        "teacher_forced_metrics_by_intent": intent_metrics,
        "classification_report": report,
        "confusion_matrix": {
            "class_order": classes.tolist(),
            "matrix": cm.astype(int).tolist(),
        },
        "development_diagnostics_for_context_only": {
            "top1_accuracy": old_top1,
            "macro_f1": old_f1,
            "semantic_accuracy": old_sem,
            "final_minus_development_top1_pp": (top1 - old_top1) * 100,
            "final_minus_development_macro_f1_pp": (macro_f1 - old_f1) * 100,
            "final_minus_development_semantic_pp": (semantic_accuracy - old_sem) * 100,
        },
        "rules": [
            "No fitting was performed in Step 77D.",
            "No architecture or hyperparameter search was performed in Step 77D.",
            "Do not retune from final-holdout results.",
            "Closed-loop exact-program/syntax metrics are evaluated separately.",
        ],
    }

    OUTPUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("Saved:")
    print(f"  {OUTPUT_METRICS.relative_to(PROJECT_ROOT)}")
    print(f"  {OUTPUT_PREDICTIONS.relative_to(PROJECT_ROOT)}")
    print()
    print("====================================================")
    print("                 STEP 77D COMPLETE")
    print("====================================================")
    print()
    print(
        "This is the first teacher-forced score on the frozen final holdout. "
        "Do NOT retune from these results."
    )
    print()
    print(
        "Next: Step 77E - frozen closed-loop generation on the same 24 final prompts."
    )


if __name__ == "__main__":
    main()
