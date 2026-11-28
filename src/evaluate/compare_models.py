"""Evaluate all trained spam classification models and select the best one."""

from __future__ import annotations

import json
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

MODEL_SPECS = [
    {
        "key": "decision_tree",
        "name": "Decision Tree",
        "filename": "decision_tree.pkl",
    },
    {
        "key": "logistic_regression",
        "name": "Logistic Regression",
        "filename": "logistic_regression_spam_pipeline.pkl",
    },
    {
        "key": "svm",
        "name": "SVM (LinearSVC + Calibrated)",
        "filename": "svm_spam_pipeline.pkl",
    },
    {
        "key": "naive_bayes",
        "name": "Multinomial Naive Bayes",
        "filename": "naive_bayes.pkl",
    },
    {
        "key": "random_forest",
        "name": "Random Forest",
        "filename": "random_forest.pkl",
    },
]


def load_test_data() -> tuple[pd.Series, pd.Series]:
    test_path = DATA_DIR / "test.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    df = pd.read_csv(test_path)
    X_test = df["text"].fillna("").astype(str)
    y_test = df["label"].astype(int)
    return X_test, y_test


def get_spam_scores(model, X: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def evaluate_all_models() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data()
    sample_1000 = X_test.iloc[: min(1000, len(X_test))]

    records: list[dict] = []
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for spec in MODEL_SPECS:
        model_path = MODELS_DIR / spec["filename"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found: {model_path}")

        with model_path.open("rb") as f:
            model = pickle.load(f)

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        y_score = get_spam_scores(model, X_test)
        total_eval_time = time.perf_counter() - t0

        t_inf = time.perf_counter()
        _ = model.predict(sample_1000)
        inf_ms_per_1000 = (time.perf_counter() - t_inf) * 1000.0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_score))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curves[spec["name"]] = (fpr, tpr, auc)

        records.append(
            {
                "key": spec["key"],
                "model_name": spec["name"],
                "model_file": spec["filename"],
                "accuracy": round(acc, 6),
                "precision_spam": round(prec, 6),
                "recall_spam": round(rec, 6),
                "f1_spam": round(f1, 6),
                "roc_auc": round(auc, 6),
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
                "errors": int(cm[0][1] + cm[1][0]),
                "inference_ms_per_1000": round(inf_ms_per_1000, 2),
                "eval_time_s": round(total_eval_time, 2),
                "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
            }
        )

    # Sort by highest accuracy, then F1-Spam, then ROC-AUC
    records.sort(
        key=lambda r: (r["accuracy"], r["f1_spam"], r["roc_auc"]),
        reverse=True,
    )
    for rank, rec in enumerate(records, start=1):
        rec["rank"] = rank
        rec["is_best"] = rank == 1

    best = records[0]
    best_src = MODELS_DIR / best["model_file"]
    best_dst = MODELS_DIR / "best_model.pkl"
    shutil.copy2(best_src, best_dst)

    # Save CSV and JSON results
    df_results = pd.DataFrame(records)
    csv_out = RESULTS_DIR / "model_comparison.csv"
    json_out = RESULTS_DIR / "model_comparison.json"
    df_results.to_csv(csv_out, index=False, encoding="utf-8-sig")
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Save metadata for the API
    metadata = {
        "best_model_key": best["key"],
        "best_model_name": best["model_name"],
        "source_model_file": best["model_file"],
        "best_model_path": "models/best_model.pkl",
        "selection_criterion": "Highest test Accuracy (tie-breaker: F1-Spam, ROC-AUC)",
        "test_samples": int(len(y_test)),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "best_metrics": best,
        "comparison": records,
    }
    meta_out = MODELS_DIR / "model_metadata.json"
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Generate comparison chart
    plot_comparison(df_results, roc_curves, FIGURES_DIR / "model_comparison.png")

    print("=== MODEL COMPARISON SUMMARY ===")
    print(
        df_results[
            ["rank", "model_name", "accuracy", "precision_spam", "recall_spam", "f1_spam", "roc_auc", "errors"]
        ].to_string(index=False)
    )
    print(f"\nSelected main model: {best['model_name']} (Accuracy: {best['accuracy']*100:.2f}%)")
    print(f"Copied to: {best_dst}")
    print(f"Metadata : {meta_out}")
    print(f"CSV      : {csv_out}")
    return metadata


def plot_comparison(
    df: pd.DataFrame,
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

    metrics = ["accuracy", "precision_spam", "recall_spam", "f1_spam", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    n_models = len(df)
    width = 0.78 / max(n_models, 1)
    colors = ["#2563eb", "#16a34a", "#f97316", "#a855f7", "#eab308"]

    min_pct = float(df[metrics].min().min() * 100)
    y_bottom = max(0.0, min(90.0, np.floor(min_pct / 5.0) * 5.0 - 2.0))

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] * 100 for m in metrics]
        offset = (i - (n_models - 1) / 2.0) * width
        bars = axes[0].bar(
            x + offset,
            vals,
            width,
            label=row["model_name"],
            color=colors[i % len(colors)],
        )
        for bar, val in zip(bars, vals):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    axes[0].set_ylim(y_bottom, 104)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("So sánh hiệu năng các thuật toán trên tập Test")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for i, (name, (fpr, tpr, auc)) in enumerate(roc_curves.items()):
        axes[1].plot(
            fpr,
            tpr,
            lw=2,
            color=colors[i % len(colors)],
            label=f"{name} (AUC = {auc:.4f})",
        )
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Đường cong ROC của các mô hình")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    evaluate_all_models()
