"""
Assignment 2: From Trees to Neural Networks
Home Credit Default Risk Dataset
GBDT (XGBoost) vs MLP comparison pipeline.
Shun Yan sy3208
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
MISSING_THRESHOLD = 0.60
DEFAULT_SAMPLE_SIZE = 80_000
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
MISSING_TOKEN = "Missing"


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def stratified_subsample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[TARGET_COL],
        random_state=SEED,
    )
    return sampled_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["CREDIT_INCOME_RATIO"] = frame["AMT_CREDIT"] / (frame["AMT_INCOME_TOTAL"] + 1.0)
    frame["ANNUITY_INCOME_RATIO"] = frame["AMT_ANNUITY"] / (frame["AMT_INCOME_TOTAL"] + 1.0)
    frame["CREDIT_ANNUITY_RATIO"] = frame["AMT_CREDIT"] / (frame["AMT_ANNUITY"] + 1.0)
    frame["AGE_YEARS"] = (-frame["DAYS_BIRTH"]) / 365.0
    frame["EMPLOYED_YEARS"] = (-frame["DAYS_EMPLOYED"]) / 365.0
    frame["INCOME_PER_PERSON"] = frame["AMT_INCOME_TOTAL"] / (frame["CNT_FAM_MEMBERS"] + 1.0)
    return frame


def encode_categoricals(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, dict[str, int]]]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    encoders: dict[str, dict[str, int]] = {}

    for col in cat_cols:
        train_series = train_df[col].fillna(MISSING_TOKEN).astype(str)
        categories = pd.Index(pd.unique(train_series))
        if MISSING_TOKEN not in categories:
            categories = categories.append(pd.Index([MISSING_TOKEN]))

        mapping = {value: idx for idx, value in enumerate(categories.tolist())}
        default_code = mapping[MISSING_TOKEN]

        for split_df in (train_df, val_df, test_df):
            encoded = (
                split_df[col]
                .fillna(MISSING_TOKEN)
                .astype(str)
                .map(mapping)
                .fillna(default_code)
                .astype(np.int32)
            )
            split_df[col] = encoded

        encoders[col] = {
            "num_classes": len(mapping),
            "default_code": default_code,
        }

    return train_df, val_df, test_df, cat_cols, encoders


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
    }


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["metrics"]
    return (
        metrics["auc_pr"],
        metrics["f1"],
        metrics["recall"],
        metrics["accuracy"],
    )


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=selection_key)


def oversample_minority(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    minority_fraction_of_majority: float = 0.40,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    y_array = y_train.to_numpy()
    majority_idx = np.flatnonzero(y_array == 0)
    minority_idx = np.flatnonzero(y_array == 1)
    extra_minority = int(len(majority_idx) * minority_fraction_of_majority)
    sampled_minority = rng.choice(minority_idx, size=extra_minority, replace=True)
    combined_idx = np.concatenate([majority_idx, sampled_minority])
    rng.shuffle(combined_idx)
    return X_train_scaled[combined_idx], y_array[combined_idx]


def train_xgb(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label: str,
    value: Any,
    keep_model: bool = False,
    keep_curves: bool = False,
) -> dict[str, Any]:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
        importance_type="gain",
        **params,
    )

    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    elapsed = time.perf_counter() - start_time

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    result = {
        "label": str(label),
        "value": value,
        "params": params.copy(),
        "metrics": compute_metrics(y_val, val_pred, val_proba),
        "train_time": float(elapsed),
        "best_iteration": int(
            getattr(model, "best_iteration", None)
            if getattr(model, "best_iteration", None) is not None
            else params["n_estimators"] - 1
        ),
    }

    if keep_curves:
        evals_result = model.evals_result()
        result["train_logloss"] = evals_result["validation_0"]["logloss"]
        result["val_logloss"] = evals_result["validation_1"]["logloss"]

    if keep_model:
        result["model"] = model

    return result


def train_mlp(
    params: dict[str, Any],
    X_train_mlp: np.ndarray,
    y_train_mlp: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: pd.Series,
    label: str,
    value: Any,
    keep_model: bool = False,
    keep_curves: bool = False,
) -> dict[str, Any]:
    model = MLPClassifier(
        solver="adam",
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        batch_size=512,
        verbose=False,
        **params,
    )

    start_time = time.perf_counter()
    model.fit(X_train_mlp, y_train_mlp)
    elapsed = time.perf_counter() - start_time

    val_pred = model.predict(X_val_scaled)
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    result = {
        "label": str(label),
        "value": value,
        "params": params.copy(),
        "metrics": compute_metrics(y_val, val_pred, val_proba),
        "train_time": float(elapsed),
        "n_iter": int(model.n_iter_),
    }

    if keep_curves:
        result["loss_curve"] = list(model.loss_curve_)
        result["validation_scores"] = list(getattr(model, "validation_scores_", []))

    if keep_model:
        result["model"] = model

    return result


def plot_xgb_loss(result: dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["train_logloss"], label="Train", linewidth=2)
    ax.plot(result["val_logloss"], label="Validation", linewidth=2)
    ax.axvline(
        x=result["best_iteration"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Early stop @ {result['best_iteration']}",
    )
    ax.set_title("XGBoost Training vs Validation Loss")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Log Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_xgb_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_features = feature_importance.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(top_features)),
        top_features["importance"].iloc[::-1],
        color="steelblue",
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].iloc[::-1], fontsize=9)
    ax.set_title("Top 20 XGBoost Features")
    ax.set_xlabel("Gain-based Importance")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return feature_importance


def plot_xgb_learning_rates(results: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.5), sharey=True)
    for ax, result in zip(axes, results):
        ax.plot(result["train_logloss"], label="Train", linewidth=1.5)
        ax.plot(result["val_logloss"], label="Validation", linewidth=1.5)
        ax.axvline(
            x=result["best_iteration"],
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax.set_title(
            f"LR={result['value']}\nAUC-PR={result['metrics']['auc_pr']:.3f}, F1={result['metrics']['f1']:.3f}"
        )
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()
    plt.suptitle("XGBoost Learning Rate Comparison", y=1.04, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_sweeps(
    sweep_panels: list[dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    cols = 2
    rows = int(np.ceil(len(sweep_panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, panel in zip(axes, sweep_panels):
        labels = [item["label"] for item in panel["results"]]
        auc_values = [item["metrics"]["auc_pr"] for item in panel["results"]]
        f1_values = [item["metrics"]["f1"] for item in panel["results"]]
        x_positions = np.arange(len(labels))

        ax.plot(x_positions, auc_values, marker="o", linewidth=2, label="Validation AUC-PR")
        ax.plot(x_positions, f1_values, marker="s", linewidth=2, label="Validation F1")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, max(max(auc_values), max(f1_values)) * 1.25)
        ax.set_title(panel["title"])
        ax.grid(True, alpha=0.3)

        best_result = select_best(panel["results"])
        best_index = labels.index(best_result["label"])
        ax.scatter(best_index, best_result["metrics"]["auc_pr"], color="red", zorder=3, s=50)

    for ax in axes[len(sweep_panels) :]:
        ax.axis("off")

    axes[0].legend(loc="upper left")
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mlp_loss(result: dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["loss_curve"], label="Training loss", linewidth=2, color="darkorange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Loss Curve")
    ax.grid(True, alpha=0.3)

    if result["validation_scores"]:
        ax2 = ax.twinx()
        ax2.plot(
            result["validation_scores"],
            label="Internal validation score",
            linewidth=2,
            color="green",
            linestyle="--",
        )
        ax2.set_ylabel("Validation score")
        ax2.legend(loc="center right")

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_architecture_comparison(results: list[dict[str, Any]], output_path: Path) -> None:
    labels = [result["label"] for result in results]
    auc_values = [result["metrics"]["auc_pr"] for result in results]
    f1_values = [result["metrics"]["f1"] for result in results]
    x_positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(x_positions - width / 2, auc_values, width, label="Validation AUC-PR", color="steelblue")
    ax.bar(x_positions + width / 2, f1_values, width, label="Validation F1", color="darkorange")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(auc_values), max(f1_values)) * 1.25)
    ax.set_title("MLP Architecture Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_comparison(
    xgb_metrics: dict[str, float],
    mlp_metrics: dict[str, float],
    y_test: pd.Series,
    y_test_proba_xgb: np.ndarray,
    y_test_proba_mlp: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    metric_order = ["accuracy", "precision", "recall", "f1", "auc_pr"]
    display_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-PR"]
    x_positions = np.arange(len(metric_order))
    width = 0.35

    xgb_values = [xgb_metrics[name] for name in metric_order]
    mlp_values = [mlp_metrics[name] for name in metric_order]

    axes[0].bar(x_positions - width / 2, xgb_values, width, label="XGBoost", color="steelblue")
    axes[0].bar(x_positions + width / 2, mlp_values, width, label="MLP", color="darkorange")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(display_names, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Test-set Metrics")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, y_test_proba_xgb)
    mlp_precision, mlp_recall, _ = precision_recall_curve(y_test, y_test_proba_mlp)
    axes[1].plot(
        xgb_recall,
        xgb_precision,
        label=f"XGBoost (AUC-PR={xgb_metrics['auc_pr']:.3f})",
        linewidth=2,
        color="steelblue",
    )
    axes[1].plot(
        mlp_recall,
        mlp_precision,
        label=f"MLP (AUC-PR={mlp_metrics['auc_pr']:.3f})",
        linewidth=2,
        color="darkorange",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def as_json_ready(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for result in results:
        sanitized = {
            key: value
            for key, value in result.items()
            if key not in {"model", "train_logloss", "val_logloss", "loss_curve", "validation_scores"}
        }
        if "value" in sanitized and isinstance(sanitized["value"], tuple):
            sanitized["value"] = list(sanitized["value"])
        if "params" in sanitized:
            sanitized["params"] = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in sanitized["params"].items()
            }
        cleaned.append(sanitized)
    return cleaned


def run_pipeline(data_path: Path, output_dir: Path, sample_size: int) -> None:
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print_header("1. DATA PREPARATION")
    df = pd.read_csv(data_path)
    print(f"Loaded {data_path}")
    print(f"Original shape: {df.shape}")
    print(f"Original target distribution:\n{df[TARGET_COL].value_counts(normalize=True).sort_index()}")

    df = stratified_subsample(df, sample_size)
    print(f"Working shape after stratified subsample: {df.shape}")
    print(f"Working target distribution:\n{df[TARGET_COL].value_counts(normalize=True).sort_index()}")

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    anomaly_mask = df["DAYS_EMPLOYED"] == 365243
    df["DAYS_EMPLOYED_ANOMALY"] = anomaly_mask.astype(int)
    df.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan
    print(f"DAYS_EMPLOYED anomaly rows: {int(anomaly_mask.sum())}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_holdout_raw,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=SEED,
    )

    high_missing_cols = X_train_raw.columns[X_train_raw.isna().mean() > MISSING_THRESHOLD].tolist()
    X_train = X_train_raw.drop(columns=high_missing_cols).copy()
    X_val = X_val_raw.drop(columns=high_missing_cols).copy()
    X_test = X_test_raw.drop(columns=high_missing_cols).copy()
    print(f"Dropped {len(high_missing_cols)} columns with >60% missing based on train split.")

    X_train = add_engineered_features(X_train)
    X_val = add_engineered_features(X_val)
    X_test = add_engineered_features(X_test)

    X_train, X_val, X_test, cat_cols, encoders = encode_categoricals(X_train, X_val, X_test)
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_mlp, y_train_mlp = oversample_minority(X_train_scaled, y_train)

    print(f"Train/Val/Test sizes: {len(X_train)}, {len(X_val)}, {len(X_test)}")
    print(f"Categorical columns encoded: {len(cat_cols)}")
    print(
        "MLP oversampled class mix: "
        f"0={np.mean(y_train_mlp == 0):.3f}, 1={np.mean(y_train_mlp == 1):.3f}"
    )

    print_header("2. XGBOOST")
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    base_xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }

    baseline_xgb = train_xgb(
        base_xgb_params,
        X_train,
        y_train,
        X_val,
        y_val,
        label="baseline",
        value="baseline",
        keep_curves=True,
    )

    xgb_params = base_xgb_params.copy()
    xgb_lr_results = [
        train_xgb(
            {**xgb_params, "learning_rate": value},
            X_train,
            y_train,
            X_val,
            y_val,
            label=f"{value}",
            value=value,
            keep_curves=True,
        )
        for value in [0.01, 0.1, 0.3]
    ]
    xgb_params["learning_rate"] = select_best(xgb_lr_results)["value"]

    xgb_depth_results = [
        train_xgb(
            {**xgb_params, "max_depth": value},
            X_train,
            y_train,
            X_val,
            y_val,
            label=str(value),
            value=value,
        )
        for value in [4, 6, 8]
    ]
    xgb_params["max_depth"] = select_best(xgb_depth_results)["value"]

    xgb_subsample_results = [
        train_xgb(
            {**xgb_params, "subsample": value},
            X_train,
            y_train,
            X_val,
            y_val,
            label=f"{value:.1f}",
            value=value,
        )
        for value in [0.6, 0.8, 1.0]
    ]
    xgb_params["subsample"] = select_best(xgb_subsample_results)["value"]

    reg_grid = [
        {"label": "0.0 / 0.5", "reg_alpha": 0.0, "reg_lambda": 0.5},
        {"label": "0.1 / 1.0", "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"label": "1.0 / 5.0", "reg_alpha": 1.0, "reg_lambda": 5.0},
    ]
    xgb_reg_results = [
        train_xgb(
            {
                **xgb_params,
                "reg_alpha": option["reg_alpha"],
                "reg_lambda": option["reg_lambda"],
            },
            X_train,
            y_train,
            X_val,
            y_val,
            label=option["label"],
            value={
                "reg_alpha": option["reg_alpha"],
                "reg_lambda": option["reg_lambda"],
            },
        )
        for option in reg_grid
    ]
    best_reg = select_best(xgb_reg_results)["value"]
    xgb_params["reg_alpha"] = best_reg["reg_alpha"]
    xgb_params["reg_lambda"] = best_reg["reg_lambda"]

    xgb_n_estimators_results = [
        train_xgb(
            {**xgb_params, "n_estimators": value},
            X_train,
            y_train,
            X_val,
            y_val,
            label=str(value),
            value=value,
        )
        for value in [300, 500, 800]
    ]
    xgb_params["n_estimators"] = select_best(xgb_n_estimators_results)["value"]

    final_xgb = train_xgb(
        xgb_params,
        X_train,
        y_train,
        X_val,
        y_val,
        label="final_xgb",
        value="final_xgb",
        keep_model=True,
        keep_curves=True,
    )
    print(f"Selected XGBoost params: {xgb_params}")
    print(f"Validation AUC-PR={final_xgb['metrics']['auc_pr']:.4f}, F1={final_xgb['metrics']['f1']:.4f}")

    print_header("3. MLP")
    base_mlp_params = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "learning_rate_init": 0.001,
        "max_iter": 300,
    }

    baseline_mlp = train_mlp(
        base_mlp_params,
        X_train_mlp,
        y_train_mlp,
        X_val_scaled,
        y_val,
        label="baseline",
        value="baseline",
        keep_curves=True,
    )

    mlp_params = base_mlp_params.copy()
    architecture_grid = {
        "(64,)": (64,),
        "(128, 64)": (128, 64),
        "(256, 128, 64)": (256, 128, 64),
    }
    mlp_architecture_results = [
        train_mlp(
            {**mlp_params, "hidden_layer_sizes": architecture},
            X_train_mlp,
            y_train_mlp,
            X_val_scaled,
            y_val,
            label=label,
            value=architecture,
        )
        for label, architecture in architecture_grid.items()
    ]
    mlp_params["hidden_layer_sizes"] = select_best(mlp_architecture_results)["value"]

    mlp_activation_results = [
        train_mlp(
            {**mlp_params, "activation": activation},
            X_train_mlp,
            y_train_mlp,
            X_val_scaled,
            y_val,
            label=activation,
            value=activation,
        )
        for activation in ["relu", "tanh"]
    ]
    mlp_params["activation"] = select_best(mlp_activation_results)["value"]

    mlp_lr_results = [
        train_mlp(
            {**mlp_params, "learning_rate_init": value},
            X_train_mlp,
            y_train_mlp,
            X_val_scaled,
            y_val,
            label=str(value),
            value=value,
        )
        for value in [0.001, 0.01, 0.1]
    ]
    mlp_params["learning_rate_init"] = select_best(mlp_lr_results)["value"]

    mlp_max_iter_results = [
        train_mlp(
            {**mlp_params, "max_iter": value},
            X_train_mlp,
            y_train_mlp,
            X_val_scaled,
            y_val,
            label=str(value),
            value=value,
        )
        for value in [150, 300, 450]
    ]
    mlp_params["max_iter"] = select_best(mlp_max_iter_results)["value"]

    final_mlp = train_mlp(
        mlp_params,
        X_train_mlp,
        y_train_mlp,
        X_val_scaled,
        y_val,
        label="final_mlp",
        value="final_mlp",
        keep_model=True,
        keep_curves=True,
    )
    print(f"Selected MLP params: {mlp_params}")
    print(f"Validation AUC-PR={final_mlp['metrics']['auc_pr']:.4f}, F1={final_mlp['metrics']['f1']:.4f}")

    print_header("4. TEST-SET COMPARISON")
    xgb_test_proba = final_xgb["model"].predict_proba(X_test)[:, 1]
    xgb_test_pred = (xgb_test_proba >= 0.5).astype(int)
    mlp_test_proba = final_mlp["model"].predict_proba(X_test_scaled)[:, 1]
    mlp_test_pred = (mlp_test_proba >= 0.5).astype(int)

    xgb_test_metrics = compute_metrics(y_test, xgb_test_pred, xgb_test_proba)
    mlp_test_metrics = compute_metrics(y_test, mlp_test_pred, mlp_test_proba)

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Accuracy": xgb_test_metrics["accuracy"],
                "Precision": xgb_test_metrics["precision"],
                "Recall": xgb_test_metrics["recall"],
                "F1-Score": xgb_test_metrics["f1"],
                "AUC-PR": xgb_test_metrics["auc_pr"],
                "Training Time (s)": final_xgb["train_time"],
            },
            {
                "Model": "MLP",
                "Accuracy": mlp_test_metrics["accuracy"],
                "Precision": mlp_test_metrics["precision"],
                "Recall": mlp_test_metrics["recall"],
                "F1-Score": mlp_test_metrics["f1"],
                "AUC-PR": mlp_test_metrics["auc_pr"],
                "Training Time (s)": final_mlp["train_time"],
            },
        ]
    )
    print(comparison_df.round(4).to_string(index=False))

    print_header("5. SAVING ARTIFACTS")
    plot_xgb_loss(final_xgb, plots_dir / "xgb_train_val_loss.png")
    feature_importance = plot_xgb_feature_importance(
        final_xgb["model"],
        list(X_train.columns),
        plots_dir / "xgb_feature_importance.png",
    )
    plot_xgb_learning_rates(xgb_lr_results, plots_dir / "xgb_learning_rate_comparison.png")
    plot_metric_sweeps(
        [
            {"title": "Max Depth", "results": xgb_depth_results},
            {"title": "Subsample", "results": xgb_subsample_results},
            {"title": "Reg Alpha / Lambda", "results": xgb_reg_results},
            {"title": "n_estimators", "results": xgb_n_estimators_results},
        ],
        title="XGBoost Hyperparameter Sweeps",
        output_path=plots_dir / "xgb_hyperparameter_sweeps.png",
    )

    plot_mlp_loss(final_mlp, plots_dir / "mlp_loss_curve.png")
    plot_architecture_comparison(mlp_architecture_results, plots_dir / "mlp_architecture_comparison.png")
    plot_metric_sweeps(
        [
            {"title": "Activation", "results": mlp_activation_results},
            {"title": "Learning Rate", "results": mlp_lr_results},
            {"title": "max_iter", "results": mlp_max_iter_results},
        ],
        title="MLP Hyperparameter Sweeps",
        output_path=plots_dir / "mlp_hyperparameter_sweeps.png",
    )
    plot_final_comparison(
        xgb_test_metrics,
        mlp_test_metrics,
        y_test,
        xgb_test_proba,
        mlp_test_proba,
        plots_dir / "gbdt_vs_mlp_comparison.png",
    )

    comparison_df.to_csv(output_dir / "final_test_metrics.csv", index=False)

    xgb_sweep_table = pd.concat(
        [
            pd.DataFrame(as_json_ready(xgb_lr_results)).assign(parameter="learning_rate"),
            pd.DataFrame(as_json_ready(xgb_depth_results)).assign(parameter="max_depth"),
            pd.DataFrame(as_json_ready(xgb_subsample_results)).assign(parameter="subsample"),
            pd.DataFrame(as_json_ready(xgb_reg_results)).assign(parameter="reg_alpha_reg_lambda"),
            pd.DataFrame(as_json_ready(xgb_n_estimators_results)).assign(parameter="n_estimators"),
        ],
        ignore_index=True,
    )
    mlp_sweep_table = pd.concat(
        [
            pd.DataFrame(as_json_ready(mlp_architecture_results)).assign(parameter="hidden_layer_sizes"),
            pd.DataFrame(as_json_ready(mlp_activation_results)).assign(parameter="activation"),
            pd.DataFrame(as_json_ready(mlp_lr_results)).assign(parameter="learning_rate_init"),
            pd.DataFrame(as_json_ready(mlp_max_iter_results)).assign(parameter="max_iter"),
        ],
        ignore_index=True,
    )
    xgb_sweep_table.to_csv(output_dir / "xgb_sweeps.csv", index=False)
    mlp_sweep_table.to_csv(output_dir / "mlp_sweeps.csv", index=False)

    report_data = {
        "data_summary": {
            "data_path": str(data_path),
            "original_rows": int(len(pd.read_csv(data_path))),
            "working_rows": int(len(df)),
            "working_target_rate": float(df[TARGET_COL].mean()),
            "dropped_columns": int(len(high_missing_cols)),
            "num_features_after_processing": int(X_train.shape[1]),
            "categorical_columns": int(len(cat_cols)),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "scale_pos_weight": scale_pos_weight,
            "oversampled_positive_rate": float(np.mean(y_train_mlp == 1)),
        },
        "xgb": {
            "baseline": as_json_ready([baseline_xgb])[0],
            "selected_params": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in xgb_params.items()
            },
            "final_validation": as_json_ready([final_xgb])[0],
            "sweeps": {
                "learning_rate": as_json_ready(xgb_lr_results),
                "max_depth": as_json_ready(xgb_depth_results),
                "subsample": as_json_ready(xgb_subsample_results),
                "reg_alpha_reg_lambda": as_json_ready(xgb_reg_results),
                "n_estimators": as_json_ready(xgb_n_estimators_results),
            },
            "top_features": feature_importance.head(10).to_dict(orient="records"),
        },
        "mlp": {
            "baseline": as_json_ready([baseline_mlp])[0],
            "selected_params": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in mlp_params.items()
            },
            "final_validation": as_json_ready([final_mlp])[0],
            "sweeps": {
                "hidden_layer_sizes": as_json_ready(mlp_architecture_results),
                "activation": as_json_ready(mlp_activation_results),
                "learning_rate_init": as_json_ready(mlp_lr_results),
                "max_iter": as_json_ready(mlp_max_iter_results),
            },
        },
        "comparison": {
            "test_metrics": {
                "XGBoost": xgb_test_metrics,
                "MLP": mlp_test_metrics,
            },
            "training_times": {
                "XGBoost": float(final_xgb["train_time"]),
                "MLP": float(final_mlp["train_time"]),
            },
        },
        "plots": {
            "xgb_train_val_loss": str(plots_dir / "xgb_train_val_loss.png"),
            "xgb_feature_importance": str(plots_dir / "xgb_feature_importance.png"),
            "xgb_learning_rate_comparison": str(plots_dir / "xgb_learning_rate_comparison.png"),
            "xgb_hyperparameter_sweeps": str(plots_dir / "xgb_hyperparameter_sweeps.png"),
            "mlp_loss_curve": str(plots_dir / "mlp_loss_curve.png"),
            "mlp_architecture_comparison": str(plots_dir / "mlp_architecture_comparison.png"),
            "mlp_hyperparameter_sweeps": str(plots_dir / "mlp_hyperparameter_sweeps.png"),
            "gbdt_vs_mlp_comparison": str(plots_dir / "gbdt_vs_mlp_comparison.png"),
        },
    }

    with (output_dir / "report_data.json").open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)

    print(f"Saved outputs to {output_dir}")

    del final_xgb["model"]
    del final_mlp["model"]
    gc.collect()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Assignment 2 experiments.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root / "home-credit-default-risk" / "application_train.csv",
        help="Path to the Home Credit training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts",
        help="Directory to store plots, tables, and report data.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Stratified sample size used for experimentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    run_pipeline(args.data_path, args.output_dir, args.sample_size)


if __name__ == "__main__":
    main()
