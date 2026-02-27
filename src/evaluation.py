"""
Evaluation framework for closing time prediction models.

Runs walk-forward (time-series) cross-validation:
  - For each circuit breaker group, use first 70% for warm-up,
    last 30% for testing (strictly no look-ahead).
  - Computes MAE, RMSE, MAPE, Max Error per model per group.
  - Aggregates across all groups.

Also provides:
  - Robustness analysis (performance on outlier subsets)
  - Drift detection (is closing time changing over time?)
  - Visualization utilities
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from .models import BaseModel, build_models, iqr_outlier_mask
from .closing_detector import ClosingResult, DetectionQuality

logger = logging.getLogger(__name__)


@dataclass
class PerModelMetrics:
    model_name: str
    mae_ms: float
    rmse_ms: float
    mape_pct: float
    max_error_ms: float
    n_predictions: int
    within_1ms_pct: float    # fraction of predictions within ±1 ms
    within_5ms_pct: float    # fraction of predictions within ±5 ms


@dataclass
class EvaluationResult:
    group_id: str           # circuit breaker fingerprint (shortened)
    n_total: int
    n_train: int
    n_test: int
    n_outliers_detected: int
    metrics: list[PerModelMetrics]

    def best_model(self) -> Optional[PerModelMetrics]:
        if not self.metrics:
            return None
        return min(self.metrics, key=lambda m: m.mae_ms)


def _compute_metrics(model_name: str, actuals: list[float], predictions: list[float]) -> PerModelMetrics:
    a = np.array(actuals)
    p = np.array(predictions)
    errors = np.abs(a - p)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    # MAPE with guard against zero
    mape = float(np.mean(errors / np.maximum(np.abs(a), 1.0)) * 100)
    max_err = float(np.max(errors))
    n = len(errors)

    return PerModelMetrics(
        model_name=model_name,
        mae_ms=mae,
        rmse_ms=rmse,
        mape_pct=mape,
        max_error_ms=max_err,
        n_predictions=n,
        within_1ms_pct=float(np.mean(errors <= 1.0) * 100),
        within_5ms_pct=float(np.mean(errors <= 5.0) * 100),
    )


def evaluate_group(
    closing_times_ms: list[float],
    results: list[ClosingResult],
    group_id: str,
    train_fraction: float = 0.7,
    cfg: Optional[dict] = None,
) -> Optional[EvaluationResult]:
    """
    Walk-forward evaluation on a single circuit breaker group.

    Only uses GOOD and ESTIMATED quality measurements for training and evaluation.
    OUTLIER quality measurements are excluded from both.
    AMBIGUOUS measurements are used in training but excluded from testing.
    """
    # Filter to usable measurements
    usable_idx = [
        i for i, r in enumerate(results)
        if r.quality in (DetectionQuality.GOOD, DetectionQuality.ESTIMATED)
        and r.t_close_ms is not None
    ]

    if len(usable_idx) < (cfg or {}).get("evaluation", {}).get("min_operations", 10):
        logger.debug(f"Group {group_id[:30]}: insufficient measurements ({len(usable_idx)}), skipping")
        return None

    times = [closing_times_ms[i] for i in usable_idx]
    n = len(times)
    n_train = max(3, int(n * train_fraction))
    n_test = n - n_train

    if n_test < 2:
        return None

    n_outliers = sum(1 for r in results if r.quality == DetectionQuality.OUTLIER)

    # Build models
    models = build_models(cfg)

    # Warm-up: feed training data sequentially
    for t in times[:n_train]:
        for model in models.values():
            model.update(t)

    # Walk-forward test
    actuals: dict[str, list[float]] = {name: [] for name in models}
    preds: dict[str, list[float]] = {name: [] for name in models}

    for t_actual in times[n_train:]:
        for name, model in models.items():
            p = model.predict()
            if p is not None:
                preds[name].append(p)
                actuals[name].append(t_actual)
        # Update all models with actual
        for model in models.values():
            model.update(t_actual)

    metrics = []
    for name in models:
        if actuals[name]:
            metrics.append(_compute_metrics(name, actuals[name], preds[name]))

    return EvaluationResult(
        group_id=group_id[:60],
        n_total=n,
        n_train=n_train,
        n_test=n_test,
        n_outliers_detected=n_outliers,
        metrics=metrics,
    )


def evaluate_all(
    groups_data: dict[str, tuple[list[float], list[ClosingResult]]],
    cfg: Optional[dict] = None,
) -> list[EvaluationResult]:
    """
    Evaluate all circuit breaker groups.

    Args:
        groups_data: dict mapping group_id → (list_of_closing_times_ms, list_of_results)
        cfg: config dict

    Returns:
        List of EvaluationResult (one per group with sufficient data)
    """
    results = []
    train_frac = (cfg or {}).get("evaluation", {}).get("train_fraction", 0.7)

    for gid, (times, det_results) in groups_data.items():
        res = evaluate_group(times, det_results, gid, train_frac, cfg)
        if res is not None:
            results.append(res)

    logger.info(f"Evaluated {len(results)}/{len(groups_data)} groups")
    return results


def aggregate_metrics(eval_results: list[EvaluationResult]) -> pd.DataFrame:
    """Aggregate per-model metrics across all evaluated groups."""
    rows = []
    for res in eval_results:
        for m in res.metrics:
            rows.append({
                "group_id": res.group_id,
                "model": m.model_name,
                "mae_ms": m.mae_ms,
                "rmse_ms": m.rmse_ms,
                "mape_pct": m.mape_pct,
                "max_error_ms": m.max_error_ms,
                "within_1ms_pct": m.within_1ms_pct,
                "within_5ms_pct": m.within_5ms_pct,
                "n_predictions": m.n_predictions,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model")
        .agg(
            MAE_mean=("mae_ms", "mean"),
            MAE_std=("mae_ms", "std"),
            RMSE_mean=("rmse_ms", "mean"),
            MAPE_mean=("mape_pct", "mean"),
            MaxError_mean=("max_error_ms", "mean"),
            Within1ms_mean=("within_1ms_pct", "mean"),
            Within5ms_mean=("within_5ms_pct", "mean"),
            n_groups=("group_id", "count"),
        )
        .round(3)
        .sort_values("MAE_mean")
    )
    return summary


def detect_drift(times_ms: list[float], window: int = 10) -> dict:
    """
    Test for monotonic trend in closing times using Mann-Kendall test.
    Returns dict with test statistic, p-value, and trend direction.
    """
    from scipy import stats

    arr = np.array(times_ms)
    if len(arr) < 4:
        return {"trend": "insufficient_data", "p_value": None, "slope_ms_per_op": None}

    # Mann-Kendall
    tau, p_value = stats.kendalltau(np.arange(len(arr)), arr)
    slope_result = None
    try:
        from scipy.stats import theilslopes
        r = theilslopes(arr, np.arange(len(arr)))
        slope_ms_per_op = float(r.slope)
    except Exception:
        slope_ms_per_op = None

    if p_value < 0.05:
        trend = "increasing" if tau > 0 else "decreasing"
    else:
        trend = "no_significant_trend"

    return {
        "trend": trend,
        "tau": float(tau),
        "p_value": float(p_value),
        "slope_ms_per_op": slope_ms_per_op,
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_cb_timeline(
    times_ms: list[float],
    results: list[ClosingResult],
    predictions: Optional[dict[str, list[float]]] = None,
    title: str = "Circuit Breaker Closing Times",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot closing time history for a single circuit breaker.
    Shows actual measurements, outliers, and model predictions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ops = np.arange(len(times_ms))
    qualities = [r.quality for r in results]

    colors = {
        DetectionQuality.GOOD: "#2196F3",
        DetectionQuality.ESTIMATED: "#4CAF50",
        DetectionQuality.AMBIGUOUS: "#FF9800",
        DetectionQuality.OUTLIER: "#F44336",
        DetectionQuality.FAILED: "#9E9E9E",
    }

    ax1 = axes[0]
    for q in DetectionQuality:
        mask = [qi == q and times_ms[i] is not None for i, qi in enumerate(qualities)]
        idx = [i for i, m in enumerate(mask) if m]
        if idx:
            ax1.scatter(idx, [times_ms[i] for i in idx], c=colors[q],
                       label=q.value, s=50, zorder=3, alpha=0.8)

    if predictions:
        model_colors = ["#FF5722", "#9C27B0", "#00BCD4", "#8BC34A", "#FFC107"]
        for (name, preds), col in zip(predictions.items(), model_colors):
            if preds:
                n_train = len(times_ms) - len(preds)
                pred_ops = np.arange(n_train, n_train + len(preds))
                ax1.plot(pred_ops, preds, "--", color=col, label=f"Pred: {name}", linewidth=1.5)

    ax1.set_ylabel("Closing time (ms)")
    ax1.set_title(title)
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Error subplot (if predictions available)
    ax2 = axes[1]
    if predictions:
        for (name, preds), col in zip(predictions.items(), model_colors):
            if preds:
                n_train = len(times_ms) - len(preds)
                actual_test = times_ms[n_train:]
                errors = [abs(a - p) for a, p in zip(actual_test, preds)]
                pred_ops = np.arange(n_train, n_train + len(errors))
                ax2.plot(pred_ops, errors, "-", color=col, label=name, linewidth=1.5, alpha=0.8)
        ax2.axhline(5.0, color="red", linestyle=":", linewidth=1, label="5 ms threshold")
        ax2.set_ylabel("|Error| (ms)")
        ax2.legend(fontsize=8)
    else:
        n = len(times_ms)
        rolling_med = [
            float(np.median(times_ms[max(0, i - 10): i + 1]))
            for i in range(n)
        ]
        ax2.plot(ops, rolling_med, "k-", linewidth=1.5, label="Rolling median (10)")
        ax2.set_ylabel("Rolling median (ms)")
    ax2.set_xlabel("Operation #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(summary_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Bar chart comparing model MAE across evaluated groups."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = ["MAE_mean", "Within1ms_mean", "Within5ms_mean"]
    titles = ["Mean Absolute Error (ms)", "Within ±1 ms (%)", "Within ±5 ms (%)"]
    colors = sns.color_palette("husl", len(summary_df))

    for ax, metric, title in zip(axes, metrics, titles):
        df_sorted = summary_df.sort_values(metric)
        bars = ax.barh(df_sorted.index, df_sorted[metric], color=colors)
        ax.set_title(title)
        ax.set_xlabel(metric.replace("_", " "))
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(
                bar.get_width() + 0.01 * df_sorted[metric].max(),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9,
            )
        ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle("Model Comparison — Circuit Breaker Closing Time Prediction", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_detection_example(
    rec,
    result: ClosingResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single COMTRADE record showing the close command, current,
    and detected closing time.
    """
    from .comtrade_parser import ComtradeRecord
    assert isinstance(rec, ComtradeRecord)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ts_ms = rec.timestamps * 1000  # convert to ms

    # Plot currents
    ax1 = axes[0]
    current_chs = rec.primary_currents()
    for ch in current_chs[:3]:  # max 3 phases
        ax1.plot(ts_ms, ch.data, label=ch.name, linewidth=1)
    ax1.axhline(result.current_threshold_a, color="orange", linestyle="--",
                linewidth=1, label=f"Threshold ({result.current_threshold_a:.1f} A)")
    if result.t_current_s is not None:
        ax1.axvline(result.t_current_s * 1000, color="blue", linestyle="-",
                    linewidth=1.5, label=f"t_current = {result.t_current_s * 1000:.1f} ms")
    ax1.set_ylabel("Current (A, primary)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot digital channels
    ax2 = axes[1]
    for i, ch in enumerate(rec.digital[:4]):  # max 4 digital channels
        offset = i * 1.2
        ax2.step(ts_ms, ch.data.astype(float) + offset, where="post",
                 label=ch.name, linewidth=1.5)
    if result.t_command_s is not None:
        ax2.axvline(result.t_command_s * 1000, color="red", linestyle="-",
                    linewidth=1.5, label=f"t_command = {result.t_command_s * 1000:.1f} ms")
    ax2.set_xlabel("Time (ms from record start)")
    ax2.set_ylabel("Digital channels")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    title = (
        f"Closing event — {rec.station_name} / {rec.rec_dev_id}\n"
        f"T_close = {result.t_close_ms:.2f} ms  |  Quality: {result.quality.value}"
        if result.t_close_ms is not None
        else f"Closing event — {rec.station_name} / {rec.rec_dev_id}  |  Quality: {result.quality.value}"
    )
    axes[0].set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
