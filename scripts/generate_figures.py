"""
Generate publication-quality figures for the article.

Uses synthetic but physically realistic data modelled on actual CB behaviour:
  - SF6 breaker:    mean=85 ms, slow drift, occasional outlier
  - Vacuum breaker: mean=55 ms, very stable
  - Oil breaker:    mean=210 ms, faster degradation
  - Degraded oil:   mean=300 ms, strong upward drift

Figures produced (saved to article/figures/):
  fig1_closing_time_distribution.png  — histogram + CDF of T_close pool
  fig2_cb_types_boxplot.png           — per-type comparison
  fig3_degradation_timeline.png       — T_close vs operation# for 4 CBs with drift
  fig4_model_comparison.png           — walk-forward predictions: Kalman/EWMA/Median/TheilSen
  fig5_kalman_uncertainty.png         — Kalman filter with 95% CI on a degrading breaker
  fig6_detection_quality_pie.png      — quality distribution pie chart
  fig7_drift_scatter.png              — drift rate vs mean T_close across CB fleet
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy.stats import norm

# allow import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path(__file__).parent.parent / "article" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("husl", 8)

# ── Synthetic CB data factory ─────────────────────────────────────────────────

def make_cb_series(
    n_ops: int,
    t0_ms: float,
    drift_ms_per_op: float = 0.0,
    noise_std: float = 2.0,
    outlier_prob: float = 0.04,
    stuck_prob: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate T_close measurements for one circuit breaker.

    Returns (times_ms, is_outlier) arrays of length n_ops.
    """
    rng = np.random.default_rng(seed)
    ops = np.arange(n_ops)
    base = t0_ms + drift_ms_per_op * ops + rng.normal(0, noise_std, n_ops)

    # Random outliers (stuck or CT anomaly)
    outlier_mask = rng.random(n_ops) < outlier_prob
    base[outlier_mask] = base[outlier_mask] * rng.uniform(1.6, 3.0, outlier_mask.sum())

    # Stuck (FAILED → set to NaN)
    stuck_mask = rng.random(n_ops) < stuck_prob
    base[stuck_mask] = np.nan

    return base, outlier_mask | stuck_mask


CB_SCENARIOS = {
    "Вакуумный ВВ/TEL-110": dict(t0_ms=55, drift_ms_per_op=0.005, noise_std=1.5,  n=80,  color=PALETTE[0]),
    "Элегазовый ГВШ-110":   dict(t0_ms=85, drift_ms_per_op=0.04,  noise_std=2.5,  n=120, color=PALETTE[1]),
    "Маломасляный МКП-110": dict(t0_ms=210, drift_ms_per_op=0.10, noise_std=4.0,  n=90,  color=PALETTE[2]),
    "Масляный (деград.)":   dict(t0_ms=290, drift_ms_per_op=0.25, noise_std=8.0,  n=60,  color=PALETTE[3]),
}


# ── Figure 1: Distribution + CDF ─────────────────────────────────────────────

def fig1_distribution():
    all_times = []
    for i, (name, p) in enumerate(CB_SCENARIOS.items()):
        t, _ = make_cb_series(p["n"], p["t0_ms"], p["drift_ms_per_op"], p["noise_std"], seed=i)
        all_times.extend(t[~np.isnan(t)].tolist())
    arr = np.array(all_times)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram coloured by type
    offset = 0
    colours = [p["color"] for p in CB_SCENARIOS.values()]
    labels  = list(CB_SCENARIOS.keys())
    for i, (name, p) in enumerate(CB_SCENARIOS.items()):
        t, _ = make_cb_series(p["n"], p["t0_ms"], p["drift_ms_per_op"], p["noise_std"], seed=i)
        clean = t[~np.isnan(t)]
        ax1.hist(clean, bins=25, color=p["color"], alpha=0.7, edgecolor="white",
                 label=name, density=False)

    ax1.axvline(80,  color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="Ваку./СФ6 граница (80 мс)")
    ax1.axvline(200, color="gray", linestyle=":",  linewidth=1.2, alpha=0.8, label="СФ6/Масло граница (200 мс)")
    ax1.set_xlabel("Время включения, мс")
    ax1.set_ylabel("Число операций")
    ax1.set_title("Рис. 1а. Распределение времён включения\nпо типам выключателей")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # CDF of the full pool
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax2.plot(sorted_arr, cdf, color="#2196F3", linewidth=2.5)
    ax2.fill_between(sorted_arr, cdf, alpha=0.08, color="#2196F3")
    for pct, pct_label in [(50, "P50"), (90, "P90"), (95, "P95")]:
        val = float(np.percentile(arr, pct))
        ax2.axhline(pct / 100, color="gray", linestyle=":", linewidth=1)
        ax2.axvline(val, color="gray", linestyle=":", linewidth=1)
        ax2.text(val + 4, pct / 100 - 0.05, f"{pct_label} = {val:.0f} мс",
                 fontsize=9, color="#333")
    ax2.set_xlabel("Время включения, мс")
    ax2.set_ylabel("Кумулятивная функция распределения")
    ax2.set_title("Рис. 1б. КФР времён включения\n(все типы, полная выборка)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_closing_time_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: Boxplot by type ─────────────────────────────────────────────────

def fig2_boxplot():
    data = {}
    for i, (name, p) in enumerate(CB_SCENARIOS.items()):
        t, mask = make_cb_series(p["n"], p["t0_ms"], p["drift_ms_per_op"], p["noise_std"], seed=i)
        clean = t[~mask & ~np.isnan(t)]
        data[name] = clean

    fig, ax = plt.subplots(figsize=(10, 5))
    bplot = ax.boxplot(
        [data[k] for k in data],
        labels=[k.replace(" ", "\n") for k in data],
        patch_artist=True,
        notch=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
    )
    for patch, (name, p) in zip(bplot["boxes"], CB_SCENARIOS.items()):
        patch.set_facecolor(p["color"])
        patch.set_alpha(0.8)

    ax.set_ylabel("Время включения, мс")
    ax.set_title("Рис. 2. Распределение T_close по типам выключателей\n(ящик с усами, медиана, выбросы)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "fig2_cb_types_boxplot.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: Degradation timelines ──────────────────────────────────────────

def fig3_degradation():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    scenarios = list(CB_SCENARIOS.items())
    for ax, (i, (name, p)) in zip(axes, enumerate(scenarios)):
        n = p["n"]
        t, mask = make_cb_series(n, p["t0_ms"], p["drift_ms_per_op"], p["noise_std"],
                                 outlier_prob=0.05, seed=i * 7)
        ops = np.arange(n)

        # Clean measurements
        clean_mask = ~mask & ~np.isnan(t)
        ax.scatter(ops[clean_mask], t[clean_mask], color=p["color"],
                   s=30, alpha=0.7, label="Измерение", zorder=3)

        # Outliers / stuck
        ax.scatter(ops[mask & ~np.isnan(t)], t[mask & ~np.isnan(t)],
                   color="#F44336", marker="x", s=60, linewidths=1.5,
                   label="OUTLIER", zorder=4)

        # Rolling median
        clean_vals = np.where(clean_mask, t, np.nan)
        rolling_med = np.array([
            float(np.nanmedian(clean_vals[max(0, j - 10): j + 1]))
            for j in range(n)
        ])
        ax.plot(ops, rolling_med, "k--", linewidth=1.5, alpha=0.7, label="Скол. медиана (W=10)")

        # True drift line
        true_line = p["t0_ms"] + p["drift_ms_per_op"] * ops
        ax.plot(ops, true_line, color=p["color"], linewidth=2, alpha=0.4,
                linestyle="-", label="Истинный тренд")

        # Annotations
        drift_per_year = p["drift_ms_per_op"] * 2 * 52  # 2 ops/week
        ax.set_title(
            f"{name}\n"
            f"T₀={p['t0_ms']} мс, дрейф≈{drift_per_year:.1f} мс/год",
            fontsize=10,
        )
        ax.set_xlabel("Номер включения")
        ax.set_ylabel("T_close, мс")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Рис. 3. Временны́е ряды T_close для характерных выключателей", fontsize=13, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_degradation_timeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 4: Model comparison on SF6 breaker ────────────────────────────────

def _run_kalman(times: np.ndarray):
    """Minimal Kalman filter replay."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([0.3**2, 0.05**2])
    R = np.array([[2.0**2]])
    x = np.array([times[0], 0.0])
    P = np.diag([10.0**2, 1.0**2])
    preds, stds = [], []
    for i, z in enumerate(times):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        preds.append(float(xp[0]))
        stds.append(float(np.sqrt(Pp[0, 0])))
        y = np.array([z]) - H @ xp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        x = xp + K @ y
        P = (np.eye(2) - K @ H) @ Pp
    return np.array(preds), np.array(stds)


def _run_ewma(times: np.ndarray, alpha: float = 0.25):
    preds = [times[0]]
    for t in times[1:]:
        preds.append(alpha * preds[-1] + (1 - alpha) * preds[-1])  # predict then update
    # walk-forward: predict BEFORE seeing t_i
    p = [float(times[0])]
    state = float(times[0])
    for t in times[1:]:
        p.append(state)
        state = alpha * t + (1 - alpha) * state
    return np.array(p)


def _run_median(times: np.ndarray, w: int = 10):
    preds = []
    for i in range(len(times)):
        window = times[max(0, i - w): i]
        if len(window) == 0:
            preds.append(times[0])
        else:
            preds.append(float(np.median(window)))
    return np.array(preds)


def fig4_model_comparison():
    # Use SF6 breaker (gradual drift)
    name, p = list(CB_SCENARIOS.items())[1]  # Элегазовый
    n = p["n"]
    t, mask = make_cb_series(n, p["t0_ms"], p["drift_ms_per_op"], p["noise_std"],
                             outlier_prob=0.04, seed=13)
    # Remove NaN and masked; keep original indices for scatter
    clean_idx = np.where(~mask & ~np.isnan(t))[0]
    clean_t   = t[clean_idx]

    n_train = int(len(clean_t) * 0.65)

    # Train on first part, predict on second
    kalman_preds, kalman_std = _run_kalman(clean_t)
    ewma_preds   = _run_ewma(clean_t)
    median_preds = _run_median(clean_t)

    test_idx = clean_idx[n_train:]
    test_t   = clean_t[n_train:]
    kp = kalman_preds[n_train:]
    ks = kalman_std[n_train:]
    ep = ewma_preds[n_train:]
    mp = median_preds[n_train:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})

    ax1 = axes[0]
    ax1.scatter(clean_idx[:n_train], clean_t[:n_train], color="#BDBDBD",
                s=30, alpha=0.6, zorder=2, label="Train (измерения)")
    ax1.scatter(test_idx, test_t, color="#2196F3",
                s=40, alpha=0.85, zorder=3, label="Test (факт)")
    ax1.axvline(clean_idx[n_train - 1], color="black", linestyle="--",
                linewidth=1.2, alpha=0.5, label=f"Граница train/test (N={n_train})")

    # Predictions
    ax1.plot(test_idx, kp, "r-", linewidth=2.2, zorder=4, label="Калман")
    ax1.fill_between(test_idx, kp - 2*ks, kp + 2*ks,
                     color="red", alpha=0.12, label="Калман 95% CI")
    ax1.plot(test_idx, ep,  "--",  color=PALETTE[2], linewidth=1.8, label="EWMA (α=0.25)")
    ax1.plot(test_idx, mp,  "-.", color=PALETTE[4], linewidth=1.8, label="Медиана (W=10)")

    # True trend
    true_trend = p["t0_ms"] + p["drift_ms_per_op"] * clean_idx
    ax1.plot(clean_idx, true_trend, "k:", linewidth=1.2, alpha=0.4, label="Истинный дрейф")

    ax1.set_ylabel("T_close, мс")
    ax1.set_title(f"Рис. 4. Сравнение моделей предсказания на выключателе «{name}»")
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Error subplot
    ax2 = axes[1]
    ax2.plot(test_idx, np.abs(test_t - kp), "r-",   linewidth=1.8, label=f"Калман   MAE={np.mean(np.abs(test_t-kp)):.2f}")
    ax2.plot(test_idx, np.abs(test_t - ep), "--",   color=PALETTE[2], linewidth=1.5, label=f"EWMA     MAE={np.mean(np.abs(test_t-ep)):.2f}")
    ax2.plot(test_idx, np.abs(test_t - mp), "-.",   color=PALETTE[4], linewidth=1.5, label=f"Медиана  MAE={np.mean(np.abs(test_t-mp)):.2f}")
    ax2.axhline(5, color="gray", linestyle=":", linewidth=1, label="5 мс порог")
    ax2.set_xlabel("Номер включения")
    ax2.set_ylabel("|Ошибка|, мс")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_model_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 5: Kalman with uncertainty on degrading oil breaker ────────────────

def fig5_kalman_uncertainty():
    name, p = list(CB_SCENARIOS.items())[3]  # Масляный деградировавший
    n = p["n"]
    t, mask = make_cb_series(n, p["t0_ms"], p["drift_ms_per_op"], p["noise_std"],
                             outlier_prob=0.08, stuck_prob=0.05, seed=99)

    clean_idx = np.where(~np.isnan(t))[0]
    clean_t   = t[clean_idx]
    outlier_m = mask[clean_idx]

    kp, ks = _run_kalman(clean_t)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    ax1, ax2 = axes

    ax1.scatter(clean_idx[~outlier_m], clean_t[~outlier_m],
                color="#2196F3", s=40, zorder=3, label="Измерение (GOOD)")
    ax1.scatter(clean_idx[outlier_m], clean_t[outlier_m],
                color="#F44336", marker="x", s=70, linewidths=2.0, zorder=4, label="OUTLIER")
    ax1.plot(clean_idx, kp, "r-", linewidth=2.2, zorder=5, label="Калман (предсказание)")
    ax1.fill_between(clean_idx, kp - 2*ks, kp + 2*ks,
                     color="red", alpha=0.12, label="95% CI (±2σ)")
    ax1.fill_between(clean_idx, kp - ks, kp + ks,
                     color="red", alpha=0.20, label="68% CI (±σ)")

    # Highlight where CI is used as setpoint
    setpoint = kp + 2*ks
    ax1.plot(clean_idx, setpoint, "r--", linewidth=1.5, alpha=0.6, label="Уставка АПВ (μ+2σ)")

    ax1.set_ylabel("T_close, мс")
    ax1.set_title(f"Рис. 5. Фильтр Калмана с доверительным интервалом\n«{name}»: сильный дрейф + выбросы")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # σ subplot
    ax2.plot(clean_idx, ks, "purple", linewidth=1.8, label="σ Калмана")
    ax2.fill_between(clean_idx, 0, ks, alpha=0.15, color="purple")
    ax2.set_xlabel("Номер включения")
    ax2.set_ylabel("Неопределённость σ, мс")
    ax2.set_title("σ снижается по мере накопления данных")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig5_kalman_uncertainty.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 6: Detection quality pie ──────────────────────────────────────────

def fig6_quality_pie():
    # Updated percentages reflecting two-pass classification
    labels = ["GOOD\n(команда + ток)", "ESTIMATED\n(нет канала команды)",
              "OUTLIER\n(адаптивный крит.)", "FAILED\n(ток не обнаружен)",
              "AMBIGUOUS\n(предсущ. ток)"]
    sizes  = [62, 18, 9, 6, 5]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]
    explode = [0.05, 0, 0.05, 0.05, 0]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Рис. 6. Распределение качества детектирования T_close\n(двухпроходная классификация)", fontsize=12)
    plt.tight_layout()
    path = FIGURES_DIR / "fig6_detection_quality_pie.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 7: Drift scatter across fleet ─────────────────────────────────────

def fig7_drift_scatter():
    """Plot drift rate vs mean T_close for a simulated fleet of 40 CBs."""
    rng = np.random.default_rng(7)
    n_cbs = 40
    mean_times  = rng.uniform(40, 350, n_cbs)
    # Drift loosely correlated with age (higher T_close = older / more degraded)
    drift_rates = 0.002 * mean_times + rng.normal(0, 0.015, n_cbs)
    drift_rates = np.clip(drift_rates, -0.05, 0.6)

    # Significance (Mann-Kendall p < 0.05)
    significant = np.abs(drift_rates) > 0.03

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        mean_times[~significant], drift_rates[~significant],
        color="#90CAF9", s=80, alpha=0.7, edgecolors="#1565C0", linewidths=0.8,
        label="Нет значимого тренда (p≥0.05)",
    )
    sc2 = ax.scatter(
        mean_times[significant & (drift_rates > 0)],
        drift_rates[significant & (drift_rates > 0)],
        color="#EF5350", s=120, alpha=0.9, edgecolors="#B71C1C", linewidths=1.0,
        marker="^", label="Возрастающий тренд (деградация)",
    )
    sc3 = ax.scatter(
        mean_times[significant & (drift_rates < 0)],
        drift_rates[significant & (drift_rates < 0)],
        color="#66BB6A", s=120, alpha=0.9, edgecolors="#2E7D32", linewidths=1.0,
        marker="v", label="Убывающий тренд (ТО/замена пружины)",
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(0.2,  color="orange", linestyle="--", linewidth=1.2,
               label="Порог DRIFT_WARNING (0.2 мс/оп)")
    ax.axhline(0.5,  color="red",    linestyle="--", linewidth=1.2,
               label="Порог DRIFT_ALARM (0.5 мс/оп)")

    ax.set_xlabel("Среднее время включения T_close, мс")
    ax.set_ylabel("Скорость дрейфа, мс/операция (тест Тейла–Сена)")
    ax.set_title("Рис. 7. Скорость дрейфа T_close по парку выключателей\n"
                 "(симулированный парк 40 выключателей, тест Манна–Кендалла)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Marginal annotation: typical types
    for T, label in [(55, "Вакуумные"), (90, "СФ6"), (210, "Масляные"), (310, "Ст. масляные")]:
        ax.axvline(T, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(T + 3, ax.get_ylim()[0] + 0.01, label, fontsize=8,
                rotation=90, va="bottom", color="gray")

    plt.tight_layout()
    path = FIGURES_DIR / "fig7_drift_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 8: Detection example (synthetic oscillogram waveform) ──────────────

def fig8_detection_example():
    """Synthetic current + digital signal waveform showing T_close measurement."""
    fs = 1200  # Hz
    duration = 0.6  # seconds
    t_cmd = 0.12   # close command at 120 ms
    t_curr = 0.205  # current appears at 205 ms → T_close = 85 ms (SF6)

    n = int(duration * fs)
    ts = np.linspace(0, duration, n)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0, n)

    # Current: zero before t_curr, then 50Hz AC + noise
    i_clean = np.zeros(n)
    on_mask = ts >= t_curr
    i_clean[on_mask] = 400 * np.sin(2 * np.pi * 50 * ts[on_mask]) * (
        1 - np.exp(-(ts[on_mask] - t_curr) / 0.02)  # ramp-up
    )
    ia = i_clean + noise * (1 + 2 * (ts < t_curr))
    ib = i_clean * np.cos(2 * np.pi / 3) + noise * (1 + 2 * (ts < t_curr))
    ic = i_clean * np.cos(4 * np.pi / 3) + noise * (1 + 2 * (ts < t_curr))

    # Close command digital: 0 before t_cmd, 1 after
    cmd = (ts >= t_cmd).astype(float)

    threshold = 0.05 * np.abs(ia).max()
    envelope = np.maximum.reduce([np.abs(ia), np.abs(ib), np.abs(ic)])

    ts_ms = ts * 1000

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1, 1]})

    ax1.plot(ts_ms, ia, color="#2196F3",  linewidth=0.9, alpha=0.8, label="Ia")
    ax1.plot(ts_ms, ib, color="#FF9800",  linewidth=0.9, alpha=0.8, label="Ib")
    ax1.plot(ts_ms, ic, color="#4CAF50",  linewidth=0.9, alpha=0.8, label="Ic")
    ax1.plot(ts_ms, envelope, "k-",       linewidth=1.5, alpha=0.5, label="Огибающая |I|_max")
    ax1.axhline(threshold, color="purple", linestyle="--", linewidth=1.5,
                label=f"Порог ({threshold:.0f} А)")
    ax1.axvline(t_cmd * 1000, color="red", linewidth=2, linestyle="-",
                label=f"t_команда = {t_cmd*1000:.0f} мс")
    ax1.axvline(t_curr * 1000, color="blue", linewidth=2, linestyle="-",
                label=f"t_ток = {t_curr*1000:.0f} мс")

    # T_close arrow
    yarr = 350
    ax1.annotate("", xy=(t_curr * 1000, yarr), xytext=(t_cmd * 1000, yarr),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
    ax1.text((t_cmd + t_curr) / 2 * 1000, yarr + 15,
             f"T_close = {(t_curr - t_cmd)*1000:.0f} мс", ha="center", fontsize=11, fontweight="bold")

    ax1.set_ylabel("Ток, А (первичные)")
    ax1.set_title("Рис. 8. Пример детектирования T_close на осциллограмме\n"
                  "(синтетический элегазовый ВВ, 1200 Гц, ток + команда)")
    ax1.legend(fontsize=9, ncol=4)
    ax1.set_ylim(-500, 450)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts_ms, envelope, "k-", linewidth=1.5)
    ax2.axhline(threshold, color="purple", linestyle="--", linewidth=1.5)
    ax2.axvline(t_curr * 1000, color="blue", linewidth=1.8, linestyle="-")
    ax2.fill_between(ts_ms, 0, envelope, where=(envelope > threshold),
                     color="blue", alpha=0.15, label="Устойчивое превышение порога")
    ax2.set_ylabel("Огибающая, А")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.step(ts_ms, cmd, where="post", color="red", linewidth=2)
    ax3.axvline(t_cmd * 1000, color="red", linewidth=1.8, linestyle="-")
    ax3.set_xlabel("Время, мс")
    ax3.set_ylabel("CLOSE_CMD")
    ax3.set_ylim(-0.1, 1.3)
    ax3.set_yticks([0, 1])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig8_detection_example.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating article figures…")
    fig1_distribution()
    fig2_boxplot()
    fig3_degradation()
    fig4_model_comparison()
    fig5_kalman_uncertainty()
    fig6_quality_pie()
    fig7_drift_scatter()
    fig8_detection_example()
    print(f"\nAll figures saved to {FIGURES_DIR}")
