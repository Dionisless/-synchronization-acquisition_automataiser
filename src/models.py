"""
Prediction models for circuit breaker closing time estimation.

Each model implements predict() → next estimated closing time (ms).
After an actual measurement arrives, call update(measurement).

Models:
  - MedianModel          : Rolling median of last N measurements
  - EWMAModel            : Exponentially weighted moving average
  - KalmanModel          : Kalman filter (handles noise + mechanical drift)
  - TheilSenModel        : Robust linear regression (Theil-Sen estimator)
  - EnsembleModel        : Weighted combination of above models

The Kalman filter is the primary recommendation for production use:
  - State: [T_close, drift_per_op] — captures slow mechanical aging
  - Process noise: tuned to breaker mechanical drift rate (~0.1 ms per 1000 ops)
  - Measurement noise: accounts for detection algorithm uncertainty (~1-3 ms)
  - Naturally handles outliers via innovation gating
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
from scipy.stats import theilslopes

logger = logging.getLogger(__name__)


# ─── Outlier detection (Hampel identifier) ───────────────────────────────────

def hampel_filter(values: np.ndarray, window: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """
    Hampel identifier: flag outliers using median absolute deviation in
    a sliding window. Returns boolean mask (True = outlier).
    """
    n = len(values)
    outliers = np.zeros(n, dtype=bool)
    k = 1.4826  # consistency factor for normal distribution
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = values[lo:hi]
        med = np.median(window_vals)
        mad = np.median(np.abs(window_vals - med))
        if mad == 0:
            continue
        if abs(values[i] - med) > n_sigma * k * mad:
            outliers[i] = True
    return outliers


def iqr_outlier_mask(values: np.ndarray, iqr_multiplier: float = 2.0) -> np.ndarray:
    """Flag outliers using IQR method."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo = q1 - iqr_multiplier * iqr
    hi = q3 + iqr_multiplier * iqr
    return (values < lo) | (values > hi)


# ─── Base class ──────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """Abstract base class for closing time prediction models."""

    def __init__(self, name: str):
        self.name = name
        self._observations: list[float] = []

    def update(self, measurement: float) -> None:
        """Incorporate a new measurement and update internal state."""
        self._observations.append(measurement)
        self._update(measurement)

    @abstractmethod
    def _update(self, measurement: float) -> None:
        pass

    @abstractmethod
    def predict(self) -> Optional[float]:
        """Return predicted closing time (ms) for the NEXT operation."""
        pass

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        self._observations = []


# ─── Rolling Median ───────────────────────────────────────────────────────────

class MedianModel(BaseModel):
    """
    Rolling median of the last `window` measurements.
    Robust to outliers by design; does not capture trend.
    """

    def __init__(self, window: int = 10):
        super().__init__("Median")
        self.window = window
        self._buffer: deque[float] = deque(maxlen=window)

    def _update(self, measurement: float) -> None:
        self._buffer.append(measurement)

    def predict(self) -> Optional[float]:
        if not self._buffer:
            return None
        return float(np.median(list(self._buffer)))

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()


# ─── EWMA ─────────────────────────────────────────────────────────────────────

class EWMAModel(BaseModel):
    """
    Exponentially Weighted Moving Average.

    T̂_{n+1} = α * T_n + (1-α) * T̂_n

    Higher alpha → more responsive to recent changes.
    Lower alpha → smoother, less sensitive to individual outliers.
    """

    def __init__(self, alpha: float = 0.25):
        super().__init__("EWMA")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._state: Optional[float] = None

    def _update(self, measurement: float) -> None:
        if self._state is None:
            self._state = measurement
        else:
            self._state = self.alpha * measurement + (1 - self.alpha) * self._state

    def predict(self) -> Optional[float]:
        return self._state

    def reset(self) -> None:
        super().reset()
        self._state = None


# ─── Kalman Filter ────────────────────────────────────────────────────────────

class KalmanModel(BaseModel):
    """
    Kalman filter for circuit breaker closing time estimation.

    State vector: x = [T_close (ms), drift (ms/operation)]
    Models gradual mechanical aging as a constant-velocity drift process.

    Innovation gating: measurements more than `gate_sigma` standard deviations
    from the prediction are flagged as outliers and receive reduced weight.
    This handles stuck breakers, CT anomalies, etc.
    """

    def __init__(
        self,
        initial_time: Optional[float] = None,
        process_noise_time_ms: float = 0.3,
        process_noise_drift_ms: float = 0.05,
        measurement_noise_ms: float = 2.0,
        initial_uncertainty_ms: float = 10.0,
        gate_sigma: float = 4.0,
    ):
        super().__init__("Kalman")
        self._initialized = False
        self._initial_time = initial_time

        # Process noise covariance Q
        self.Q = np.diag([process_noise_time_ms**2, process_noise_drift_ms**2])
        # Measurement noise variance R
        self.R = np.array([[measurement_noise_ms**2]])
        # State transition (constant-velocity model; dt=1 operation)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        # Observation matrix
        self.H = np.array([[1.0, 0.0]])
        # Innovation gate threshold (Mahalanobis distance)
        self.gate_sigma = gate_sigma
        # Initial covariance
        self._P0 = np.diag([initial_uncertainty_ms**2, (initial_uncertainty_ms * 0.1)**2])

        # State
        self.x: np.ndarray = np.zeros(2)
        self.P: np.ndarray = self._P0.copy()

        self._innovations: list[float] = []   # for diagnostics
        self._gated: list[bool] = []

        if initial_time is not None:
            self._init_state(initial_time)

    def _init_state(self, first_measurement: float) -> None:
        self.x = np.array([first_measurement, 0.0])
        self.P = self._P0.copy()
        self._initialized = True

    def _update(self, measurement: float) -> None:
        if not self._initialized:
            self._init_state(measurement)
            return

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Innovation
        z = np.array([measurement])
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        # Innovation gating (outlier detection)
        mahal_sq = float(y.T @ np.linalg.inv(S) @ y)
        is_outlier = mahal_sq > self.gate_sigma**2
        self._innovations.append(float(y[0]))
        self._gated.append(is_outlier)

        if is_outlier:
            # Soft rejection: reduce measurement weight by factor 10
            R_gated = self.R * 100.0
            S = self.H @ P_pred @ self.H.T + R_gated
            logger.debug(
                f"Kalman gated outlier: measurement={measurement:.2f} ms, "
                f"prediction={x_pred[0]:.2f} ms, innovation={y[0]:.2f} ms"
            )
        else:
            R_gated = self.R

        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

    def predict(self) -> Optional[float]:
        if not self._initialized:
            return self._initial_time
        x_pred = self.F @ self.x
        return float(x_pred[0])

    def predict_with_uncertainty(self) -> tuple[float, float]:
        """Return (predicted_ms, std_ms) for 95% confidence interval."""
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        pred_ms = float(x_pred[0])
        pred_std = float(np.sqrt(P_pred[0, 0]))
        return pred_ms, pred_std

    def current_drift_rate(self) -> float:
        """Estimated drift per operation (ms/operation)."""
        return float(self.x[1]) if self._initialized else 0.0

    def reset(self) -> None:
        super().reset()
        self._initialized = False
        self.x = np.zeros(2)
        self.P = self._P0.copy()
        self._innovations = []
        self._gated = []


# ─── Theil-Sen Regression ─────────────────────────────────────────────────────

class TheilSenModel(BaseModel):
    """
    Robust linear regression using Theil-Sen estimator on the most recent
    `window` observations. Predicts next value by extrapolating the trend.

    Resistant to ~29% contamination by outliers.
    Captures monotonic drift (breaker aging).
    """

    def __init__(self, window: int = 20):
        super().__init__("TheilSen")
        self.window = window
        self._buffer: deque[float] = deque(maxlen=window)
        self._slope: Optional[float] = None
        self._intercept: Optional[float] = None

    def _fit(self) -> None:
        vals = np.array(list(self._buffer))
        x = np.arange(len(vals), dtype=float)
        if len(vals) < 3:
            self._slope = 0.0
            self._intercept = float(np.median(vals))
        else:
            result = theilslopes(vals, x)
            self._slope = float(result.slope)
            self._intercept = float(result.intercept)

    def _update(self, measurement: float) -> None:
        self._buffer.append(measurement)
        self._fit()

    def predict(self) -> Optional[float]:
        if self._intercept is None or self._slope is None:
            return None
        n = len(self._buffer)
        return self._intercept + self._slope * n  # predict index n (next)

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._slope = None
        self._intercept = None


# ─── Ensemble ─────────────────────────────────────────────────────────────────

class EnsembleModel(BaseModel):
    """
    Weighted median ensemble of multiple sub-models.
    Automatically falls back to available predictions.
    """

    def __init__(
        self,
        models: Optional[list[BaseModel]] = None,
        weights: Optional[list[float]] = None,
    ):
        super().__init__("Ensemble")
        if models is None:
            models = [
                KalmanModel(measurement_noise_ms=2.0),
                MedianModel(window=10),
                EWMAModel(alpha=0.25),
                TheilSenModel(window=20),
            ]
        self.sub_models = models
        self.weights = weights or [1.0] * len(models)

    def _update(self, measurement: float) -> None:
        for m in self.sub_models:
            m.update(measurement)

    def predict(self) -> Optional[float]:
        preds = []
        wts = []
        for m, w in zip(self.sub_models, self.weights):
            p = m.predict()
            if p is not None:
                preds.append(p)
                wts.append(w)
        if not preds:
            return None
        wts_arr = np.array(wts)
        preds_arr = np.array(preds)
        return float(np.average(preds_arr, weights=wts_arr))

    def reset(self) -> None:
        super().reset()
        for m in self.sub_models:
            m.reset()


# ─── Convenience factory ──────────────────────────────────────────────────────

def build_models(cfg: Optional[dict] = None) -> dict[str, BaseModel]:
    """Build all models from config dict. Returns name → model mapping."""
    if cfg is None:
        mc = {}
    else:
        mc = cfg.get("models", {})

    kalman_cfg = mc.get("kalman", {})
    return {
        "Median": MedianModel(window=mc.get("median_window", 10)),
        "EWMA": EWMAModel(alpha=mc.get("ewma_alpha", 0.25)),
        "Kalman": KalmanModel(
            process_noise_time_ms=kalman_cfg.get("process_noise_time_ms", 0.3),
            process_noise_drift_ms=kalman_cfg.get("process_noise_drift_ms", 0.05),
            measurement_noise_ms=kalman_cfg.get("measurement_noise_ms", 2.0),
            initial_uncertainty_ms=kalman_cfg.get("initial_uncertainty_ms", 10.0),
        ),
        "TheilSen": TheilSenModel(window=mc.get("regression_window", 20)),
        "Ensemble": EnsembleModel(),
    }
