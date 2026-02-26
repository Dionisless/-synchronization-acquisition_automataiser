"""
Circuit breaker closing time detector.

Closing time T_close = t_current_appears - t_close_command

Strategy (tried in order):
  1. Digital close command channel + current rise
  2. Breaker position change (52A/52B) + current rise
  3. Current rise from start of record (reference = trigger time)

Edge cases handled:
  - Stuck breaker: no current detected within search window → quality=FAILED
  - CT noise: current must exceed threshold for N consecutive samples
  - Pre-existing current (breaker already closed at trigger): quality=AMBIGUOUS
  - Multiple current rises (contact bounce): use first sustained rise
  - No command signal: fall back to record start as reference
  - Outliers flagged but not discarded; quality scoring communicates confidence
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import yaml

from .comtrade_parser import ComtradeRecord, AnalogChannel, DigitalChannel

logger = logging.getLogger(__name__)


class DetectionQuality(str, Enum):
    GOOD = "good"             # Command + current both detected, plausible T_close
    ESTIMATED = "estimated"   # No command channel; used record start as reference
    AMBIGUOUS = "ambiguous"   # Current existed before command; edge case
    OUTLIER = "outlier"       # T_close outside plausible range (stuck / noise)
    FAILED = "failed"         # Could not detect current rise at all


@dataclass
class ClosingResult:
    """Result of closing time detection for a single oscillogram."""
    t_command_s: Optional[float]    # seconds from record start; None if not found
    t_current_s: Optional[float]    # seconds from record start; None if not found
    t_close_ms: Optional[float]     # = (t_current - t_command) * 1000; main value
    quality: DetectionQuality
    command_channel: Optional[str]  # name of the channel used as command
    current_channel: Optional[str]  # name of the channel used for current
    current_threshold_a: float      # actual threshold used (primary amperes)
    note: str = ""


# ─── Channel pattern matching ───────────────────────────────────────────────

_CLOSE_CMD_PATTERNS = [
    r"clos", r"\bkv\b", r"кв", r"52c", r"vcmd", r"\bcmd\b",
    r"comm", r"на\s*вкл", r"\bвкл\b", r"close\s*cmd", r"close_cmd",
]

_BREAKER_POS_PATTERNS = [
    r"52a", r"52b", r"\bpv\b", r"пв", r"posit", r"положен", r"breaker\s*pos",
]

_CURRENT_PATTERNS = [
    r"^i[abc]$", r"^i[abc][_\s]", r"^i[abc][0-9]?$",
    r"ток", r"current", r"^ил", r"^il",
]

_CURRENT_UNITS = {"a", "ka", "ma", "amps", "amp", "а", "ка", "ма"}


def _matches(name: str, patterns: list[str]) -> bool:
    nl = name.lower().strip()
    return any(re.search(p, nl) for p in patterns)


def _find_close_cmd_channel(rec: ComtradeRecord) -> Optional[DigitalChannel]:
    for ch in rec.digital:
        if _matches(ch.name, _CLOSE_CMD_PATTERNS):
            return ch
    return None


def _find_breaker_pos_channel(rec: ComtradeRecord) -> Optional[DigitalChannel]:
    for ch in rec.digital:
        if _matches(ch.name, _BREAKER_POS_PATTERNS):
            return ch
    return None


def _find_current_channels(rec: ComtradeRecord) -> list[AnalogChannel]:
    """Find all current analog channels."""
    channels = []
    for ch in rec.analog:
        unit_match = ch.unit.lower().strip() in _CURRENT_UNITS
        name_match = _matches(ch.name, _CURRENT_PATTERNS)
        if unit_match or name_match:
            channels.append(ch)
    return channels


# ─── Signal processing helpers ───────────────────────────────────────────────

def _rising_edge(signal: np.ndarray, threshold: float = 0.5) -> Optional[int]:
    """
    Find index of first rising edge (0→1) in a digital-like signal.
    Returns None if no edge found.
    """
    above = signal >= threshold
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            return i
    return None


def _first_sustained_rise(
    current_abs: np.ndarray,
    threshold: float,
    start_idx: int,
    end_idx: int,
    min_sustain: int = 3,
) -> Optional[int]:
    """
    Find the first index ≥ start_idx (up to end_idx) where |current| stays
    above `threshold` for at least `min_sustain` consecutive samples.
    Returns sample index or None.
    """
    n = len(current_abs)
    end_idx = min(end_idx, n - min_sustain)
    i = start_idx
    while i < end_idx:
        if current_abs[i] >= threshold:
            # Check that it stays above for min_sustain samples
            if all(current_abs[i : i + min_sustain] >= threshold):
                return i
            else:
                # Skip past false alarm
                i += 1
        else:
            i += 1
    return None


def _compute_current_threshold(
    current_channels: list[AnalogChannel],
    fraction: float,
    abs_fallback: float,
) -> float:
    """
    Compute detection threshold from the maximum current in the file.
    Uses fraction of file max; falls back to abs_fallback if max is too small.
    """
    if not current_channels:
        return abs_fallback
    file_max = max(np.abs(ch.data).max() for ch in current_channels)
    threshold = fraction * file_max
    return max(threshold, abs_fallback * 0.1)  # at least 10% of fallback


# ─── Main detector ───────────────────────────────────────────────────────────

def detect_closing_time(
    rec: ComtradeRecord,
    cfg: Optional[dict] = None,
) -> ClosingResult:
    """
    Detect circuit breaker closing time from a COMTRADE record.

    Returns a ClosingResult with the closing time in milliseconds and a
    quality indicator.
    """
    if cfg is None:
        cfg = _default_cfg()

    det = cfg["closing_detector"]
    current_frac = det.get("current_threshold_fraction", 0.05)
    current_abs = det.get("current_threshold_abs_a", 10.0)
    min_sustain = det.get("min_sustain_samples", 3)
    max_ms = det.get("max_closing_time_ms", 350.0)
    min_ms = det.get("min_closing_time_ms", 10.0)
    search_ms = det.get("search_window_ms", 500.0)

    ts = rec.timestamps                # seconds
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else (1.0 / rec.sampling_rate)
    search_samples = int(search_ms / 1000.0 / dt)

    # --- Find current channels ---
    current_chs = _find_current_channels(rec)
    threshold_a = _compute_current_threshold(current_chs, current_frac, current_abs)

    if not current_chs:
        return ClosingResult(
            t_command_s=None, t_current_s=None, t_close_ms=None,
            quality=DetectionQuality.FAILED,
            command_channel=None, current_channel=None,
            current_threshold_a=threshold_a,
            note="No current channels found in record",
        )

    # Envelope: max absolute current across all phases
    envelope = np.stack([np.abs(ch.data) for ch in current_chs]).max(axis=0)
    current_ch_names = ", ".join(ch.name for ch in current_chs)

    # Check if current existed BEFORE any possible command (pre-existing load)
    pre_current = envelope[:min(20, len(envelope))].mean()
    pre_existing = pre_current > threshold_a

    # ── Strategy 1: digital CLOSE command ────────────────────────────────
    cmd_ch = _find_close_cmd_channel(rec)
    t_cmd_s: Optional[float] = None
    cmd_name: Optional[str] = None
    cmd_idx: Optional[int] = None

    if cmd_ch is not None:
        edge_idx = _rising_edge(cmd_ch.data)
        if edge_idx is not None:
            t_cmd_s = float(ts[edge_idx])
            cmd_name = cmd_ch.name
            cmd_idx = edge_idx

    # ── Strategy 2: breaker position change ──────────────────────────────
    if cmd_idx is None:
        pos_ch = _find_breaker_pos_channel(rec)
        if pos_ch is not None:
            # 52A goes 1→0 when breaker closes (auxiliary contact "a")
            pos_inv = 1 - pos_ch.data
            edge_idx = _rising_edge(pos_inv)
            if edge_idx is not None:
                t_cmd_s = float(ts[edge_idx])
                cmd_name = f"{pos_ch.name} (pos)"
                cmd_idx = edge_idx

    # ── Strategy 3: start of record as reference ─────────────────────────
    using_record_start = False
    if cmd_idx is None:
        cmd_idx = 0
        t_cmd_s = float(ts[0])
        cmd_name = None
        using_record_start = True

    # ── Detect current rise after command ─────────────────────────────────
    search_end = min(cmd_idx + search_samples, len(envelope) - 1)
    rise_idx = _first_sustained_rise(envelope, threshold_a, cmd_idx, search_end, min_sustain)

    if rise_idx is None:
        # Stuck breaker or no operation recorded
        return ClosingResult(
            t_command_s=t_cmd_s, t_current_s=None, t_close_ms=None,
            quality=DetectionQuality.FAILED,
            command_channel=cmd_name, current_channel=current_ch_names,
            current_threshold_a=threshold_a,
            note="No current rise detected after command — possible stuck breaker",
        )

    t_current_s = float(ts[rise_idx])
    t_close_ms = (t_current_s - t_cmd_s) * 1000.0

    # ── Quality assessment ────────────────────────────────────────────────
    note = ""
    if using_record_start:
        quality = DetectionQuality.ESTIMATED
        note = "No command channel; record start used as reference"
    elif pre_existing and not using_record_start:
        quality = DetectionQuality.AMBIGUOUS
        note = "Current detected before command; breaker may have been already closing"
    elif t_close_ms < min_ms:
        quality = DetectionQuality.OUTLIER
        note = f"T_close={t_close_ms:.1f} ms < minimum {min_ms} ms (noise?)"
    elif t_close_ms > max_ms:
        quality = DetectionQuality.OUTLIER
        note = f"T_close={t_close_ms:.1f} ms > maximum {max_ms} ms (stuck?)"
    else:
        quality = DetectionQuality.GOOD

    return ClosingResult(
        t_command_s=t_cmd_s,
        t_current_s=t_current_s,
        t_close_ms=t_close_ms,
        quality=quality,
        command_channel=cmd_name,
        current_channel=current_ch_names,
        current_threshold_a=threshold_a,
        note=note,
    )


def _default_cfg() -> dict:
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "closing_detector": {
                "current_threshold_fraction": 0.05,
                "current_threshold_abs_a": 10.0,
                "min_sustain_samples": 3,
                "max_closing_time_ms": 350.0,
                "min_closing_time_ms": 10.0,
                "search_window_ms": 500.0,
            }
        }


def batch_detect(
    records: list[ComtradeRecord],
    cfg: Optional[dict] = None,
    verbose: bool = False,
) -> list[ClosingResult]:
    """Run closing time detection on a list of ComtradeRecord objects."""
    from tqdm import tqdm
    iterable = tqdm(records, desc="Detecting closing times") if verbose else records
    return [detect_closing_time(r, cfg) for r in iterable]
