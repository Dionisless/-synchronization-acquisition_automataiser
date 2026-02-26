"""
COMTRADE parser wrapper (IEEE C37.111).

Wraps the `comtrade` library to return structured records with:
  - Metadata: station, device, start_time, trigger_time, sampling_rate
  - Analog channels: name, unit, phase, data array
  - Digital channels: name, normal_state, data array
  - Timestamps array (seconds from trigger)

Handles both ASCII and BINARY dat formats.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import comtrade

logger = logging.getLogger(__name__)


@dataclass
class AnalogChannel:
    index: int
    name: str
    phase: str
    unit: str        # 'A', 'kV', 'V', etc.
    primary: float   # primary ratio
    secondary: float # secondary ratio
    ratio: str       # 'P' or 'S' (use primary or secondary)
    data: np.ndarray


@dataclass
class DigitalChannel:
    index: int
    name: str
    phase: str
    normal: int      # normal state (0 or 1)
    data: np.ndarray


@dataclass
class ComtradeRecord:
    """Parsed COMTRADE oscillogram."""
    cfg_path: Path
    station_name: str
    rec_dev_id: str
    start_time: Optional[datetime]
    trigger_time: Optional[datetime]
    sampling_rate: float          # Hz (first/primary rate)
    line_frequency: float         # Hz (50 or 60)
    timestamps: np.ndarray        # seconds relative to trigger
    analog: list[AnalogChannel]
    digital: list[DigitalChannel]
    n_samples: int

    def get_analog(self, name: str) -> Optional[AnalogChannel]:
        """Case-insensitive channel lookup."""
        nl = name.lower()
        for ch in self.analog:
            if ch.name.lower() == nl:
                return ch
        return None

    def get_digital(self, name: str) -> Optional[DigitalChannel]:
        nl = name.lower()
        for ch in self.digital:
            if ch.name.lower() == nl:
                return ch
        return None

    def primary_currents(self) -> list[AnalogChannel]:
        """Return analog channels whose unit is amperes."""
        current_units = {"a", "ka", "ma", "amps", "amp"}
        return [
            ch for ch in self.analog
            if ch.unit.lower().strip() in current_units
        ]

    def primary_voltages(self) -> list[AnalogChannel]:
        voltage_units = {"v", "kv", "mv", "volts"}
        return [
            ch for ch in self.analog
            if ch.unit.lower().strip() in voltage_units
        ]

    def fingerprint(self) -> str:
        """
        Unique identifier for the circuit breaker / bay this oscillogram
        belongs to. Used for grouping.
        """
        analog_names = tuple(sorted(ch.name.lower() for ch in self.analog))
        digital_names = tuple(sorted(ch.name.lower() for ch in self.digital))
        return f"{self.station_name}|{self.rec_dev_id}|{analog_names}|{digital_names}"


def parse_comtrade(cfg_path: Path) -> Optional[ComtradeRecord]:
    """
    Parse a COMTRADE pair (.cfg + .dat) and return a ComtradeRecord.
    Returns None on parse failure.
    """
    try:
        rec = comtrade.load(str(cfg_path))
    except Exception as exc:
        logger.warning(f"Failed to parse {cfg_path}: {exc}")
        return None

    # --- Timestamps ---
    try:
        ts = np.array(rec.time)  # seconds relative to start of record
    except Exception:
        ts = np.arange(rec.total_samples) / (rec.cfg.sample_rates[0][0] or 1.0)

    n = len(ts)

    # --- Sampling rate ---
    try:
        samp_rate = float(rec.cfg.sample_rates[0][0])
    except Exception:
        samp_rate = 1.0 / (ts[1] - ts[0]) if len(ts) > 1 else 0.0

    # --- Line frequency ---
    try:
        lf = float(rec.cfg.frequency)
    except Exception:
        lf = 50.0

    # --- Start / trigger times ---
    try:
        start_dt = rec.start_timestamp
        trig_dt = rec.trigger_timestamp
    except Exception:
        start_dt = None
        trig_dt = None

    # --- Analog channels ---
    analog_channels: list[AnalogChannel] = []
    for i, ch_cfg in enumerate(rec.cfg.analog_channels):
        try:
            raw = np.array(rec.analog[i])
        except Exception:
            raw = np.zeros(n)

        # Apply scaling: y = a*x + b
        a = float(ch_cfg.a) if ch_cfg.a is not None else 1.0
        b = float(ch_cfg.b) if ch_cfg.b is not None else 0.0
        data = a * raw + b

        primary = float(ch_cfg.primary) if ch_cfg.primary else 1.0
        secondary = float(ch_cfg.secondary) if ch_cfg.secondary else 1.0
        ratio_flag = ch_cfg.pors.upper() if ch_cfg.pors else "P"

        # Convert to primary values if data is in secondary
        if ratio_flag == "S" and secondary != 0:
            data = data * (primary / secondary)

        analog_channels.append(AnalogChannel(
            index=i,
            name=ch_cfg.id.strip() if ch_cfg.id else f"A{i+1}",
            phase=ch_cfg.ph.strip() if ch_cfg.ph else "",
            unit=ch_cfg.uu.strip() if ch_cfg.uu else "",
            primary=primary,
            secondary=secondary,
            ratio=ratio_flag,
            data=data,
        ))

    # --- Digital channels ---
    digital_channels: list[DigitalChannel] = []
    for i, ch_cfg in enumerate(rec.cfg.status_channels):
        try:
            raw = np.array(rec.status[i])
        except Exception:
            raw = np.zeros(n, dtype=int)

        digital_channels.append(DigitalChannel(
            index=i,
            name=ch_cfg.id.strip() if ch_cfg.id else f"D{i+1}",
            phase=ch_cfg.ph.strip() if ch_cfg.ph else "",
            normal=int(ch_cfg.y) if ch_cfg.y is not None else 0,
            data=raw.astype(int),
        ))

    station = (rec.cfg.station_name or "").strip()
    dev_id = (rec.cfg.rec_dev_id or "").strip()

    return ComtradeRecord(
        cfg_path=cfg_path,
        station_name=station,
        rec_dev_id=dev_id,
        start_time=start_dt,
        trigger_time=trig_dt,
        sampling_rate=samp_rate,
        line_frequency=lf,
        timestamps=ts,
        analog=analog_channels,
        digital=digital_channels,
        n_samples=n,
    )


def batch_parse(cfg_paths: list[Path], verbose: bool = False) -> list[ComtradeRecord]:
    """Parse multiple COMTRADE files; skip failures."""
    from tqdm import tqdm
    records = []
    iterable = tqdm(cfg_paths, desc="Parsing COMTRADE") if verbose else cfg_paths
    for p in iterable:
        rec = parse_comtrade(p)
        if rec is not None:
            records.append(rec)
    logger.info(f"Parsed {len(records)}/{len(cfg_paths)} records successfully")
    return records
