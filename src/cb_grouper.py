"""
Circuit breaker identification and grouping.

Groups COMTRADE records that belong to the same physical circuit breaker,
then sorts each group by event timestamp (oldest first).

Grouping key = fingerprint(station_name, rec_dev_id, channel_names).
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .comtrade_parser import ComtradeRecord

logger = logging.getLogger(__name__)


class CircuitBreakerGroup:
    """A set of oscillograms from a single circuit breaker, sorted by time."""

    def __init__(self, fingerprint: str, records: list[ComtradeRecord]):
        self.fingerprint = fingerprint
        # Sort by start_time; put records with no timestamp at the end
        self.records: list[ComtradeRecord] = sorted(
            records, key=lambda r: (r.start_time is None, r.start_time or datetime.min)
        )

    def __repr__(self) -> str:
        n = len(self.records)
        first_station = self.records[0].station_name if self.records else "?"
        first_dev = self.records[0].rec_dev_id if self.records else "?"
        t_range = self._time_range()
        return (
            f"CBGroup(station={first_station!r}, dev={first_dev!r}, "
            f"n={n}, span={t_range})"
        )

    def _time_range(self) -> str:
        times = [r.start_time for r in self.records if r.start_time]
        if not times:
            return "unknown"
        t0, t1 = min(times), max(times)
        if t0 == t1:
            return str(t0.date())
        return f"{t0.date()} – {t1.date()}"

    def to_dataframe(self) -> pd.DataFrame:
        """Return metadata of all records in this group as a DataFrame."""
        rows = []
        for i, r in enumerate(self.records):
            rows.append({
                "index": i,
                "cfg_path": str(r.cfg_path),
                "station_name": r.station_name,
                "rec_dev_id": r.rec_dev_id,
                "start_time": r.start_time,
                "trigger_time": r.trigger_time,
                "sampling_rate": r.sampling_rate,
                "n_samples": r.n_samples,
                "n_analog": len(r.analog),
                "n_digital": len(r.digital),
            })
        return pd.DataFrame(rows)


def group_by_breaker(
    records: list[ComtradeRecord],
    min_records: int = 2,
) -> dict[str, CircuitBreakerGroup]:
    """
    Group ComtradeRecord objects by circuit breaker fingerprint.

    Args:
        records:     List of parsed COMTRADE records.
        min_records: Drop groups with fewer than this many records.

    Returns:
        Dict mapping fingerprint → CircuitBreakerGroup.
    """
    grouped: dict[str, list[ComtradeRecord]] = defaultdict(list)
    for rec in records:
        fp = rec.fingerprint()
        grouped[fp].append(rec)

    result: dict[str, CircuitBreakerGroup] = {}
    skipped = 0
    for fp, recs in grouped.items():
        if len(recs) < min_records:
            skipped += 1
            continue
        result[fp] = CircuitBreakerGroup(fp, recs)

    logger.info(
        f"Grouped {len(records)} records → {len(result)} circuit breakers "
        f"(skipped {skipped} groups with < {min_records} records)"
    )
    return result


def summary_dataframe(groups: dict[str, CircuitBreakerGroup]) -> pd.DataFrame:
    """Return a summary DataFrame of all groups."""
    rows = []
    for fp, grp in groups.items():
        r0 = grp.records[0]
        rows.append({
            "fingerprint": fp,
            "station_name": r0.station_name,
            "rec_dev_id": r0.rec_dev_id,
            "n_records": len(grp.records),
            "sampling_rate_hz": r0.sampling_rate,
            "n_analog_ch": len(r0.analog),
            "n_digital_ch": len(r0.digital),
            "analog_channels": ", ".join(ch.name for ch in r0.analog),
            "digital_channels": ", ".join(ch.name for ch in r0.digital),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_records", ascending=False).reset_index(drop=True)
    return df
