"""
Unlabeled data analysis: process a chunk of 1000 oscillograms from the
large unlabeled OscGrid archives.

Goals:
  1. Attempt circuit breaker grouping by COMTRADE fingerprint
  2. If grouping works → run full detection + model evaluation on top groups
  3. If grouping fails (all unique) → assess detection quality only and
     report that CB-level modeling is not feasible without labels

Usage:
    python scripts/analyze_unlabeled.py [--n-files 1000] [--data-dir data/raw]
                                        [--output-dir results/unlabeled]
                                        [--skip-download]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.download import download_file, extract_7z, find_comtrade_pairs, load_config
from src.comtrade_parser import batch_parse
from src.cb_grouper import group_by_breaker, summary_dataframe
from src.closing_detector import (
    detect_closing_time, DetectionQuality, reclassify_group_outliers
)
from src.evaluation import (
    evaluate_all, aggregate_metrics, detect_drift, plot_model_comparison
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("unlabeled_analysis")

# Unlabeled archive (smaller one first: 2.26 GB)
UNLABELED_ARCHIVE = {
    "file_id": 56314376,
    "name": "unlabeled_50_1200.7z",
    "size_gb": 2.26,
}


def parse_args():
    p = argparse.ArgumentParser(description="Unlabeled data chunk analysis")
    p.add_argument("--n-files", type=int, default=1000,
                   help="Number of COMTRADE files to process (default: 1000)")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="results/unlabeled")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-download", action="store_true",
                   help="Use already-downloaded data if available")
    p.add_argument("--min-group-size", type=int, default=5,
                   help="Min records per CB group to include in modeling")
    return p.parse_args()


def download_unlabeled(cfg: dict, data_dir: Path) -> Path:
    """Download the unlabeled archive. Returns extracted directory."""
    base_url = cfg["dataset"]["figshare_base_url"]
    archives_dir = data_dir / "archives"
    extracted_dir = data_dir / "extracted" / "unlabeled_50_1200"
    archives_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base_url}{UNLABELED_ARCHIVE['file_id']}"
    archive_path = archives_dir / UNLABELED_ARCHIVE["name"]

    logger.info(f"Downloading {UNLABELED_ARCHIVE['name']} ({UNLABELED_ARCHIVE['size_gb']} GB)…")
    logger.info("This may take a while on slow connections. Use --skip-download if already downloaded.")

    ok = download_file(url, archive_path)
    if not ok:
        raise RuntimeError(f"Failed to download {UNLABELED_ARCHIVE['name']}")

    ok = extract_7z(archive_path, extracted_dir)
    if not ok:
        raise RuntimeError(f"Failed to extract {UNLABELED_ARCHIVE['name']}")

    return extracted_dir


def assess_grouping_quality(groups: dict) -> dict:
    """
    Assess whether the fingerprint-based grouping produces meaningful CB groups.

    Returns a dict with grouping quality metrics.
    """
    n_total_records = sum(len(g.records) for g in groups.values())
    n_groups = len(groups)
    sizes = [len(g.records) for g in groups.values()]
    singleton_count = sum(1 for s in sizes if s == 1)
    multi_record_groups = sum(1 for s in sizes if s > 1)

    if not sizes:
        return {"groupable": False, "reason": "No groups formed"}

    # If >80% of groups are singletons, grouping is effectively impossible
    singleton_fraction = singleton_count / n_groups
    groupable = singleton_fraction < 0.80 and multi_record_groups >= 3

    return {
        "groupable": groupable,
        "n_total_records": n_total_records,
        "n_groups": n_groups,
        "singleton_fraction": round(singleton_fraction, 3),
        "multi_record_groups": multi_record_groups,
        "max_group_size": max(sizes),
        "median_group_size": float(np.median(sizes)),
        "reason": (
            "Fingerprint-based grouping successful" if groupable
            else (
                f"Most groups are singletons ({singleton_fraction:.0%}) — "
                "unique fingerprints per oscillogram; likely no repeated "
                "per-CB structure in unlabeled data. Model evaluation skipped."
            )
        ),
    }


def detection_quality_report(det_results: list, cfg: dict) -> dict:
    """Compute detection quality statistics from a list of ClosingResult."""
    total = len(det_results)
    quality_counts = Counter(r.quality.value for r in det_results)
    valid = [r for r in det_results if r.t_close_ms is not None
             and r.quality in (DetectionQuality.GOOD, DetectionQuality.ESTIMATED)]
    valid_times = [r.t_close_ms for r in valid]

    report = {
        "total_records": total,
        "quality_distribution": dict(quality_counts),
        "valid_detections": len(valid),
        "valid_fraction": round(len(valid) / total, 3) if total else 0,
    }
    if valid_times:
        arr = np.array(valid_times)
        report["closing_time_stats"] = {
            "mean_ms": round(float(arr.mean()), 2),
            "std_ms": round(float(arr.std()), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p5_ms": round(float(np.percentile(arr, 5)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "min_ms": round(float(arr.min()), 2),
            "max_ms": round(float(arr.max()), 2),
        }
        # Breaker type classification heuristic
        # Vacuum: 30-80 ms, SF6: 50-150 ms, Oil: 80-300 ms, Old oil: >300 ms
        def classify_type(t):
            if t < 80:
                return "vacuum/fast_SF6"
            elif t < 200:
                return "SF6/oil"
            elif t < 500:
                return "oil"
            else:
                return "old_oil_or_degraded"

        type_counts = Counter(classify_type(t) for t in valid_times)
        report["breaker_type_heuristic"] = dict(type_counts)

    return report


def plot_detection_distribution(valid_times: list[float], out_dir: Path) -> None:
    """Plot histogram of detected closing times from unlabeled data."""
    if not valid_times:
        return

    arr = np.array(valid_times)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, min(arr.max(), 1000), 60)
    ax1.hist(arr, bins=bins, color="#2196F3", alpha=0.8, edgecolor="white")
    for thresh, label, color in [(80, "Vacuum/SF6 boundary", "#FF9800"),
                                  (200, "SF6/Oil boundary", "#F44336")]:
        ax1.axvline(thresh, color=color, linestyle="--", linewidth=1.5, label=label)
    ax1.set_xlabel("Closing time (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Distribution of Closing Times\n(n={len(arr)} unlabeled oscillograms)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # CDF
    ax2 = axes[1]
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax2.plot(sorted_arr, cdf, color="#2196F3", linewidth=2)
    for p, label in [(0.5, "P50"), (0.9, "P90"), (0.95, "P95")]:
        val = float(np.percentile(arr, p * 100))
        ax2.axhline(p, color="gray", linestyle=":", linewidth=1)
        ax2.axvline(val, color="gray", linestyle=":", linewidth=1)
        ax2.text(val + 5, p - 0.04, f"{label}={val:.0f} ms", fontsize=9)
    ax2.set_xlabel("Closing time (ms)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative Distribution of Closing Times")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "unlabeled_closing_time_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Distribution plot saved → {out_dir}/unlabeled_closing_time_distribution.png")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ── 1. Find or download unlabeled data ───────────────────────────────────
    logger.info("=== STEP 1: Locating unlabeled data ===")

    unlabeled_extracted = data_dir / "extracted" / "unlabeled_50_1200"
    if unlabeled_extracted.exists():
        logger.info(f"Found existing extraction: {unlabeled_extracted}")
    elif args.skip_download:
        # Also check annotated data as fallback
        annotated_dirs = list((data_dir / "extracted").glob("*hz_sampling*"))
        if annotated_dirs:
            logger.info(
                "Unlabeled archive not found, using annotated data for demonstration. "
                "Run without --skip-download to download the full unlabeled archive."
            )
            unlabeled_extracted = annotated_dirs[0]
        else:
            logger.error("No data found. Run without --skip-download first.")
            sys.exit(1)
    else:
        logger.info(
            f"Downloading unlabeled archive ({UNLABELED_ARCHIVE['size_gb']} GB). "
            f"Use --skip-download to skip if already downloaded."
        )
        unlabeled_extracted = download_unlabeled(cfg, data_dir)

    # ── 2. Find and limit COMTRADE pairs ─────────────────────────────────────
    logger.info("=== STEP 2: Finding COMTRADE files ===")
    all_pairs = find_comtrade_pairs(unlabeled_extracted)
    logger.info(f"Total COMTRADE pairs in directory: {len(all_pairs)}")

    # Take first N files
    selected_pairs = all_pairs[:args.n_files]
    logger.info(f"Processing first {len(selected_pairs)} files")

    if not selected_pairs:
        logger.error("No COMTRADE files found.")
        sys.exit(1)

    # ── 3. Parse ─────────────────────────────────────────────────────────────
    logger.info("=== STEP 3: Parsing COMTRADE files ===")
    cfg_paths = [p for p, _ in selected_pairs]
    records = batch_parse(cfg_paths, verbose=True)
    logger.info(f"Successfully parsed: {len(records)}/{len(selected_pairs)}")

    # ── 4. Attempt CB grouping ────────────────────────────────────────────────
    logger.info("=== STEP 4: Attempting circuit breaker grouping ===")
    groups = group_by_breaker(records, min_records=args.min_group_size)
    grouping_assessment = assess_grouping_quality(
        group_by_breaker(records, min_records=1)  # use min=1 for assessment
    )
    logger.info(f"Grouping assessment: {json.dumps(grouping_assessment, indent=2)}")

    summary_df = summary_dataframe(groups)
    if not summary_df.empty:
        summary_df.to_csv(out_dir / "unlabeled_cb_groups.csv", index=False)
        logger.info(f"Top groups:\n{summary_df.head(10).to_string()}")

    # ── 5. Detect closing times for all parsed records ────────────────────────
    logger.info("=== STEP 5: Detecting closing times ===")
    stuck_multiplier = cfg.get("closing_detector", {}).get("stuck_median_multiplier", 1.5)
    all_det_results = []

    if grouping_assessment["groupable"] and groups:
        # Group-aware detection with adaptive reclassification
        logger.info("Groupable data: running group-aware detection with adaptive outlier filter")
        for fp, grp in groups.items():
            raw = [detect_closing_time(r, cfg) for r in grp.records]
            reclassified = reclassify_group_outliers(raw, stuck_multiplier)
            all_det_results.extend(reclassified)
        # Also detect for ungrouped records
        grouped_cfg_set = {
            str(r.cfg_path) for grp in groups.values() for r in grp.records
        }
        for rec in records:
            if str(rec.cfg_path) not in grouped_cfg_set:
                all_det_results.append(detect_closing_time(rec, cfg))
    else:
        # No meaningful grouping: run single-pass detection only
        logger.info("No meaningful grouping: running single-pass detection only")
        all_det_results = [detect_closing_time(r, cfg) for r in records]

    # Detection quality report
    det_report = detection_quality_report(all_det_results, cfg)
    logger.info(f"Detection report: {json.dumps(det_report, indent=2)}")

    # Save detection results
    det_rows = [
        {
            "quality": r.quality.value,
            "t_close_ms": r.t_close_ms,
            "command_ch": r.command_channel,
            "note": r.note,
        }
        for r in all_det_results
    ]
    pd.DataFrame(det_rows).to_csv(out_dir / "unlabeled_detection_results.csv", index=False)

    # Plot distribution
    valid_times = [r.t_close_ms for r in all_det_results
                   if r.t_close_ms is not None
                   and r.quality in (DetectionQuality.GOOD, DetectionQuality.ESTIMATED)]
    plot_detection_distribution(valid_times, out_dir)

    # ── 6. Model evaluation (only if grouping is meaningful) ─────────────────
    eval_summary = None
    modeling_note = ""

    if grouping_assessment["groupable"] and groups:
        logger.info("=== STEP 6: Model evaluation on grouped data ===")

        groups_data = {}
        for fp, grp in groups.items():
            raw = [detect_closing_time(r, cfg) for r in grp.records]
            reclassified = reclassify_group_outliers(raw, stuck_multiplier)
            times_ms = [
                r.t_close_ms if r.t_close_ms is not None else float("nan")
                for r in reclassified
            ]
            groups_data[fp] = (times_ms, reclassified)

        eval_results = evaluate_all(groups_data, cfg)
        if eval_results:
            eval_summary = aggregate_metrics(eval_results)
            logger.info(f"\nModel comparison:\n{eval_summary.to_string()}")
            eval_summary.to_csv(out_dir / "unlabeled_model_metrics.csv")

            try:
                fig = plot_model_comparison(eval_summary,
                                            save_path=str(out_dir / "unlabeled_model_comparison.png"))
                plt.close(fig)
            except Exception as exc:
                logger.warning(f"Plot failed: {exc}")

            modeling_note = f"Evaluated on {len(eval_results)} CB groups"
        else:
            modeling_note = "Grouping possible but groups too small for walk-forward evaluation"
    else:
        logger.info("=== STEP 6: Skipping model evaluation (no meaningful grouping) ===")
        modeling_note = (
            "UNLABELED DATA LIMITATION: The unlabeled archive does not contain "
            "circuit-breaker-level metadata (station_name / rec_dev_id are not "
            "consistent across files). The fingerprint-based grouping yields "
            f"{grouping_assessment['singleton_fraction']:.0%} singletons, making "
            "temporal modeling (predict N+1) statistically impossible without "
            "additional labeling. Detection quality assessment was performed instead "
            f"(n={len(all_det_results)} records, "
            f"valid_fraction={det_report['valid_fraction']:.1%})."
        )
        logger.info(f"Modeling note: {modeling_note}")

    # ── 7. Final report ───────────────────────────────────────────────────────
    report = {
        "n_files_processed": len(records),
        "grouping_assessment": grouping_assessment,
        "detection_report": det_report,
        "modeling_note": modeling_note,
    }
    if eval_summary is not None:
        report["model_comparison"] = eval_summary.reset_index().to_dict(orient="records")

    report_path = out_dir / "unlabeled_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved → {report_path}")

    print("\n" + "=" * 60)
    print("UNLABELED DATA ANALYSIS COMPLETE")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
