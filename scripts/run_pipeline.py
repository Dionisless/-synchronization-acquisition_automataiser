"""
Full pipeline: Download → Parse → Detect → Evaluate → Report

Usage:
    python scripts/run_pipeline.py [--annotated-only] [--data-dir data/raw]
                                   [--output-dir results] [--config config.yaml]
                                   [--skip-download]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.download import download_annotated, find_comtrade_pairs, load_config
from src.comtrade_parser import parse_comtrade, batch_parse
from src.cb_grouper import group_by_breaker, summary_dataframe
from src.closing_detector import detect_closing_time, DetectionQuality
from src.models import build_models
from src.evaluation import (
    evaluate_all,
    aggregate_metrics,
    detect_drift,
    plot_cb_timeline,
    plot_model_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="CB Closing Time Pipeline")
    p.add_argument("--annotated-only", action="store_true", default=True,
                   help="Download only annotated archives (default: True)")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download if data already present")
    p.add_argument("--max-groups", type=int, default=None,
                   help="Limit number of CB groups to process (for debugging)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ──────────────────────────────────────────────────────────
    if not args.skip_download:
        logger.info("=== STEP 1: Downloading dataset ===")
        extracted_dirs = download_annotated(cfg, Path(args.data_dir), annotated_only=args.annotated_only)
    else:
        logger.info("=== STEP 1: Skipping download (--skip-download) ===")
        data_dir = Path(args.data_dir) / "extracted"
        extracted_dirs = [d for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []

    if not extracted_dirs:
        logger.error("No extracted directories found. Exiting.")
        sys.exit(1)

    # ── 2. Find & parse COMTRADE files ───────────────────────────────────────
    logger.info("=== STEP 2: Parsing COMTRADE files ===")
    all_cfg_paths = []
    for d in extracted_dirs:
        pairs = find_comtrade_pairs(d)
        all_cfg_paths.extend(p for p, _ in pairs)
        logger.info(f"  {d.name}: {len(pairs)} pairs")

    logger.info(f"Total COMTRADE files: {len(all_cfg_paths)}")
    records = batch_parse(all_cfg_paths, verbose=True)
    logger.info(f"Successfully parsed: {len(records)}")

    if not records:
        logger.error("No records parsed. Exiting.")
        sys.exit(1)

    # ── 3. Group by circuit breaker ──────────────────────────────────────────
    logger.info("=== STEP 3: Grouping by circuit breaker ===")
    min_ops = cfg.get("evaluation", {}).get("min_operations", 5)
    groups = group_by_breaker(records, min_records=min_ops)
    logger.info(f"Found {len(groups)} circuit breaker groups")

    summary_df = summary_dataframe(groups)
    summary_path = out_dir / "cb_groups_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Group summary saved → {summary_path}")
    print("\n" + summary_df.head(10).to_string() + "\n")

    if args.max_groups:
        group_items = list(groups.items())[:args.max_groups]
        groups = dict(group_items)

    # ── 4. Detect closing times ───────────────────────────────────────────────
    logger.info("=== STEP 4: Detecting closing times ===")
    groups_data: dict[str, tuple[list[float], list]] = {}
    all_detection_rows = []

    for fp, grp in groups.items():
        det_results = []
        times_ms = []
        for rec in grp.records:
            res = detect_closing_time(rec, cfg)
            det_results.append(res)
            t = res.t_close_ms
            times_ms.append(t if t is not None else float("nan"))

            all_detection_rows.append({
                "fingerprint": fp[:60],
                "station": rec.station_name,
                "dev_id": rec.rec_dev_id,
                "start_time": rec.start_time,
                "t_close_ms": t,
                "quality": res.quality.value,
                "command_ch": res.command_channel,
                "current_ch": res.current_channel,
                "threshold_a": res.current_threshold_a,
                "note": res.note,
            })
        groups_data[fp] = (times_ms, det_results)

    det_df = pd.DataFrame(all_detection_rows)
    det_path = out_dir / "detection_results.csv"
    det_df.to_csv(det_path, index=False)
    logger.info(f"Detection results saved → {det_path}")

    # Summary statistics
    quality_counts = det_df["quality"].value_counts()
    logger.info(f"Detection quality breakdown:\n{quality_counts.to_string()}")
    valid = det_df[det_df["quality"].isin(["good", "estimated"])]["t_close_ms"]
    if not valid.empty:
        logger.info(
            f"Valid closing times: n={len(valid)}, "
            f"mean={valid.mean():.2f} ms, std={valid.std():.2f} ms, "
            f"range=[{valid.min():.1f}, {valid.max():.1f}] ms"
        )

    # ── 5. Drift analysis ─────────────────────────────────────────────────────
    logger.info("=== STEP 5: Drift analysis ===")
    drift_rows = []
    for fp, (times_ms, _) in groups_data.items():
        clean_times = [t for t in times_ms if not np.isnan(t)]
        if len(clean_times) >= 4:
            drift = detect_drift(clean_times)
            drift["group_id"] = fp[:60]
            drift["n_ops"] = len(clean_times)
            drift_rows.append(drift)

    if drift_rows:
        drift_df = pd.DataFrame(drift_rows)
        drift_path = out_dir / "drift_analysis.csv"
        drift_df.to_csv(drift_path, index=False)
        trend_counts = drift_df["trend"].value_counts()
        logger.info(f"Drift analysis:\n{trend_counts.to_string()}")

    # ── 6. Model evaluation ───────────────────────────────────────────────────
    logger.info("=== STEP 6: Model evaluation ===")

    # Prepare data: filter to groups with sufficient GOOD measurements
    eval_data = {}
    for fp, (times_ms, det_results) in groups_data.items():
        good_count = sum(
            1 for r in det_results
            if r.quality in (DetectionQuality.GOOD, DetectionQuality.ESTIMATED)
            and r.t_close_ms is not None
        )
        if good_count >= min_ops:
            eval_data[fp] = (times_ms, det_results)

    logger.info(f"Groups eligible for evaluation: {len(eval_data)}")

    eval_results = evaluate_all(eval_data, cfg)

    if not eval_results:
        logger.warning("No evaluation results produced (insufficient data per group).")
    else:
        summary = aggregate_metrics(eval_results)
        logger.info(f"\n{'='*60}\nModel comparison summary:\n{summary.to_string()}\n{'='*60}")
        summary_metrics_path = out_dir / "model_metrics_summary.csv"
        summary.to_csv(summary_metrics_path)
        logger.info(f"Metrics saved → {summary_metrics_path}")

        # Per-group results
        per_group_rows = []
        for er in eval_results:
            best = er.best_model()
            per_group_rows.append({
                "group_id": er.group_id,
                "n_total": er.n_total,
                "n_train": er.n_train,
                "n_test": er.n_test,
                "n_outliers": er.n_outliers_detected,
                "best_model": best.model_name if best else None,
                "best_mae_ms": best.mae_ms if best else None,
            })
        per_group_df = pd.DataFrame(per_group_rows)
        per_group_df.to_csv(out_dir / "per_group_results.csv", index=False)

        # Plot model comparison
        try:
            fig = plot_model_comparison(summary, save_path=str(out_dir / "model_comparison.png"))
            import matplotlib.pyplot as plt
            plt.close(fig)
            logger.info(f"Plot saved → {out_dir}/model_comparison.png")
        except Exception as exc:
            logger.warning(f"Could not save comparison plot: {exc}")

    # ── 7. Generate report ────────────────────────────────────────────────────
    logger.info("=== STEP 7: Generating report ===")
    report = {
        "total_comtrade_files": len(all_cfg_paths),
        "parsed_records": len(records),
        "cb_groups": len(groups),
        "evaluated_groups": len(eval_results) if eval_results else 0,
        "detection_quality": quality_counts.to_dict() if not det_df.empty else {},
    }
    if not valid.empty:
        report["closing_time_stats"] = {
            "mean_ms": round(float(valid.mean()), 3),
            "std_ms": round(float(valid.std()), 3),
            "min_ms": round(float(valid.min()), 3),
            "max_ms": round(float(valid.max()), 3),
            "median_ms": round(float(valid.median()), 3),
        }
    if eval_results:
        summary_dict = summary.reset_index().to_dict(orient="records")
        report["model_comparison"] = summary_dict

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved → {report_path}")

    logger.info("=== Pipeline complete ===")
    print("\n📊 Final report:")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
