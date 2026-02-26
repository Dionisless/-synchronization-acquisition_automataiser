"""
Download module for OscGrid dataset (Figshare 10.6084/m9.figshare.28465427).

Downloads COMTRADE oscillogram archives and extracts them.
Prioritizes smaller annotated archives to avoid excessive storage/bandwidth.
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests
import py7zr
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
    max_retries: int = 4,
) -> bool:
    """
    Download a file from URL with progress bar and retry logic.
    Skips download if file already exists.
    """
    if dest_path.exists():
        logger.info(f"Already downloaded: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    backoff = 2  # seconds

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            with open(dest_path, "wb") as f, tqdm(
                desc=dest_path.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))
            return True

        except (requests.RequestException, IOError) as exc:
            logger.warning(f"Download attempt {attempt + 1} failed: {exc}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < max_retries:
                logger.info(f"Retrying in {backoff}s …")
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(f"Failed to download {url} after {max_retries} retries")
                return False

    return False


def extract_7z(archive_path: Path, dest_dir: Path) -> bool:
    """Extract a 7z archive. Skips if already extracted (marker file present)."""
    marker = dest_dir / f".extracted_{archive_path.stem}"
    if marker.exists():
        logger.info(f"Already extracted: {archive_path.name}")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Extracting {archive_path.name} → {dest_dir}")
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(dest_dir)
        marker.touch()
        return True
    except Exception as exc:
        logger.error(f"Extraction failed for {archive_path.name}: {exc}")
        return False


def download_annotated(
    config: dict,
    data_dir: Optional[Path] = None,
    annotated_only: bool = True,
) -> list[Path]:
    """
    Download OscGrid dataset archives from Figshare.

    Args:
        config:          Loaded config dict.
        data_dir:        Override data directory.
        annotated_only:  If True, skip the large unlabeled archives (>2 GB each).

    Returns:
        List of paths to extracted directories.
    """
    cfg = config["dataset"]
    base_url = cfg["figshare_base_url"]
    raw_dir = Path(data_dir or cfg["data_dir"])
    archives_dir = raw_dir / "archives"
    extracted_dir = raw_dir / "extracted"

    archives_dir.mkdir(parents=True, exist_ok=True)

    archives = cfg["annotated_archives"]
    if not annotated_only:
        archives = archives + cfg.get("unlabeled_archives", [])

    extracted_dirs: list[Path] = []

    for archive_info in archives:
        file_id = archive_info["file_id"]
        name = archive_info["name"]
        url = f"{base_url}{file_id}"
        archive_path = archives_dir / name

        logger.info(f"Downloading {name} ({archive_info.get('size_kb', '?')} KB)…")
        ok = download_file(url, archive_path)
        if not ok:
            logger.error(f"Skipping {name} due to download failure")
            continue

        stem = Path(name).stem  # e.g. "600hz_sampling"
        out_dir = extracted_dir / stem
        ok = extract_7z(archive_path, out_dir)
        if ok:
            extracted_dirs.append(out_dir)

    return extracted_dirs


def find_comtrade_pairs(root_dir: Path) -> list[tuple[Path, Path]]:
    """
    Recursively find all COMTRADE (.cfg + .dat) pairs under root_dir.
    Returns list of (cfg_path, dat_path) tuples.
    """
    pairs: list[tuple[Path, Path]] = []
    for cfg_path in root_dir.rglob("*.cfg"):
        dat_path = cfg_path.with_suffix(".dat")
        if not dat_path.exists():
            # Try uppercase
            dat_path = cfg_path.with_suffix(".DAT")
        if dat_path.exists():
            pairs.append((cfg_path, dat_path))
        else:
            logger.debug(f"No .dat for {cfg_path}, skipping")
    return sorted(pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    dirs = download_annotated(cfg, annotated_only=True)
    print(f"Extracted to: {[str(d) for d in dirs]}")
    all_pairs = []
    for d in dirs:
        pairs = find_comtrade_pairs(d)
        all_pairs.extend(pairs)
        print(f"  {d.name}: {len(pairs)} COMTRADE pairs")
    print(f"Total: {len(all_pairs)} COMTRADE pairs")
