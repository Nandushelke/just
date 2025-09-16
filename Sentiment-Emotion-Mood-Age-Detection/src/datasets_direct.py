from __future__ import annotations

import os
import tarfile
import urllib.request
from typing import Dict, Any


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download(url: str, dest_path: str) -> str:
    _ensure_dir(os.path.dirname(dest_path))
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def download_goemotions() -> Dict[str, Any]:
    """
    Download GoEmotions splits (train/dev/test) from Google Research GitHub.
    Files are TSV with text and multi-label columns.
    """
    _ensure_dir(DATA_DIR)
    base = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
    out_dir = os.path.join(DATA_DIR, "goemotions")
    _ensure_dir(out_dir)

    results: Dict[str, Any] = {"name": "goemotions", "files": {}}
    for split in ["train.tsv", "dev.tsv", "test.tsv"]:
        url = f"{base}/{split}"
        dest = os.path.join(out_dir, split)
        _download(url, dest)
        results["files"][split] = dest
    return results


def download_rt_polarity() -> Dict[str, Any]:
    """
    Download the RT-Polarity sentiment dataset (Cornell) and extract it.
    ~5MB tar.gz containing pos/neg sentence files.
    """
    _ensure_dir(DATA_DIR)
    url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
    tar_path = os.path.join(DATA_DIR, "rt-polaritydata.tar.gz")
    _download(url, tar_path)

    extract_dir = os.path.join(DATA_DIR, "rt_polarity")
    _ensure_dir(extract_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    return {"name": "rt_polarity", "archive": tar_path, "extracted": extract_dir}


def download_direct(name: str) -> Dict[str, Any]:
    name = name.lower().strip()
    if name in {"goemotions", "go-emotions", "go_emotions"}:
        return download_goemotions()
    if name in {"rt_polarity", "rt-polarity", "rtpolarity"}:
        return download_rt_polarity()
    raise ValueError(f"Unsupported direct dataset: {name}")

