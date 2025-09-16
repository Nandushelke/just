import os
from typing import Dict, Any

from datasets import load_dataset, DatasetDict

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def ensure_data_dir() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def download_imdb(save_jsonl: bool = True) -> Dict[str, Any]:
    ensure_data_dir()
    ds: DatasetDict = load_dataset("imdb")  # splits: train, test, unsupervised
    out = {}
    for split in ds:
        out_path = os.path.join(DATA_DIR, f"imdb_{split}.jsonl")
        if save_jsonl:
            ds[split].to_json(out_path, orient="records", lines=True, force_ascii=False)
        out[split] = {
            "num_rows": ds[split].num_rows,
            "features": ds[split].features,
            "path": out_path if save_jsonl else None,
        }
    return {"name": "imdb", "splits": out}


def download_emotion(save_jsonl: bool = True) -> Dict[str, Any]:
    ensure_data_dir()
    ds: DatasetDict = load_dataset("emotion")  # splits: train, validation, test
    out = {}
    for split in ds:
        out_path = os.path.join(DATA_DIR, f"emotion_{split}.jsonl")
        if save_jsonl:
            ds[split].to_json(out_path, orient="records", lines=True, force_ascii=False)
        out[split] = {
            "num_rows": ds[split].num_rows,
            "features": ds[split].features,
            "path": out_path if save_jsonl else None,
        }
    return {"name": "emotion", "splits": out}


def download_dataset(name: str) -> Dict[str, Any]:
    name = name.lower().strip()
    if name == "imdb":
        return download_imdb()
    if name == "emotion":
        return download_emotion()
    raise ValueError(f"Unsupported dataset: {name}")