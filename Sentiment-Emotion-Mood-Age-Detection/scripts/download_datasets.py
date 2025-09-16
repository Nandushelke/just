import os
from datasets import load_dataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def save_dataset(ds, prefix: str):
    for split in ds:
        out_path = os.path.join(DATA_DIR, f"{prefix}_{split}.jsonl")
        ds[split].to_json(out_path, orient="records", lines=True, force_ascii=False)
        print(f"Wrote {out_path} ({ds[split].num_rows} rows)")


def main():
    imdb = load_dataset("imdb")
    save_dataset(imdb, "imdb")

    emotion = load_dataset("emotion")
    save_dataset(emotion, "emotion")

    print("Done")


if __name__ == "__main__":
    main()