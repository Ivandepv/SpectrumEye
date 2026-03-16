"""
split_dataset.py

Splits augmented dataset into train / val / test sets.
Split ratios: 70% train / 15% val / 15% test (stratified per class).

IMPORTANT: augmented variants of the same source image are always kept
in the same split. This prevents data leakage where the val/test set
contains augmented copies of training images.

Input:  ml/dataset/augmented/{class}/*.png
Output: ml/dataset/train|val|test/{class}/*.png  (copies)

Usage:
  python split_dataset.py
  python split_dataset.py --seed 99
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse

# ─── PATHS ────────────────────────────────────────────────────────

AUG_DIR   = Path(__file__).parent / "dataset" / "augmented"
TRAIN_DIR = Path(__file__).parent / "dataset" / "train"
VAL_DIR   = Path(__file__).parent / "dataset" / "val"
TEST_DIR  = Path(__file__).parent / "dataset" / "test"

CLASS_LABELS = ["Key_Signal", "Walkie_Talkie", "LTE"]

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# ─── HELPERS ──────────────────────────────────────────────────────

def _group_by_source(pngs: list) -> dict:
    """
    Group PNG paths by their source image stem.

    Augmented files follow the naming convention:
      {original_stem}__{augmentation}.png

    This ensures that all augmented variants of one source image
    go into the same split (no leakage between train and val/test).

    Returns: dict mapping source_stem -> list of Path objects
    """
    groups = defaultdict(list)
    for p in pngs:
        # Strip augmentation suffix if present
        source_stem = p.stem.split("__")[0]
        groups[source_stem].append(p)
    return dict(groups)


def _clear_split_dirs(class_name: str) -> None:
    """Remove existing split files for this class to allow clean re-runs."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        class_dir = split_dir / class_name
        if class_dir.exists():
            shutil.rmtree(class_dir)


# ─── MAIN ─────────────────────────────────────────────────────────

def split_dataset(seed: int = 42) -> None:
    random.seed(seed)

    split_dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}

    print(
        f"Splitting dataset  train {SPLIT['train']:.0%} / "
        f"val {SPLIT['val']:.0%} / test {SPLIT['test']:.0%}\n"
    )

    totals = {"train": 0, "val": 0, "test": 0}

    for class_name in CLASS_LABELS:
        aug_class_dir = AUG_DIR / class_name
        pngs = sorted(aug_class_dir.glob("*.png"))

        if not pngs:
            print(f"  {class_name}: no augmented images found — skipping")
            continue

        # Group by source image to prevent leakage
        groups      = _group_by_source(pngs)
        source_keys = list(groups.keys())
        random.shuffle(source_keys)

        n        = len(source_keys)
        n_val    = max(1, int(n * SPLIT["val"]))
        n_test   = max(1, int(n * SPLIT["test"]))
        n_train  = n - n_val - n_test

        split_keys = {
            "train": source_keys[:n_train],
            "val":   source_keys[n_train : n_train + n_val],
            "test":  source_keys[n_train + n_val:],
        }

        _clear_split_dirs(class_name)

        split_counts = {"train": 0, "val": 0, "test": 0}
        for split_name, keys in split_keys.items():
            dest_dir = split_dirs[split_name] / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for key in keys:
                for f in groups[key]:
                    shutil.copy2(f, dest_dir / f.name)
                    split_counts[split_name] += 1
            totals[split_name] += split_counts[split_name]

        print(
            f"  {class_name}: {len(pngs)} images ({n} sources) -> "
            f"train={split_counts['train']}  "
            f"val={split_counts['val']}  "
            f"test={split_counts['test']}"
        )

    print(
        f"\n  Grand total — "
        f"train: {totals['train']}  "
        f"val: {totals['val']}  "
        f"test: {totals['test']}"
    )
    print("\nDone. Ready for train.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split augmented dataset into train/val/test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    split_dataset(seed=args.seed)
