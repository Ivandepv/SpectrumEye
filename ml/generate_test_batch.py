"""
ml/generate_test_batch.py

Generates a fresh batch of test images that the model has NEVER seen.

This simulates what happens when the partner runs simulation_final.py
a second time independently — different Colab session, different random
state, no overlap with the training data.

Output:
  test_batch/
  ├── Key_Signal/     test_key_0000.png  ...  test_key_0099.png
  ├── Walkie_Talkie/  test_walkie_0000.png ... test_walkie_0099.png
  └── LTE/            test_lte_0000.png  ...  test_lte_0099.png

Usage:
  python ml/generate_test_batch.py
  python ml/evaluate.py --folder test_batch/
"""

import os
import sys
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────

N_PER_CLASS = 100          # 100 images per class is enough for a solid test
OUT_DIR     = Path("test_batch")
FS          = 1000
DURATION    = 2
SEED        = 777          # different from training (42) and quick test (999)

# ── Import simulation from partner code ───────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

try:
    from simulation_final import (
        generate_t_sampling,
        key_signal_simulation,
        walkie_talkie_simulation,
        lte_simulation,
        awgn,
        spectrogram_to_image,
    )
except ImportError:
    print("ERROR: simulation/simulation_final.py not found.")
    print("Run from the project root: python ml/generate_test_batch.py")
    sys.exit(1)

# ── Generate ──────────────────────────────────────────────────────

def generate(n_per_class: int = N_PER_CLASS, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed=seed)
    t   = generate_t_sampling(FS, DURATION)

    classes = {
        "Key_Signal":    (key_signal_simulation,    "test_key"),
        "Walkie_Talkie": (walkie_talkie_simulation,  "test_walkie"),
        "LTE":           (lte_simulation,            "test_lte"),
    }

    print(f"Generating test batch (seed={seed}, {n_per_class} images per class)\n")

    for cls_name, (sim_fn, prefix) in classes.items():
        out_path = OUT_DIR / cls_name
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"  {cls_name}: ", end="", flush=True)
        for n in range(n_per_class):
            np.random.seed(int(rng.integers(0, 999999)))
            pure  = sim_fn(t, FS)
            noisy = awgn(pure, noise_level=0.5)
            img   = spectrogram_to_image(noisy, FS)
            img.save(out_path / f"{prefix}_{n:04d}.png")

        print(f"{n_per_class} images → {out_path}/")

    total = n_per_class * len(classes)
    print(f"\nDone. {total} test images saved to {OUT_DIR}/")
    print("\nNext step:")
    print(f"  python ml/evaluate.py --folder {OUT_DIR}/")


if __name__ == "__main__":
    generate()
