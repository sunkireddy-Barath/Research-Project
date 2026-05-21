"""
main.py — Single-Command Entry Point
=====================================
Runs the complete research pipeline in sequence:

  1. experiments.py  → trains the framework + evaluates all baselines
                        saves results to config.EXP_DATA_PATH
  2. visualize.py    → loads results and generates all 9 paper figures
                        saves figures to config.FIGURES_DIR

Usage
-----
    python main.py

Flags
-----
    --skip-train     load existing weights instead of retraining
    --exp-only       run experiments only, skip figure generation
    --viz-only       skip experiments, generate figures from saved results
"""

import argparse
import os
import sys
import time

import config


def _header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_experiments(skip_train: bool = False) -> None:
    _header("Stage 1 — Experiments")
    from experiments import main as exp_main
    exp_main(skip_train=skip_train)
    print(f"\n[main] Results saved → {config.EXP_DATA_PATH}")


def run_visualize() -> None:
    _header("Stage 2 — Figures")
    if not os.path.exists(config.EXP_DATA_PATH):
        print(f"[main] ERROR: {config.EXP_DATA_PATH} not found — run experiments first")
        sys.exit(1)
    from visualize import main as viz_main
    viz_main()
    print(f"\n[main] Figures saved → {config.FIGURES_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Healing Blockchain IDS — full research pipeline"
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="load existing DQN weights instead of retraining")
    parser.add_argument("--exp-only",   action="store_true",
                        help="run experiments only, skip figure generation")
    parser.add_argument("--viz-only",   action="store_true",
                        help="generate figures from saved results, skip experiments")
    args = parser.parse_args()

    t0 = time.time()

    if not args.viz_only:
        run_experiments(skip_train=args.skip_train)

    if not args.exp_only:
        run_visualize()

    elapsed = time.time() - t0
    _header(f"Pipeline complete — {elapsed:.1f}s")
    print(f"  Results : {config.EXP_DATA_PATH}")
    print(f"  Figures : {config.FIGURES_DIR}/")
    print()


if __name__ == "__main__":
    main()
