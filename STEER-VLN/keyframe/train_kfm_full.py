import argparse
import sys
from pathlib import Path

STEER_VLN_DIR = Path(__file__).resolve().parents[1]
if str(STEER_VLN_DIR) not in sys.path:
    sys.path.insert(0, str(STEER_VLN_DIR))

from steer_vln_runner import STEER_VLN, ensure_dir, remove_dir, require_dir, require_file, run_with_log, now


def parse_args():
    p = argparse.ArgumentParser("STEER_VLN-KFM full main experiment")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    require_file(STEER_VLN["train_json"])
    require_dir(STEER_VLN["parquet_root"])

    if args.overwrite:
        remove_dir(STEER_VLN["kfm_dir"])

    ensure_dir(STEER_VLN["kfm_dir"])
    ensure_dir(STEER_VLN["logs_train"])

    cmd = [
        sys.executable,
        "STEER-VLN/keyframe/train_kfm_full_core.py",

        "--annotation_path", STEER_VLN["train_json"],
        "--parquet_root", STEER_VLN["parquet_root"],
        "--output_dir", STEER_VLN["kfm_dir"],

        "--max_episodes", "0",
        "--max_history", STEER_VLN["max_history"],
        "--image_size", "224",

        "--batch_size", "16",
        "--epochs", "3",
        "--num_workers", "4",
        "--prefetch_factor", "2",
        "--cache_size", STEER_VLN["cache_size"],

        "--learning_rate", "2e-4",
        "--val_ratio", "0.001",
        "--save_interval", "1000",
        "--bf16",
    ]

    log_path = f'{STEER_VLN["logs_train"]}/train_steer_vln_kfm_full_{now()}.log'
    run_with_log(cmd, log_path)


if __name__ == "__main__":
    main()
