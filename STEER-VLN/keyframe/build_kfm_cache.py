import argparse
import sys
from pathlib import Path

STEER_VLN_DIR = Path(__file__).resolve().parents[1]
if str(STEER_VLN_DIR) not in sys.path:
    sys.path.insert(0, str(STEER_VLN_DIR))

from steer_vln_runner import STEER_VLN, ensure_dir, require_dir, require_file, run_with_log, now


def parse_args():
    p = argparse.ArgumentParser("STEER_VLN build learned keyframe cache")
    p.add_argument("--split", type=str, default="both", choices=["train", "eval", "both"])
    return p.parse_args()


def build_cache(split: str):
    require_file(f'{STEER_VLN["kfm_dir"]}/keyframe_scorer_best.pt')
    require_file(f'{STEER_VLN["kfm_dir"]}/simple_tokenizer_vocab.json')
    require_dir(STEER_VLN["parquet_root"])

    if split == "train":
        annotation = STEER_VLN["train_json"]
        output_json = STEER_VLN["cache_train"]
        log_dir = STEER_VLN["logs_train"]
    else:
        annotation = STEER_VLN["eval_json"]
        output_json = STEER_VLN["cache_eval"]
        log_dir = STEER_VLN["logs_eval"]

    require_file(annotation)
    ensure_dir("runs/STEER-VLN/modules")
    ensure_dir(log_dir)

    cmd = [
        sys.executable,
        "STEER-VLN/keyframe/build_kfm_cache_core.py",

        "--checkpoint", f'{STEER_VLN["kfm_dir"]}/keyframe_scorer_best.pt',
        "--vocab_path", f'{STEER_VLN["kfm_dir"]}/simple_tokenizer_vocab.json',
        "--annotation_path", annotation,
        "--parquet_root", STEER_VLN["parquet_root"],
        "--output_json", output_json,

        "--max_episodes", "0",
        "--max_history", STEER_VLN["max_history"],
        "--image_size", "224",
        "--horizon", STEER_VLN["horizon"],
        "--stride", "1",
        "--min_timestep", "2",

        "--batch_size", "64",
        "--num_workers", "4",
        "--cache_size", STEER_VLN["cache_size"],

        "--exclude_previous",
        "--device", "cuda",
    ]

    log_path = f"{log_dir}/build_steer_vln_kfm_cache_{split}_{now()}.log"
    run_with_log(cmd, log_path)


def main():
    args = parse_args()

    if args.split in ["train", "both"]:
        build_cache("train")

    if args.split in ["eval", "both"]:
        build_cache("eval")


if __name__ == "__main__":
    main()
