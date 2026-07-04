import argparse
import sys
from pathlib import Path

STEER_VLN_DIR = Path(__file__).resolve().parents[1]
if str(STEER_VLN_DIR) not in sys.path:
    sys.path.insert(0, str(STEER_VLN_DIR))

from steer_vln_runner import STEER_VLN, ensure_dir, require_dir, require_file, run_with_log, now


def parse_args():
    p = argparse.ArgumentParser("STEER_VLN-KFM full eval wrapper")
    return p.parse_args()


def main():
    parse_args()

    require_file(f'{STEER_VLN["kfm_dir"]}/keyframe_scorer_best.pt')
    require_file(f'{STEER_VLN["kfm_dir"]}/simple_tokenizer_vocab.json')
    require_file(STEER_VLN["eval_json"])
    require_dir(STEER_VLN["parquet_root"])

    ensure_dir(STEER_VLN["kfm_eval_dir"])
    ensure_dir(STEER_VLN["logs_eval"])

    cmd = [
        sys.executable,
        "STEER-VLN/keyframe/eval_kfm_full_core.py",

        "--checkpoint", f'{STEER_VLN["kfm_dir"]}/keyframe_scorer_best.pt',
        "--vocab_path", f'{STEER_VLN["kfm_dir"]}/simple_tokenizer_vocab.json',
        "--annotation_path", STEER_VLN["eval_json"],
        "--parquet_root", STEER_VLN["parquet_root"],
        "--output_dir", STEER_VLN["kfm_eval_dir"],

        "--max_episodes", "0",
        "--max_history", STEER_VLN["max_history"],
        "--image_size", "224",
        "--exclude_previous",
    ]

    log_path = f'{STEER_VLN["logs_eval"]}/eval_steer_vln_kfm_full_{now()}.log'
    run_with_log(cmd, log_path)


if __name__ == "__main__":
    main()

