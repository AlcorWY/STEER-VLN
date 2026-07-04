# ===== STEER_VLN force isolated import path =====
import sys as _steer_vln_sys
from pathlib import Path as _STEER_VLNPath

_STEER_VLN_FILE = _STEER_VLNPath(__file__).resolve()
_STEER_VLN_DIR = _STEER_VLN_FILE.parent
_STEER_VLN_ROOT = _STEER_VLN_DIR.parents[0]
_STEER_VLN_CODE = _STEER_VLN_ROOT / "code"
_STEER_VLN_TRAIN = _STEER_VLN_ROOT / "train"

for _p in [str(_STEER_VLN_DIR), str(_STEER_VLN_CODE), str(_STEER_VLN_ROOT), str(_STEER_VLN_TRAIN)]:
    if _p in _steer_vln_sys.path:
        _steer_vln_sys.path.remove(_p)

# priority:
#   STEER-VLN -> code -> project root -> train
_steer_vln_sys.path.insert(0, str(_STEER_VLN_ROOT))
_steer_vln_sys.path.insert(0, str(_STEER_VLN_CODE))
_steer_vln_sys.path.insert(0, str(_STEER_VLN_DIR))
_steer_vln_sys.path.append(str(_STEER_VLN_TRAIN))
# ===== end STEER_VLN force isolated import path =====

import sys
from pathlib import Path as _PathForSysPath

_PROJECT_ROOT = _PathForSysPath(__file__).resolve().parents[1]
_TRAIN_ROOT = _PROJECT_ROOT / "train"
_CODE_ROOT = _PROJECT_ROOT / "code"

for _p in [str(_STEER_VLN_DIR), str(_CODE_ROOT), str(_PROJECT_ROOT), str(_TRAIN_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)

sys.path.insert(0, str(_STEER_VLN_DIR))
sys.path.insert(1, str(_CODE_ROOT))
sys.path.insert(2, str(_PROJECT_ROOT))
sys.path.append(str(_TRAIN_ROOT))

import os

# Keep OpenFly evaluations offline by default.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from unrealcv import Client
import cv2
import numpy as np
import io
import time
import math
import subprocess, threading
import airsim
from common import *


# =========================
# OpenFly paper metric helpers
# =========================
def openfly_scene_category(item, env_name="unknown"):
    """
    Extract a scene/category label for scene-wise reporting.
    Typical image_path: env_airsim_16/astar_data/medium_average/xxx
    -> scene_category: medium_average
    Fallback: env name, so seen/unseen official json can still be grouped.
    """
    try:
        image_path = str(item.get("image_path", ""))
    except Exception:
        image_path = ""

    parts = [p for p in image_path.replace("\\", "/").split("/") if p]
    if "astar_data" in parts:
        i = parts.index("astar_data")
        if i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]

    # OpenFly official json usually starts with the environment name.
    if parts:
        for p in parts:
            if p.startswith("env_"):
                return p

    # fallback: use parent folder before file name
    if len(parts) >= 2 and parts[-2]:
        return parts[-2]

    return str(env_name)


def openfly_compute_spl_values(success, traj_len, pass_len):
    """
    Return both SPL definitions used in reporting:
      - spl_openfly: OpenFly source-compatible formula, success * traj_len / pass_len
      - spl_standard: standard bounded SPL, success * traj_len / max(traj_len, pass_len)
    """
    ref_len = max(float(traj_len), 1e-6)
    pred_len = max(float(pass_len), 1e-6)
    if int(success) == 1:
        spl_openfly = ref_len / pred_len
        spl_standard = ref_len / max(ref_len, pred_len)
        spl_standard = max(0.0, min(1.0, float(spl_standard)))
    else:
        spl_openfly = 0.0
        spl_standard = 0.0
    return float(spl_openfly), float(spl_standard)


def openfly_append_scene_metric(
    nav_episode_metrics,
    env_name,
    scene_category,
    sample_idx,
    ne_m,
    sr,
    osr,
    spl_raw,
    spl_bounded,
    pred_stop=False,
    successful_stop=False,
    forced_end=False,
    image_error=False,
):
    """Append one episode result for total/scene/env-scene summaries.

    spl_raw keeps backward compatibility with old logs and equals SPL_openfly.
    spl_bounded keeps backward compatibility with old logs and equals SPL_standard.
    Stop rates are computed over valid stop samples, i.e. non-image-error samples.
    """
    valid_stop = not bool(image_error)
    nav_episode_metrics.append({
        "env": str(env_name),
        "scene": str(scene_category),
        "sample_idx": int(sample_idx),
        "NE/m": float(ne_m),
        "SR": int(sr),
        "OSR": int(osr),
        "SPL_openfly": float(spl_raw),
        "SPL_standard": float(spl_bounded),
        "SPL_raw": float(spl_raw),
        "SPL_bounded": float(spl_bounded),
        "SPL": float(spl_bounded),
        "pred_stop": int(bool(pred_stop) and valid_stop),
        "successful_stop": int(bool(successful_stop) and valid_stop),
        "forced_end": int(bool(forced_end) and valid_stop),
        "image_error": int(bool(image_error)),
        "valid_stop": int(valid_stop),
    })


def openfly_print_and_save_metric_summary(nav_episode_metrics, metrics_tag="eval"):
    import csv as _csv
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    from collections import defaultdict as _defaultdict

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _summ(rows):
        valid_stop_samples = sum(int(r.get("valid_stop", 1)) for r in rows)
        stop_denom = valid_stop_samples if valid_stop_samples > 0 else len(rows)
        pred_stop_count = sum(int(r.get("pred_stop", 0)) for r in rows if int(r.get("valid_stop", 1)) == 1)
        successful_stop_count = sum(int(r.get("successful_stop", 0)) for r in rows if int(r.get("valid_stop", 1)) == 1)
        forced_end_count = sum(int(r.get("forced_end", 0)) for r in rows if int(r.get("valid_stop", 1)) == 1)
        image_error_count = sum(int(r.get("image_error", 0)) for r in rows)
        return {
            "samples": len(rows),
            "Total samples": len(rows),
            "Valid samples for stop metrics": valid_stop_samples,
            "NE/m": _mean([r["NE/m"] for r in rows]),
            "SR/%": 100.0 * _mean([r["SR"] for r in rows]),
            "OSR/%": 100.0 * _mean([r["OSR"] for r in rows]),
            "SPL/%": 100.0 * _mean([r.get("SPL_standard", r.get("SPL_bounded", r.get("SPL", 0.0))) for r in rows]),
            "SPL_openfly/%": 100.0 * _mean([r.get("SPL_openfly", r.get("SPL_raw", 0.0)) for r in rows]),
            "SPL_standard/%": 100.0 * _mean([r.get("SPL_standard", r.get("SPL_bounded", r.get("SPL", 0.0))) for r in rows]),
            "SPL_raw/%": 100.0 * _mean([r.get("SPL_raw", r.get("SPL_openfly", 0.0)) for r in rows]),
            "SPL_bounded/%": 100.0 * _mean([r.get("SPL_bounded", r.get("SPL_standard", r.get("SPL", 0.0))) for r in rows]),
            "Image error count": image_error_count,
            "Image error rate": image_error_count / len(rows) if rows else 0.0,
            "Pred stop count": pred_stop_count,
            "Pred stop rate": pred_stop_count / stop_denom if stop_denom > 0 else 0.0,
            "Successful stop count": successful_stop_count,
            "Successful stop rate": successful_stop_count / stop_denom if stop_denom > 0 else 0.0,
            "Forced end count": forced_end_count,
            "Forced end rate": forced_end_count / stop_denom if stop_denom > 0 else 0.0,
        }

    total = _summ(nav_episode_metrics)

    print("\n[OverallMetrics]")
    print(f"Total samples: {total['Total samples']}")
    print(f"Valid samples for stop metrics: {total['Valid samples for stop metrics']}")
    print(f"NE/m: {total['NE/m']:.4f}")
    print(f"SR/%: {total['SR/%']:.2f}")
    print(f"OSR/%: {total['OSR/%']:.2f}")
    print(f"SPL/%: {total['SPL/%']:.2f}")
    print(f"SPL_openfly/%: {total['SPL_openfly/%']:.2f}")
    print(f"SPL_standard/%: {total['SPL_standard/%']:.2f}")
    print(f"SPL_raw/%: {total['SPL_raw/%']:.2f}")
    print(f"SPL_bounded/%: {total['SPL_bounded/%']:.2f}")
    print(f"Image error count: {total['Image error count']}")
    print(f"Image error rate: {total['Image error rate']:.4f}")
    print(f"Pred stop count: {total['Pred stop count']}")
    print(f"Pred stop rate: {total['Pred stop rate']:.4f}")
    print(f"Successful stop count: {total['Successful stop count']}")
    print(f"Successful stop rate: {total['Successful stop rate']:.4f}")
    print(f"Forced end count: {total['Forced end count']}")
    print(f"Forced end rate: {total['Forced end rate']:.4f}")

    by_scene = _defaultdict(list)
    by_env_scene = _defaultdict(list)
    for r in nav_episode_metrics:
        by_scene[r["scene"]].append(r)
        by_env_scene[(r["env"], r["scene"])].append(r)

    print("\nScene-wise metrics:")
    scene_rows = []
    for scene in sorted(by_scene.keys()):
        info = _summ(by_scene[scene])
        row = {"scene": scene, **info}
        scene_rows.append(row)
        print(
            f"[SceneMetrics] scene={scene} samples={info['samples']} "
            f"valid_stop={info['Valid samples for stop metrics']} "
            f"NE/m={info['NE/m']:.4f} SR/%={info['SR/%']:.2f} OSR/%={info['OSR/%']:.2f} "
            f"SPL/%={info['SPL/%']:.2f} SPL_openfly/%={info['SPL_openfly/%']:.2f} "
            f"SPL_standard/%={info['SPL_standard/%']:.2f} "
            f"PredStop={info['Pred stop count']} PredStopRate={info['Pred stop rate']:.4f} "
            f"SuccessfulStop={info['Successful stop count']} SuccessfulStopRate={info['Successful stop rate']:.4f} "
            f"ForcedEnd={info['Forced end count']} ForcedEndRate={info['Forced end rate']:.4f} "
            f"ImageError={info['Image error count']}"
        )

    print("\nEnv-scene-wise metrics:")
    env_scene_rows = []
    for env, scene in sorted(by_env_scene.keys()):
        info = _summ(by_env_scene[(env, scene)])
        row = {"env": env, "scene": scene, **info}
        env_scene_rows.append(row)
        print(
            f"[EnvSceneMetrics] env={env} scene={scene} samples={info['samples']} "
            f"valid_stop={info['Valid samples for stop metrics']} "
            f"NE/m={info['NE/m']:.4f} SR/%={info['SR/%']:.2f} OSR/%={info['OSR/%']:.2f} "
            f"SPL/%={info['SPL/%']:.2f} SPL_openfly/%={info['SPL_openfly/%']:.2f} "
            f"SPL_standard/%={info['SPL_standard/%']:.2f} "
            f"PredStop={info['Pred stop count']} PredStopRate={info['Pred stop rate']:.4f} "
            f"SuccessfulStop={info['Successful stop count']} SuccessfulStopRate={info['Successful stop rate']:.4f} "
            f"ForcedEnd={info['Forced end count']} ForcedEndRate={info['Forced end rate']:.4f} "
            f"ImageError={info['Image error count']}"
        )

    metrics_root = _Path(_os.environ.get("EVAL_METRICS_ROOT", "runs/eval_metrics"))
    metrics_root.mkdir(parents=True, exist_ok=True)
    safe_tag = str(metrics_tag).replace("/", "_").replace("\\", "_").replace(" ", "_")

    episode_csv = metrics_root / f"{safe_tag}_episode_metrics.csv"
    scene_csv = metrics_root / f"{safe_tag}_scene_metrics.csv"
    env_scene_csv = metrics_root / f"{safe_tag}_env_scene_metrics.csv"
    summary_json = metrics_root / f"{safe_tag}_metric_summary.json"

    episode_fields = [
        "env", "scene", "sample_idx", "NE/m", "SR", "OSR",
        "SPL/%", "SPL_openfly/%", "SPL_standard/%", "SPL_raw/%", "SPL_bounded/%",
        "SPL", "SPL_openfly", "SPL_standard", "SPL_raw", "SPL_bounded",
        "valid_stop", "pred_stop", "successful_stop", "forced_end", "image_error",
    ]
    with open(episode_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=episode_fields)
        w.writeheader()
        for r in nav_episode_metrics:
            rr = dict(r)
            rr["SPL/%"] = 100.0 * rr.get("SPL_standard", rr.get("SPL_bounded", rr.get("SPL", 0.0)))
            rr["SPL_openfly/%"] = 100.0 * rr.get("SPL_openfly", rr.get("SPL_raw", 0.0))
            rr["SPL_standard/%"] = 100.0 * rr.get("SPL_standard", rr.get("SPL_bounded", rr.get("SPL", 0.0)))
            rr["SPL_raw/%"] = 100.0 * rr.get("SPL_raw", rr.get("SPL_openfly", 0.0))
            rr["SPL_bounded/%"] = 100.0 * rr.get("SPL_bounded", rr.get("SPL_standard", rr.get("SPL", 0.0)))
            w.writerow({k: rr.get(k, "") for k in episode_fields})

    scene_fields = [
        "scene", "samples", "Total samples", "Valid samples for stop metrics",
        "NE/m", "SR/%", "OSR/%", "SPL/%", "SPL_openfly/%", "SPL_standard/%", "SPL_raw/%", "SPL_bounded/%",
        "Image error count", "Image error rate", "Pred stop count", "Pred stop rate",
        "Successful stop count", "Successful stop rate", "Forced end count", "Forced end rate",
    ]
    with open(scene_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=scene_fields)
        w.writeheader()
        for r in scene_rows:
            w.writerow({k: r.get(k, "") for k in scene_fields})

    env_scene_fields = ["env"] + scene_fields
    with open(env_scene_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=env_scene_fields)
        w.writeheader()
        for r in env_scene_rows:
            w.writerow({k: r.get(k, "") for k in env_scene_fields})

    with open(summary_json, "w", encoding="utf-8") as f:
        _json.dump({
            "total": total,
            "scene_metrics": scene_rows,
            "env_scene_metrics": env_scene_rows,
            "episode_metrics_csv": str(episode_csv),
            "scene_metrics_csv": str(scene_csv),
            "env_scene_metrics_csv": str(env_scene_csv),
        }, f, ensure_ascii=False, indent=2)

    print(f"[MetricFiles] episode_metrics_csv={episode_csv}")
    print(f"[MetricFiles] scene_metrics_csv={scene_csv}")
    print(f"[MetricFiles] env_scene_metrics_csv={env_scene_csv}")
    print(f"[MetricFiles] summary_json={summary_json}")

    return total, scene_rows, env_scene_rows
# =========================
# End OpenFly paper metric helpers
# =========================

import psutil
import requests
import random
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
import socket
import signal
from pathlib import Path

from keyframe.keyframe_scorer import AttentionKeyframeScorer, KeyframeScorerConfig
from keyframe.simple_tokenizer import SimpleTextTokenizer
from lora_utils import load_lora_adapter_for_eval
from openfly_trend_decoder_runtime import (
    OpenFlyTrendDecoderRuntime as OpenFlyFeatureDecoderRuntime,
    select_final_action,
)
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openfly_trend_state_runtime import OpenFlyTrendStateRuntime
from openfly_common import get_eval_image_dir, save_eval_task_summary
from openfly_eval_utils import get_eval_split_name, load_eval_info


AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

def load_openfly_model_local_first(
    device="cuda:0",
    dtype=torch.bfloat16,
):
    """
    Load OpenFly-Agent with local-first strategy.

    1. Try local snapshot:
       models/openfly-agent-7b

    2. If local loading fails and HF_FALLBACK_REMOTE=1, try:
       IPEC-COMMUNITY/openfly-agent-7b

    Set HF_FALLBACK_REMOTE=0 for strict offline reproducibility.
    """
    local_model_path = os.environ.get(
        "OPENFLY_LOCAL_MODEL_PATH",
        str(_PROJECT_ROOT / "models" / "openfly-agent-7b"),
    )
    remote_model_id = os.environ.get(
        "OPENFLY_REMOTE_MODEL_ID",
        "IPEC-COMMUNITY/openfly-agent-7b",
    )
    fallback_remote = os.environ.get("HF_FALLBACK_REMOTE", "0") == "1"

    processor_kwargs = dict(
        trust_remote_code=True,
    )

    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )

    print(f"[Load OpenFly] try local first: {local_model_path}", flush=True)

    try:
        processor = AutoProcessor.from_pretrained(
            local_model_path,
            local_files_only=True,
            **processor_kwargs,
        )
        policy = AutoModelForVision2Seq.from_pretrained(
            local_model_path,
            local_files_only=True,
            **model_kwargs,
        ).to(device)
        print(f"[Load OpenFly] loaded from local: {local_model_path}", flush=True)
        return processor, policy
    except Exception as e:
        print(f"[Load OpenFly][WARN] local load failed: {repr(e)}", flush=True)

        if not fallback_remote:
            raise RuntimeError(
                "Local OpenFly load failed and HF_FALLBACK_REMOTE=0. "
                "Check OPENFLY_LOCAL_MODEL_PATH."
            ) from e

    print(f"[Load OpenFly] fallback to remote: {remote_model_id}", flush=True)

    processor = AutoProcessor.from_pretrained(
        remote_model_id,
        local_files_only=False,
        **processor_kwargs,
    )
    policy = AutoModelForVision2Seq.from_pretrained(
        remote_model_id,
        local_files_only=False,
        **model_kwargs,
    ).to(device)

    print(f"[Load OpenFly] loaded from remote/cache: {remote_model_id}", flush=True)
    return processor, policy




# =========================
# Global switches
# =========================
# Modes:
#   residual  : source-compatible residual baseline [older, previous, current]
#   heuristic : old heuristic keyframe scorer [keyframe, previous, current]
#   learned   : learned keyframe scorer [learned_keyframe, previous, current]
KEYFRAME_MODE = os.environ.get("KEYFRAME_MODE", "residual").lower()

SAVE_FRAME_DEBUG = os.environ.get("SAVE_FRAME_DEBUG", "0") == "1"
SAVE_KEYFRAME_DEBUG = os.environ.get("SAVE_KEYFRAME_DEBUG", "0") == "1"

# Heuristic scorer params.
KEYFRAME_MAX_CANDIDATES = int(os.environ.get("KEYFRAME_MAX_CANDIDATES", "6"))

# Learned scorer params.
LEARNED_KEYFRAME_CHECKPOINT = os.environ.get(
    "LEARNED_KEYFRAME_CHECKPOINT",
    str(_PROJECT_ROOT / "runs" / "code" / "STEER-VLN" / "train" / "final_model" / "keyframe_scorer_best.pt"),
)
LEARNED_KEYFRAME_VOCAB = os.environ.get(
    "LEARNED_KEYFRAME_VOCAB",
    str(_PROJECT_ROOT / "runs" / "code" / "STEER-VLN" / "train" / "final_model" / "simple_tokenizer_vocab.json"),
)
LEARNED_KEYFRAME_MAX_HISTORY = int(os.environ.get("LEARNED_KEYFRAME_MAX_HISTORY", "8"))
LEARNED_KEYFRAME_IMAGE_SIZE = int(os.environ.get("LEARNED_KEYFRAME_IMAGE_SIZE", "224"))
LEARNED_KEYFRAME_EXCLUDE_PREVIOUS = os.environ.get("LEARNED_KEYFRAME_EXCLUDE_PREVIOUS", "1") == "1"
LEARNED_KEYFRAME_DEBUG_SCORE = os.environ.get("LEARNED_KEYFRAME_DEBUG_SCORE", "0") == "1"
LEARNED_KEYFRAME_TOPK = int(os.environ.get("LEARNED_KEYFRAME_TOPK", "3"))

DEBUG_SAVE_STEPS = int(os.environ.get("DEBUG_SAVE_STEPS", "5"))

_LEARNED_KEYFRAME_RUNTIME = None

# =========================
# Waypoint decoder / hybrid action mode
# =========================
ACTION_MODE = os.environ.get("ACTION_MODE", "openfly").lower()

WAYPOINT_FEATURE_DUAL_CHECKPOINT = os.environ.get(
    "M3C_TREND_HEAD_CHECKPOINT",
    os.environ.get(
        "WAYPOINT_FEATURE_DUAL_CHECKPOINT",
        str(_PROJECT_ROOT / "runs" / "code" / "STEER-VLN" / "train" / "final_model" / "m3c_trend_head_best.pt"),
    ),
)

WAYPOINT_D_SCALE = float(os.environ.get("WAYPOINT_D_SCALE", "12.0"))
WAYPOINT_Z_SCALE = float(os.environ.get("WAYPOINT_Z_SCALE", "5.0"))
WAYPOINT_YAW_TURN_THRESHOLD = float(os.environ.get("WAYPOINT_YAW_TURN_THRESHOLD", "0.35"))
WAYPOINT_Z_THRESHOLD = float(os.environ.get("WAYPOINT_Z_THRESHOLD", "1.0"))
WAYPOINT_USE_Z_DECODE = os.environ.get("WAYPOINT_USE_Z_DECODE", "1") == "1"
WAYPOINT_FORWARD_POLICY = os.environ.get("WAYPOINT_FORWARD_POLICY", "distance_bins")
WAYPOINT_FORWARD_3_THRESHOLD = float(os.environ.get("WAYPOINT_FORWARD_3_THRESHOLD", "4.5"))
WAYPOINT_FORWARD_6_THRESHOLD = float(os.environ.get("WAYPOINT_FORWARD_6_THRESHOLD", "8.0"))
WAYPOINT_STOP_PROB_THRESHOLD = float(os.environ.get("WAYPOINT_STOP_PROB_THRESHOLD", "0.50"))
WAYPOINT_STOP_LOGIT_MARGIN = float(os.environ.get("WAYPOINT_STOP_LOGIT_MARGIN", "0.0"))
WAYPOINT_DEBUG = os.environ.get("WAYPOINT_DEBUG", "0") == "1"

TREND_STOP_THRESHOLD = float(os.environ.get("TREND_STOP_THRESHOLD", "0.50"))
TREND_PRESTOP_THRESHOLD = float(os.environ.get("TREND_PRESTOP_THRESHOLD", "0.50"))
TREND_STOP_USE_MARGIN = os.environ.get("TREND_STOP_USE_MARGIN", "0") == "1"
TREND_STOP_MARGIN_THRESHOLD = float(os.environ.get("TREND_STOP_MARGIN_THRESHOLD", "-20.0"))
TREND_STOP_USE_DISTANCE = os.environ.get("TREND_STOP_USE_DISTANCE", "0") == "1"
TREND_STOP_DISTANCE_THRESHOLD = float(os.environ.get("TREND_STOP_DISTANCE_THRESHOLD", "999.0"))

_WAYPOINT_DECODER_RUNTIME = None

# Final STEER-VLN: trend head provides auxiliary state; never overrides action.
TREND_CONDITIONED = os.environ.get('TREND_CONDITIONED', '0') == '1'
TREND_STATE_TEMPORAL_WINDOW = int(os.environ.get('TREND_STATE_TEMPORAL_WINDOW', '3'))
TREND_STATE_STOP_THRESHOLD = float(os.environ.get('TREND_STATE_STOP_THRESHOLD', '0.50'))
TREND_STATE_PRESTOP_THRESHOLD = float(os.environ.get('TREND_STATE_PRESTOP_THRESHOLD', '0.50'))
TREND_STATE_DEBUG = os.environ.get('TREND_STATE_DEBUG', '1') == '1'
_TREND_STATE_RUNTIME = None


def kill_env_process(keyword):
    """Kill all matching simulator processes, not only the newest one.

    The original eval used `pgrep -n`, which only kills one process.
    That can leave old AirVLN/CitySample/CrashReport processes and stale
    UnrealCV TCP connections in CLOSE-WAIT, causing the next environment to
    hang even when the port appears to be listening.
    """
    try:
        result = subprocess.run(
            ['pgrep', '-f', keyword],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pids = []
        for x in result.stdout.split():
            try:
                pid = int(x)
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)

        if not pids:
            return

        print(f"[kill_env_process] keyword={keyword}, pids={pids}", flush=True)
        for pid in sorted(set(pids)):
            subprocess.run(['kill', '-9', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[kill_env_process][WARN] keyword={keyword}, error={repr(e)}", flush=True)


class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name

        # AirSim/UE startup can be slow. Make waiting/retry configurable.
        self.airsim_ip = os.environ.get("AIRSIM_IP", "")
        self.airsim_port = int(os.environ.get("AIRSIM_PORT", "41451"))
        self.startup_wait = float(os.environ.get("AIRSIM_STARTUP_WAIT", "30"))
        self.connect_max_wait = float(os.environ.get("AIRSIM_CONNECT_MAX_WAIT", "300"))
        self.connect_retry_interval = float(os.environ.get("AIRSIM_CONNECT_RETRY_INTERVAL", "5"))
        self.skip_launch = os.environ.get("AIRSIM_SKIP_LAUNCH", "0") == "1"

        if not self.skip_launch:
            self._sim_thread = threading.Thread(
                target=self._init_airsim_sim,
                daemon=True,
            )
            self._sim_thread.start()

            print(
                f"[AirSim] Launching env={env_name}, initial wait {self.startup_wait:.1f}s...",
                flush=True,
            )
            time.sleep(self.startup_wait)
        else:
            print(
                f"[AirSim] AIRSIM_SKIP_LAUNCH=1, will only connect to existing AirSim.",
                flush=True,
            )

        self._client = self._connect_airsim_with_retry()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        self.distance_to_goal = []
        self.spl = []
        self.spl_raw = []
        self.spl_bounded = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def _connect_airsim_with_retry(self):
        deadline = time.time() + self.connect_max_wait
        attempt = 0
        last_error = None

        print(
            f"[AirSim] Connecting to ip='{self.airsim_ip or 'default'}', "
            f"port={self.airsim_port}, max_wait={self.connect_max_wait:.1f}s",
            flush=True,
        )

        while time.time() < deadline:
            attempt += 1
            try:
                client = airsim.MultirotorClient(
                    ip=self.airsim_ip,
                    port=self.airsim_port,
                )
                client.confirmConnection()
                print(f"[AirSim] Connected after attempt {attempt}.", flush=True)
                return client
            except Exception as e:
                last_error = e
                remain = max(0.0, deadline - time.time())
                print(
                    f"[AirSim] connect attempt {attempt} failed: {repr(e)}; "
                    f"retry in {self.connect_retry_interval:.1f}s; remain={remain:.1f}s",
                    flush=True,
                )
                time.sleep(self.connect_retry_interval)

        raise RuntimeError(
            f"AirSim connection failed after {self.connect_max_wait:.1f}s. "
            f"ip={self.airsim_ip or 'default'}, port={self.airsim_port}, "
            f"last_error={repr(last_error)}"
        )

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()

    def print_info(self):
        _spl_raw_list = getattr(self, "spl_raw", self.spl)
        _spl_bounded_list = getattr(self, "spl_bounded", self.spl)
        _spl_raw = _spl_raw_list[-1] if _spl_raw_list else self.spl[-1]
        _spl_bounded = _spl_bounded_list[-1] if _spl_bounded_list else self.spl[-1]
        print(
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )
        return (
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(
            airsim.Vector3r(x, -y, -z),
            airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw))
        )
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(
            airsim.Vector3r(x, y, z),
            airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3])
        )
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(1)

    def _camera_init(self):
        camera_pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(math.radians(15), 0, 0)
        )
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type='color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages(
            [airsim.ImageRequest('front_custom', image_type, False, False)]
        )
        response = responses[0]

        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")


class UEBridge:
    """Robust UE/UnrealCV bridge for CitySample environments.

    Fixes compared with the source-style implementation:
    1. use a deterministic UnrealCV port by default, instead of random 9000-9100;
    2. kill all stale UE/CrashReport processes before launch;
    3. remove stale /tmp/unrealcv_<port>.socket before launch;
    4. normalize DISPLAY=localhost:10.0 -> DISPLAY=:10 for SSH sessions;
    5. wait until UnrealCV status responds before continuing.
    """

    def __init__(self, ue_ip, ue_port, env_name):
        self.env_name = env_name
        self.process = None
        self._client = None

        self.ue_ip = os.environ.get("UE_IP", str(ue_ip or "127.0.0.1"))
        self.ue_port = int(os.environ.get("UE_UNREALCV_PORT", "9048"))
        self.startup_wait = float(os.environ.get("UE_STARTUP_WAIT", "30"))
        self.connect_max_wait = float(os.environ.get("UE_CONNECT_MAX_WAIT", "180"))
        self.connect_retry_interval = float(os.environ.get("UE_CONNECT_RETRY_INTERVAL", "5"))
        self.connect_timeout = int(float(os.environ.get("UE_CONNECT_TIMEOUT", "20")))
        self.request_timeout = int(float(os.environ.get("UE_REQUEST_TIMEOUT", "20")))
        self.skip_launch = os.environ.get("UE_SKIP_LAUNCH", "0") == "1"

        if not self.skip_launch:
            self.kill_failed_process()
            self._cleanup_unrealcv_socket(self.ue_port)
            time.sleep(float(os.environ.get("UE_KILL_WAIT", "3")))
        else:
            print("[UEBridge] UE_SKIP_LAUNCH=1, keep existing CitySample/UnrealCV process.", flush=True)

        print(f"[UEBridge] env={env_name}, ip={self.ue_ip}, port={self.ue_port}", flush=True)
        self.modify_port_in_ini(self.ue_port, env_name)

        if not self.skip_launch:
            self._sim_thread = threading.Thread(target=self._init_ue_sim, daemon=True)
            self._sim_thread.start()
            print(f"[UEBridge] Launching {env_name}, initial wait {self.startup_wait:.1f}s...", flush=True)
            time.sleep(self.startup_wait)
        else:
            print("[UEBridge] UE_SKIP_LAUNCH=1, will only connect to existing UnrealCV.", flush=True)

        self._client = self._connect_unrealcv_with_retry(self.ue_ip, self.ue_port)
        self._camera_init()

        self.distance_to_goal = []
        self.spl = []
        self.spl_raw = []
        self.spl_bounded = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        _spl_raw_list = getattr(self, "spl_raw", self.spl)
        _spl_bounded_list = getattr(self, "spl_bounded", self.spl)
        _spl_raw = _spl_raw_list[-1] if _spl_raw_list else self.spl[-1]
        _spl_bounded = _spl_bounded_list[-1] if _spl_bounded_list else self.spl[-1]
        print(
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )
        return (
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )

    def find_available_port(self):
        # Kept for compatibility, but the final STEER_VLN eval uses fixed 9048 by default.
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()
            if 'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        if not os.path.exists(ini_file):
            raise FileNotFoundError(f"UnrealCV ini not found: {ini_file}")

        with open(ini_file, 'r') as file:
            lines = file.readlines()

        has_port = False
        with open(ini_file, 'w') as file:
            for line in lines:
                if line.startswith("Port="):
                    file.write(f"Port={int(port)}\n")
                    has_port = True
                else:
                    file.write(line)
            if not has_port:
                file.write(f"Port={int(port)}\n")

    def kill_failed_process(self):
        # Kill all possible UE leftovers. One stale CitySample can keep 9048 in CLOSE-WAIT.
        for keyword in ["CrashReport", "CitySample", "City_UE52"]:
            kill_env_process(keyword)

    def _cleanup_unrealcv_socket(self, port):
        sock_path = f"/tmp/unrealcv_{int(port)}.socket"
        try:
            if os.path.exists(sock_path):
                os.remove(sock_path)
                print(f"[UEBridge] removed stale socket: {sock_path}", flush=True)
        except Exception as e:
            print(f"[UEBridge][WARN] cannot remove {sock_path}: {repr(e)}", flush=True)

    def _normalize_display_env(self):
        env = os.environ.copy()
        display = env.get("DISPLAY", "")
        if display.startswith("localhost:"):
            d = display.split("localhost:", 1)[1].split(".", 1)[0]
            env["DISPLAY"] = f":{d}"
        elif not display:
            if os.path.exists("/tmp/.X11-unix/X10"):
                env["DISPLAY"] = ":10"
            elif os.path.exists("/tmp/.X11-unix/X0"):
                env["DISPLAY"] = ":0"

        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        return env

    def _init_ue_sim(self):
        env_dir = "envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = [
            "bash",
            f"{env_dir}/CitySample.sh",
            "-windowed",
            "-ResX=960",
            "-ResY=540",
            "-ForceRes",
            "-NoSound",
            "-NoVSync",
        ]

        log_dir = Path("runs/STEER-VLN/logs/env_debug")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.env_name}_uebridge_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
        env = self._normalize_display_env()

        print(f"[UEBridge] command={' '.join(command)}", flush=True)
        print(f"[UEBridge] DISPLAY={env.get('DISPLAY', '')}", flush=True)
        print(f"[UEBridge] XDG_RUNTIME_DIR={env.get('XDG_RUNTIME_DIR', '')}", flush=True)
        print(f"[UEBridge] stdout/stderr -> {log_path}", flush=True)

        with open(log_path, "ab") as log_f:
            self.process = subprocess.Popen(
                command,
                text=False,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            self.process.wait()

    def _alarm_call(self, fn, timeout_s, desc):
        # UnrealCV Client can block indefinitely when its TCP queue is stuck.
        # On Linux/main thread, SIGALRM prevents eval from hanging forever.
        if threading.current_thread() is not threading.main_thread():
            return fn()

        def _handler(signum, frame):
            raise TimeoutError(f"{desc} timed out after {timeout_s}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(max(1, int(timeout_s)))
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _request_unrealcv(self, command, timeout=None, desc=None):
        timeout = self.request_timeout if timeout is None else timeout
        desc = desc or command
        return self._alarm_call(
            lambda: self._client.request(command),
            timeout,
            f"UnrealCV request: {desc}",
        )

    def _probe_tcp_ready(self, host, port, timeout_s=3.0):
        # Do NOT open a raw TCP probe here. UnrealCV sends a one-shot
        # connection-confirm message (e.g. b"connected to City_UE52") when a
        # client connects. If a raw socket consumes that confirm and closes, the
        # following official unrealcv.Client may hang and the server can enter
        # CLOSE-WAIT/LISTEN-queue-full states. Keep this method only for
        # backward-compatible naming; readiness is checked by Client.connect() +
        # vget /unrealcv/status below.
        return True

    def _connect_unrealcv_with_retry(self, ue_ip, ue_port):
        deadline = time.time() + self.connect_max_wait
        attempt = 0
        last_error = None

        while time.time() < deadline:
            attempt += 1
            try:
                # Do not pre-connect with a raw socket. The unrealcv.Client
                # must be the first consumer of UnrealCV's connection-confirm
                # handshake for this attempt.
                client = Client((ue_ip, int(ue_port)))
                self._alarm_call(lambda: client.connect(), self.connect_timeout, "UnrealCV connect")

                if hasattr(client, "isconnected") and not client.isconnected():
                    raise RuntimeError("UnrealCV client reports isconnected=False")

                status = self._alarm_call(
                    lambda: client.request('vget /unrealcv/status'),
                    self.connect_timeout,
                    "UnrealCV status request",
                )
                print(f"[UEBridge] UnrealCV connected after attempt {attempt}.", flush=True)
                print(f"[UEBridge] status: {status}", flush=True)
                return client

            except Exception as e:
                last_error = e
                remain = max(0.0, deadline - time.time())
                print(
                    f"[UEBridge] connect attempt {attempt} failed: {repr(e)}; "
                    f"retry in {self.connect_retry_interval:.1f}s; remain={remain:.1f}s",
                    flush=True,
                )
                try:
                    if 'client' in locals():
                        client.disconnect()
                except Exception:
                    pass
                time.sleep(self.connect_retry_interval)

        raise RuntimeError(
            f"UnrealCV connection failed after {self.connect_max_wait:.1f}s. "
            f"env={self.env_name}, ip={ue_ip}, port={ue_port}, last_error={repr(last_error)}"
        )

    def __del__(self):
        try:
            if getattr(self, "_client", None) is not None:
                self._client.disconnect()
        except Exception:
            pass

    def _connection_check(self):
        # Compatibility method. New code uses _connect_unrealcv_with_retry().
        if self._client is not None and getattr(self._client, "isconnected", lambda: False)():
            print('UnrealCV connected successfully')
        else:
            raise RuntimeError('UnrealCV is not connected')

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._request_unrealcv('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']), desc='camera 0 location')
        self._request_unrealcv('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']), desc='camera 1 location')
        self._request_unrealcv('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']), desc='camera 0 rotation')
        self._request_unrealcv('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']), desc='camera 1 rotation')
        print('camera_settings', camera_settings)

    def _camera_init(self):
        time.sleep(2)
        self._request_unrealcv('vset /cameras/spawn', desc='spawn cameras')
        self._request_unrealcv('vset /camera/1/size 1920 1080', desc='camera 1 size')
        time.sleep(2)
        self.set_camera_pose(150, 400, 15, 0, 0, 0)
        time.sleep(2)

    def get_camera_data(self, camera_type='lit'):
        valid_types = {'lit', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'lit':
            data = self._client.request('vget /camera/1/lit png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'object_mask':
            data = self._client.request('vget /camera/1/object_mask png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'depth':
            data = self._client.request('vget /camera/1/depth npy')
            depth_np = np.load(io.BytesIO(data))
            return depth_np

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='lit'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)



def get_gs_pos_ratio(env_name):
    env_name = str(env_name)
    env_key = env_name.upper().replace("-", "_")
    for k in (f"OPENFLY_GS_POS_RATIO_{env_key}", f"GS_POS_RATIO_{env_key}", "GS_POS_RATIO"):
        v = os.environ.get(k)
        if v:
            try:
                return float(v)
            except Exception:
                pass
    cfg = Path("configs") / f"{env_name}.yaml"
    if cfg.exists():
        try:
            m = re.search(r"pcd_scale_ratio\s*:\s*([0-9.]+)", cfg.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return {
        "env_gs_ecust": 5.6,
        "env_gs_nwpu01": 6.65,
        "env_gs_nwpu02": 5.15,
        "env_gs_sjtu01": 5.42,
        "env_gs_sjtu02": 4.75,
    }.get(env_name, 5.15)

class GSBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)

        self.distance_to_goal = []
        self.spl = []
        self.spl_raw = []
        self.spl_bounded = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        _spl_raw_list = getattr(self, "spl_raw", self.spl)
        _spl_bounded_list = getattr(self, "spl_bounded", self.spl)
        _spl_raw = _spl_raw_list[-1] if _spl_raw_list else self.spl[-1]
        _spl_bounded = _spl_bounded_list[-1] if _spl_bounded_list else self.spl[-1]
        print(
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )
        return (
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )

    def _init_gs_sim(self):
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"
        if not os.path.exists(dataset_dir):
            raise ValueError(
                f"Specified 3DGS directory for {self.env_name} does not exist: {dataset_dir}. "
                f"Set OPENFLY_GS_DATASET_DIR_{_gs_key}=<scene_reconstruction_dir> "
                "or OPENFLY_GS_DATA_ROOT=<root_containing_gs_scenes>."
            )
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw

    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params=None):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw)
        )
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        if path_params is None:
            path_params = os.environ.get("GS_RENDER_OUTPUT_DIR", "runs/STEER-VLN/gs_render_tmp")
        render_dir = Path(path_params)
        render_dir.mkdir(parents=True, exist_ok=True)
        self._last_render_dir = str(render_dir)
        _render_before = time.time()
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text)
                self._load_latest_gs_render(render_dir, after_mtime=_render_before)
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__(self.env_name)
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(30)

    def _load_latest_gs_render(self, render_dir, after_mtime=0.0, timeout=None):
        timeout = float(os.environ.get("GS_RENDER_TIMEOUT", "30")) if timeout is None else float(timeout)
        deadline = time.time() + timeout
        render_dir = Path(render_dir)
        latest = None
        while time.time() < deadline:
            files = []
            if render_dir.exists():
                files = [p for p in render_dir.glob("*.png") if p.stat().st_mtime >= after_mtime]
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                img = cv2.imread(str(latest), cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    self._last_image = img
                    self._last_image_path = str(latest)
                    return img
            time.sleep(0.2)
        raise RuntimeError(f"GS render succeeded but no PNG was found under {render_dir}; latest={latest}")

    def get_camera_data(self, camera_type='color'):
        img = getattr(self, "_last_image", None)
        if img is None:
            raise RuntimeError("GSBridge has no rendered image. set_camera_pose(..., path_params=...) must run first.")
        return img.copy()

    def process_camera_data(self, file_path):
        img = self.get_camera_data()
        cv2.imwrite(str(file_path), img)


def validate_frame_bundle(frame_bundle):
    if not isinstance(frame_bundle, list):
        raise TypeError(f"frame_bundle must be a list, got {type(frame_bundle)}")
    if len(frame_bundle) != 3:
        raise ValueError(f"frame_bundle must contain exactly 3 frames, got {len(frame_bundle)}")

    ref_shape = None
    for idx, frame in enumerate(frame_bundle):
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Frame {idx} must be np.ndarray, got {type(frame)}")
        if frame.size == 0:
            raise ValueError(f"Frame {idx} is empty")
        if frame.ndim not in (2, 3):
            raise ValueError(f"Frame {idx} has invalid ndim={frame.ndim}, expected 2 or 3")
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
            raise ValueError(f"Frame {idx} has invalid channel count={frame.shape[2]}, expected 1/3/4")

        if ref_shape is None:
            ref_shape = frame.shape[:2]
        elif frame.shape[:2] != ref_shape:
            raise ValueError(f"Frame {idx} spatial shape mismatch: {frame.shape[:2]} vs {ref_shape}")


def prepare_images_for_processor(frame_bundle):
    images = []
    for frame in frame_bundle:
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(frame))
    return images



def build_current_only_frame_bundle(history_buffer):
    if len(history_buffer) == 0:
        raise ValueError("history_buffer is empty")
    current = history_buffer[-1]
    bundle = [current, current, current]
    validate_frame_bundle(bundle)
    return bundle


def build_residual_frame_bundle(history_buffer):
    """
    Source-compatible baseline order:
    [older, previous, current]
    """
    if len(history_buffer) == 0:
        raise ValueError("history_buffer is empty")

    if len(history_buffer) == 1:
        f0 = history_buffer[0]
        bundle = [f0, f0, f0]
    elif len(history_buffer) == 2:
        f0, f1 = history_buffer[0], history_buffer[1]
        bundle = [f0, f0, f1]
    else:
        bundle = [
            history_buffer[-3],
            history_buffer[-2],
            history_buffer[-1],
        ]

    validate_frame_bundle(bundle)
    return bundle


def compute_frame_difference(frame_a, frame_b, size=(160, 90)):
    a = cv2.resize(frame_a, size)
    b = cv2.resize(frame_b, size)

    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
    return float(diff)


def compute_center_difference(frame_a, frame_b, size=(160, 90)):
    a = cv2.resize(frame_a, size)
    b = cv2.resize(frame_b, size)

    h, w = a.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)

    a = a[y1:y2, x1:x2]
    b = b[y1:y2, x1:x2]

    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
    return float(diff)


def score_candidate_keyframe(candidate, current, previous, age_rank):
    diff_to_current = compute_frame_difference(candidate, current)
    diff_to_previous = compute_frame_difference(candidate, previous)
    center_diff = compute_center_difference(candidate, current)

    score = (
        0.6 * diff_to_current +
        0.3 * diff_to_previous +
        0.2 * center_diff -
        0.1 * age_rank
    )
    return float(score)


def select_keyframe_from_history(history_buffer, max_candidates=3):
    n = len(history_buffer)

    if n == 0:
        raise ValueError("history_buffer is empty")
    if n == 1:
        return history_buffer[0]
    if n == 2:
        return history_buffer[0]
    if n == 3:
        return history_buffer[0]

    current = history_buffer[-1]
    previous = history_buffer[-2]
    candidates = history_buffer[max(0, n - 2 - max_candidates):-2]

    if len(candidates) == 0:
        return history_buffer[0]

    best_score = -1e9
    best_frame = candidates[0]

    for i, candidate in enumerate(candidates):
        age_rank = len(candidates) - i
        score = score_candidate_keyframe(candidate, current, previous, age_rank)
        if score > best_score:
            best_score = score
            best_frame = candidate

    return best_frame


def build_keyframe_frame_bundle(history_buffer, max_candidates=3):
    """
    Source-compatible keyframe order:
    [keyframe, previous, current]

    Fallback:
    - len(history_buffer) < 4 => return source-compatible residual baseline
    """
    if len(history_buffer) == 0:
        raise ValueError("history_buffer is empty")

    if len(history_buffer) < 4:
        return build_residual_frame_bundle(history_buffer)

    current = history_buffer[-1]
    previous = history_buffer[-2]
    keyframe = select_keyframe_from_history(history_buffer, max_candidates=max_candidates)
    bundle = [keyframe, previous, current]

    validate_frame_bundle(bundle)
    return bundle


def save_frame_bundle_debug(frame_bundle, step, sample_idx=None, env_name=None, save_dir="test/frame_debug"):
    os.makedirs(save_dir, exist_ok=True)

    prefix = f"step_{step:03d}"
    if sample_idx is not None:
        prefix = f"sample_{sample_idx:04d}_" + prefix
    if env_name is not None:
        prefix = f"{env_name}_" + prefix

    names = ["first", "second", "third"]
    for frame, name in zip(frame_bundle, names):
        out_path = os.path.join(save_dir, f"{prefix}_{name}.jpg")
        cv2.imwrite(out_path, frame)


def save_keyframe_debug(first, second, third, step, sample_idx=None, env_name=None, save_dir="test/keyframe_debug"):
    os.makedirs(save_dir, exist_ok=True)

    prefix = f"step_{step:03d}"
    if sample_idx is not None:
        prefix = f"sample_{sample_idx:04d}_" + prefix
    if env_name is not None:
        prefix = f"{env_name}_" + prefix

    cv2.imwrite(os.path.join(save_dir, f"{prefix}_first.jpg"), first)
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_second.jpg"), second)
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_third.jpg"), third)



def wrap_angle_for_keyframe(x: float) -> float:
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


def frame_to_keyframe_tensor(frame: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """
    Convert simulator frame ndarray to scorer normalized tensor [3, H, W].

    AirSim/UE get_camera_data usually returns OpenCV-style BGR arrays.
    The scorer was trained with PIL RGB images normalized by ImageNet mean/std.
    """
    if frame.ndim == 2:
        frame_rgb = np.stack([frame, frame, frame], axis=-1)
    else:
        frame_rgb = frame[:, :, :3]

    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

    # BGR -> RGB for cv2-style images.
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame_rgb).resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr).float()


def motion_feat_for_keyframe(candidate_pose, current_pose):
    """
    pose format: [x, y, z, yaw]
    Match training normalization:
      dx/dy/dz / 50
      dyaw / pi
    """
    cx, cy, cz, cyaw = current_pose
    hx, hy, hz, hyaw = candidate_pose

    dx = float((cx - hx) / 50.0)
    dy = float((cy - hy) / 50.0)
    dz = float((cz - hz) / 50.0)
    dyaw = wrap_angle_for_keyframe(float(cyaw - hyaw)) / math.pi

    return [dx, dy, dz, float(dyaw)]


class LearnedKeyframeRuntime:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "cuda:0",
        image_size: int = 224,
        max_history: int = 8,
    ):
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.max_history = max_history

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Learned keyframe checkpoint not found: {checkpoint_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Learned keyframe vocab not found: {vocab_path}")

        self.tokenizer = SimpleTextTokenizer.load(vocab_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = dict(ckpt["cfg"])

        # Keep runtime compatible with training checkpoint.
        cfg_dict["vocab_size"] = len(self.tokenizer)
        cfg_dict["image_size"] = image_size
        cfg_dict["max_history"] = max_history

        self.cfg = KeyframeScorerConfig(**cfg_dict)
        self.model = AttentionKeyframeScorer(self.cfg, padding_idx=self.tokenizer.pad_token_id)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        print("[LearnedKeyframeRuntime] loaded")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  vocab:      {vocab_path}")
        print(f"  device:     {self.device}")
        print(f"  image_size: {self.image_size}")
        print(f"  max_history:{self.max_history}")

    @torch.no_grad()
    def score_candidates(
        self,
        history_buffer,
        pose_history,
        text: str,
        exclude_previous: bool = True,
    ):
        """
        history_buffer includes current frame as last element.
        pose_history includes current pose as last element.

        Returns:
          selected_frame, selected_local_idx, selected_global_idx, probs, candidate_indices
        """
        n = len(history_buffer)

        if n == 0:
            raise ValueError("history_buffer is empty")

        if n < 4:
            return None

        current_frame = history_buffer[-1]
        current_pose = pose_history[-1]

        # Candidate range:
        #   exclude_previous=True  -> candidates before previous frame, matching old heuristic bundle semantics.
        #   exclude_previous=False -> candidates may include previous frame, matching scorer training/eval setting.
        if exclude_previous:
            end_exclusive = n - 2
        else:
            end_exclusive = n - 1

        start = max(0, end_exclusive - self.max_history)
        candidate_indices = list(range(start, end_exclusive))

        if len(candidate_indices) == 0:
            return None

        history_tensors = []
        motion_feats = []
        history_mask = []

        for j in candidate_indices:
            history_tensors.append(frame_to_keyframe_tensor(history_buffer[j], self.image_size))
            motion_feats.append(motion_feat_for_keyframe(pose_history[j], current_pose))
            history_mask.append(True)

        candidate_with_pad = list(candidate_indices)

        while len(history_tensors) < self.max_history:
            history_tensors.insert(0, torch.zeros(3, self.image_size, self.image_size))
            motion_feats.insert(0, [0.0, 0.0, 0.0, 0.0])
            history_mask.insert(0, False)
            candidate_with_pad.insert(0, None)

        if len(history_tensors) > self.max_history:
            history_tensors = history_tensors[-self.max_history:]
            motion_feats = motion_feats[-self.max_history:]
            history_mask = history_mask[-self.max_history:]
            candidate_with_pad = candidate_with_pad[-self.max_history:]

        encoded = self.tokenizer(
            str(text).lower(),
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding=False,
            return_attention_mask=True,
        )

        batch = {
            "history_images": torch.stack(history_tensors, dim=0).unsqueeze(0).to(self.device),
            "current_image": frame_to_keyframe_tensor(current_frame, self.image_size).unsqueeze(0).to(self.device),
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device),
            "motion_feats": torch.tensor(motion_feats, dtype=torch.float32).unsqueeze(0).to(self.device),
            "history_mask": torch.tensor(history_mask, dtype=torch.bool).unsqueeze(0).to(self.device),
        }

        logits = self.model(
            history_images=batch["history_images"],
            current_image=batch["current_image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            motion_feats=batch["motion_feats"],
            history_mask=batch["history_mask"],
        )[0]

        mask = batch["history_mask"][0]
        logits = logits.masked_fill(~mask.bool(), -1e4)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        valid_pairs = []
        for local_i, global_i in enumerate(candidate_with_pad):
            if global_i is None:
                continue
            valid_pairs.append((local_i, global_i, float(probs[local_i])))

        if len(valid_pairs) == 0:
            return None

        local_i, global_i, best_prob = max(valid_pairs, key=lambda x: x[2])

        if LEARNED_KEYFRAME_DEBUG_SCORE:
            top_pairs = sorted(valid_pairs, key=lambda x: x[2], reverse=True)[:3]
            print("[LearnedKeyframe] top candidates:", top_pairs)

        return history_buffer[global_i], local_i, global_i, probs, candidate_with_pad




def get_trend_state_runtime():
    global _TREND_STATE_RUNTIME

    if _TREND_STATE_RUNTIME is None:
        _TREND_STATE_RUNTIME = OpenFlyTrendStateRuntime(
            checkpoint_path=WAYPOINT_FEATURE_DUAL_CHECKPOINT,
            device="cuda:0",
            dtype=torch.bfloat16,
            d_scale=WAYPOINT_D_SCALE,
            z_scale=WAYPOINT_Z_SCALE,
            yaw_turn_threshold=WAYPOINT_YAW_TURN_THRESHOLD,
            z_threshold=WAYPOINT_Z_THRESHOLD,
            use_z_decode=WAYPOINT_USE_Z_DECODE,
            forward_policy=WAYPOINT_FORWARD_POLICY,
            forward_3_threshold=WAYPOINT_FORWARD_3_THRESHOLD,
            forward_6_threshold=WAYPOINT_FORWARD_6_THRESHOLD,
            stop_threshold=TREND_STATE_STOP_THRESHOLD,
            prestop_threshold=TREND_STATE_PRESTOP_THRESHOLD,
            temporal_window=TREND_STATE_TEMPORAL_WINDOW,
        )

    return _TREND_STATE_RUNTIME


def get_waypoint_decoder_runtime():
    global _WAYPOINT_DECODER_RUNTIME

    if ACTION_MODE == "openfly":
        return None

    if _WAYPOINT_DECODER_RUNTIME is None:
        _WAYPOINT_DECODER_RUNTIME = OpenFlyFeatureDecoderRuntime(
            checkpoint_path=WAYPOINT_FEATURE_DUAL_CHECKPOINT,
            device="cuda:0",
            dtype=torch.bfloat16,
            d_scale=WAYPOINT_D_SCALE,
            z_scale=WAYPOINT_Z_SCALE,
            yaw_turn_threshold=WAYPOINT_YAW_TURN_THRESHOLD,
            z_threshold=WAYPOINT_Z_THRESHOLD,
            use_z_decode=WAYPOINT_USE_Z_DECODE,
            forward_policy=WAYPOINT_FORWARD_POLICY,
            forward_3_threshold=WAYPOINT_FORWARD_3_THRESHOLD,
            forward_6_threshold=WAYPOINT_FORWARD_6_THRESHOLD,
            stop_prob_threshold=WAYPOINT_STOP_PROB_THRESHOLD,
            stop_logit_margin=WAYPOINT_STOP_LOGIT_MARGIN,
            trend_stop_threshold=TREND_STOP_THRESHOLD,
            trend_prestop_threshold=TREND_PRESTOP_THRESHOLD,
            trend_use_margin=TREND_STOP_USE_MARGIN,
            trend_margin_threshold=TREND_STOP_MARGIN_THRESHOLD,
            trend_use_distance=TREND_STOP_USE_DISTANCE,
            trend_distance_threshold=TREND_STOP_DISTANCE_THRESHOLD,
        )

    return _WAYPOINT_DECODER_RUNTIME


def maybe_override_action_with_waypoint(
    policy,
    processor,
    openfly_action,
    frame_bundle,
    text,
    step: int = 0,
    max_step: int = 100,
):
    if ACTION_MODE == "openfly":
        return int(openfly_action), None

    runtime = get_waypoint_decoder_runtime()

    result = runtime.predict(
        policy=policy,
        processor=processor,
        frame_bundle=frame_bundle,
        text=text,
        step=step,
        max_step=max_step,
    )

    final_action = select_final_action(
        action_mode=ACTION_MODE,
        openfly_action=int(openfly_action),
        waypoint_result=result,
    )

    if WAYPOINT_DEBUG:
        print(
            "[WaypointAction] "
            f"mode={ACTION_MODE} "
            f"openfly={int(openfly_action)} "
            f"aux={result['aux_action']} "
            f"decoder={result['decoder_action']} "
            f"hybrid_aux={result['hybrid_aux_action']} "
            f"trend={result.get('trend_hybrid_action', 'NA')} "
            f"final={final_action} "
            f"stop_prob={result.get('stop_prob', 'NA')} "
            f"prestop_prob={result.get('prestop_prob', 'NA')} "
            f"wp_raw={result['waypoint_pred_raw']}",
            flush=True,
        )

    return int(final_action), result


def get_learned_keyframe_runtime():
    global _LEARNED_KEYFRAME_RUNTIME

    if _LEARNED_KEYFRAME_RUNTIME is None:
        _LEARNED_KEYFRAME_RUNTIME = LearnedKeyframeRuntime(
            checkpoint_path=LEARNED_KEYFRAME_CHECKPOINT,
            vocab_path=LEARNED_KEYFRAME_VOCAB,
            device="cuda:0",
            image_size=LEARNED_KEYFRAME_IMAGE_SIZE,
            max_history=LEARNED_KEYFRAME_MAX_HISTORY,
        )

    return _LEARNED_KEYFRAME_RUNTIME


def build_learned_keyframe_frame_bundle(history_buffer, pose_history, text):
    """
    Learned keyframe order:
      [learned_keyframe, previous, current]

    Fallback:
      len(history_buffer) < 4 or scorer failure => residual baseline.
    """
    if len(history_buffer) == 0:
        raise ValueError("history_buffer is empty")

    if len(history_buffer) < 4:
        return build_residual_frame_bundle(history_buffer)

    current = history_buffer[-1]
    previous = history_buffer[-2]

    runtime = get_learned_keyframe_runtime()
    result = runtime.score_candidates(
        history_buffer=history_buffer,
        pose_history=pose_history,
        text=text,
        exclude_previous=LEARNED_KEYFRAME_EXCLUDE_PREVIOUS,
    )

    if result is None:
        return build_keyframe_frame_bundle(history_buffer, max_candidates=KEYFRAME_MAX_CANDIDATES)

    keyframe, local_i, global_i, probs, candidate_with_pad = result

    bundle = [keyframe, previous, current]
    validate_frame_bundle(bundle)

    return bundle



def build_learned_topk_fusion_frame_bundle(history_buffer, pose_history, text):
    """
    Final STEER-VLN keyframe mode:
      [fused_topk_keyframe, previous, current]
    """
    if len(history_buffer) == 0:
        raise ValueError("history_buffer is empty")

    if len(history_buffer) < 4:
        return build_residual_frame_bundle(history_buffer)

    current = history_buffer[-1]
    previous = history_buffer[-2]

    runtime = get_learned_keyframe_runtime()
    result = runtime.score_candidates(
        history_buffer=history_buffer,
        pose_history=pose_history,
        text=text,
        exclude_previous=LEARNED_KEYFRAME_EXCLUDE_PREVIOUS,
    )

    if result is None:
        return build_keyframe_frame_bundle(history_buffer, max_candidates=KEYFRAME_MAX_CANDIDATES)

    _, _, _, probs, candidate_with_pad = result

    valid = []
    for local_i, global_i in enumerate(candidate_with_pad):
        if global_i is None:
            continue
        valid.append((int(global_i), float(probs[local_i])))

    if len(valid) == 0:
        return build_keyframe_frame_bundle(history_buffer, max_candidates=KEYFRAME_MAX_CANDIDATES)

    valid = sorted(valid, key=lambda x: x[1], reverse=True)[:max(1, LEARNED_KEYFRAME_TOPK)]
    weight_sum = sum(max(v, 1e-8) for _, v in valid)
    weights = [(idx, max(score, 1e-8) / weight_sum) for idx, score in valid]

    fused = None
    for global_i, w in weights:
        arr = history_buffer[global_i]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr[:, :, :3].astype(np.float32)
        fused = arr * float(w) if fused is None else fused + arr * float(w)

    fused = np.clip(fused, 0, 255).astype(np.uint8)

    bundle = [fused, previous, current]
    validate_frame_bundle(bundle)
    return bundle



def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break

    if not matched:
        # Never silently map an unmatched continuous vector to STOP.
        # Use nearest legal OpenFly action template and print diagnostics.
        try:
            action_f = action.astype(np.float32)
            dists = [float(np.linalg.norm(action_f - v.astype(np.float32))) for v in action_values]
            result = int(np.argmin(dists))
            print(
                f"[ActionDecode][WARN] unmatched action vector: {action}; "
                f"nearest_action={result}; dist={dists[result]:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[ActionDecode][WARN] unmatched action vector and nearest decode failed: {e}", flush=True)
            result = 1  # safer than STOP: small forward action
    return result


def get_action(policy, processor, frame_bundle, text, device="cuda:0", dtype=torch.bfloat16):
    validate_frame_bundle(frame_bundle)

    images = prepare_images_for_processor(frame_bundle)
    inputs = processor(text, images).to(device, dtype=dtype)
    action = policy.predict_action(**inputs, unnorm_key=os.environ.get("OPENFLY_UNNORM_KEY", "vlnv1"), do_sample=False)

    print("raw action:", action)
    action = action.round().astype(int)

    action_id = convert_to_action_id(action)
    print("Action:", action_id)
    return action_id


def calculate_distance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) ** 2 +
        (point2[1] - point1[1]) ** 2 +
        (point2[2] - point1[2]) ** 2
    )


def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose
    step_size = 3.0

    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) * 2
        y += step_size * math.sin(yaw) * 2
    elif action == 9:
        x += step_size * math.cos(yaw) * 3
        y += step_size * math.sin(yaw) * 3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return [x, y, z, yaw]



def parse_eval_cli_args(default_baseline_method=None):
    """Parse optional eval CLI arguments while keeping old env-only scripts compatible."""
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "eval_json_positional",
        nargs="?",
        default=None,
        help="Optional eval annotation json path, e.g. dataset/Annotation/seen.json.",
    )
    parser.add_argument(
        "--eval_json", "--eval-json", "--eval_config", "--eval-config", "-c",
        dest="eval_json",
        default=None,
        help="Evaluation annotation json path. Supports seen.json / unseen.json.",
    )
    parser.add_argument(
        "--seen_json", "--seen-json",
        dest="seen_json",
        default=None,
        help="Shortcut for --eval_json PATH --eval_split test_seen.",
    )
    parser.add_argument(
        "--unseen_json", "--unseen-json",
        dest="unseen_json",
        default=None,
        help="Shortcut for --eval_json PATH --eval_split test_unseen.",
    )
    parser.add_argument(
        "--eval_split", "--eval-split", "--split",
        dest="eval_split",
        default=None,
        help="demo, seen/test_seen, or unseen/test_unseen.",
    )
    parser.add_argument(
        "--metrics_tag", "--metrics-tag",
        dest="metrics_tag",
        default=None,
        help="Tag used for output csv/json metric files.",
    )
    parser.add_argument(
        "--metrics_root", "--metrics-root",
        dest="metrics_root",
        default=None,
        help="Directory for metric csv/json outputs.",
    )
    parser.add_argument(
        "--max_step", "--max-step",
        dest="max_step",
        type=int,
        default=None,
        help="Maximum rollout steps per sample. Defaults to MAX_STEP env or 100.",
    )
    parser.add_argument(
        "--baseline_method", "--baseline-method",
        dest="baseline_method",
        default=default_baseline_method,
        help="Baseline/model tag used in image and log outputs.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Eval CLI] ignored unknown args: {unknown}")

    if args.seen_json and args.unseen_json:
        raise ValueError("Use only one of --seen_json or --unseen_json.")
    if args.seen_json:
        args.eval_json = args.seen_json
        args.eval_split = args.eval_split or "test_seen"
    if args.unseen_json:
        args.eval_json = args.unseen_json
        args.eval_split = args.eval_split or "test_unseen"
    if args.eval_json is None and args.eval_json_positional:
        args.eval_json = args.eval_json_positional

    # Infer split name from common file names when the user only passes seen.json/unseen.json.
    if args.eval_split is None and args.eval_json:
        name = Path(args.eval_json).name.lower().replace("-", "_")
        if name in {"seen.json", "test_seen.json", "eval_seen.json", "eval_test_seen.json"}:
            args.eval_split = "test_seen"
        elif name in {"unseen.json", "test_unseen.json", "eval_unseen.json", "eval_test_unseen.json"}:
            args.eval_split = "test_unseen"
        else:
            args.eval_split = "custom"

    if args.eval_split:
        os.environ["OPENFLY_EVAL_SPLIT"] = args.eval_split
        os.environ["STEER_VLN_EVAL_SPLIT"] = args.eval_split
    if args.metrics_tag:
        os.environ["EVAL_METRICS_TAG"] = args.metrics_tag
    if args.metrics_root:
        os.environ["EVAL_METRICS_ROOT"] = args.metrics_root
    if args.baseline_method:
        os.environ["BASELINE_METHOD"] = args.baseline_method

    return args


def main():
    args = parse_eval_cli_args(default_baseline_method="steer_vln")
    eval_split = args.eval_split or get_eval_split_name()
    eval_config_arg = args.eval_json or os.environ.get("EVAL_CONFIG")
    if args.eval_json:
        os.environ["EVAL_CONFIG"] = args.eval_json
    all_eval_info, eval_info, eval_split = load_eval_info(split=eval_split, explicit_config=eval_config_arg)
    print(f"[STEER-VLN Eval] split={eval_split}")
    print(f"[STEER-VLN Eval] eval_config={eval_info}")
    print(f"[STEER-VLN Eval] num_samples={len(all_eval_info)}")

    # Load base OpenFly-Agent: local first, then remote fallback if allowed.
    processor, policy = load_openfly_model_local_first(
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    # ===== Load LoRA adapter for eval =====
    lora_adapter_path = os.environ.get(
        "LORA_ADAPTER_PATH",
        "",
    ).strip()
    lora_target_path = os.environ.get("LORA_TARGET_PATH", "").strip()

    if lora_adapter_path:
        policy = load_lora_adapter_for_eval(
            model=policy,
            adapter_dir=lora_adapter_path,
            lora_target_path=lora_target_path if lora_target_path else None,
        )
        policy.eval()
        print(f"[LoRA Eval] loaded adapter: {lora_adapter_path}", flush=True)
    # ===== Load LoRA adapter for eval end =====

    if KEYFRAME_MODE in ("learned", "learned_topk", "topk_fusion", "learned_topk_fusion"):
        get_learned_keyframe_runtime()

    # Final STEER-VLN safety guard:
    # trend head is auxiliary only; decoder override is not allowed.
    if TREND_CONDITIONED and ACTION_MODE != "openfly":
        raise RuntimeError(
            f"Invalid final STEER_VLN config: TREND_CONDITIONED=1 requires ACTION_MODE=openfly, "
            f"but got ACTION_MODE={ACTION_MODE}. Decoder override is not allowed."
        )

    print(f"[Eval] KEYFRAME_MODE={KEYFRAME_MODE}")
    print(f"[Eval] LEARNED_KEYFRAME_EXCLUDE_PREVIOUS={LEARNED_KEYFRAME_EXCLUDE_PREVIOUS}")
    print(f"[Eval] LEARNED_KEYFRAME_MAX_HISTORY={LEARNED_KEYFRAME_MAX_HISTORY}")

    acc = 0
    data_num = 0
    # Episode-level paper metrics for final and scene-wise reporting.
    nav_episode_metrics = []
    MAX_STEP = int(args.max_step if args.max_step is not None else os.environ.get("MAX_STEP", "100"))
    baseline_method = os.environ.get("BASELINE_METHOD", "steer_vln")

    # Stop-related metrics.
    # Keep navigation success definition unchanged.
    pred_stop_count = 0
    successful_stop_count = 0
    forced_end_count = 0
    image_error_count = 0
    valid_data_num = 0

    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)

    for env_name, eval_group in env_groups.items():
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_group)} data entries")
        time.sleep(5)

        if "airsim" in env_name:
            env_bridge = AirsimBridge(env_name)
            pos_ratio = 1.0
        elif "ue" in env_name:
            env_bridge = UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name)
            pos_ratio = 1.0
        elif "gs" in env_name:
            env_bridge = GSBridge(env_name)
            pos_ratio = get_gs_pos_ratio(env_name)
        else:
            print(f"Unknown environment type: {env_name}, skipping")
            continue

        for idx, item in enumerate(eval_group):
            acts = []
            data_num += 1
            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")

            image_root = Path(os.environ.get("EVAL_IMAGE_ROOT", "images"))
            image_dir = get_eval_image_dir(image_root, baseline_method, env_name, idx)
            image_paths = []
            action_sequence = []

            image_error = False
            sample_pred_stop = False

            if TREND_CONDITIONED:
                trend_runtime_for_reset = get_trend_state_runtime()
                if hasattr(trend_runtime_for_reset, "reset_episode"):
                    trend_runtime_for_reset.reset_episode()

            pitch = -45.0 if 'high' in item['image_path'] else 0.0

            if "gs" in env_name:
                env_bridge.set_camera_pose(
                    start_postion[0] / pos_ratio,
                    start_postion[1] / pos_ratio,
                    start_postion[2] / pos_ratio,
                    pitch,
                    np.rad2deg(start_yaw),
                    0,
                    str(Path(image_dir).resolve())
                )
            else:
                env_bridge.set_camera_pose(
                    start_postion[0] / pos_ratio,
                    start_postion[1] / pos_ratio,
                    start_postion[2] / pos_ratio,
                    pitch,
                    np.rad2deg(start_yaw),
                    0
                )

            step = 0
            flag_osr = 0
            image_list = []
            pose_history = []
            env_bridge.pass_len = 1e-3
            old_pose = new_pose

            while step < MAX_STEP:
                try:
                    raw_image = env_bridge.get_camera_data()
                    image = raw_image
                    frame_path = image_dir / f"{step:04d}.png"
                    cv2.imwrite(str(frame_path), image)
                    image_paths.append(str(frame_path.relative_to(image_root)))
                    image_list.append(image)
                    pose_history.append(list(new_pose))

                    if KEYFRAME_MODE == "learned":
                        frame_bundle = build_learned_keyframe_frame_bundle(
                            image_list,
                            pose_history,
                            text=text,
                        )
                    elif KEYFRAME_MODE in ("learned_topk", "topk_fusion", "learned_topk_fusion"):
                        frame_bundle = build_learned_topk_fusion_frame_bundle(
                            image_list,
                            pose_history,
                            text=text,
                        )
                    elif KEYFRAME_MODE == "heuristic":
                        frame_bundle = build_keyframe_frame_bundle(
                            image_list,
                            max_candidates=KEYFRAME_MAX_CANDIDATES
                        )
                    elif KEYFRAME_MODE in ("current_only", "no_residual", "noresidual"):
                        frame_bundle = build_current_only_frame_bundle(image_list)
                    else:
                        frame_bundle = build_residual_frame_bundle(image_list)

                    if step < DEBUG_SAVE_STEPS and SAVE_FRAME_DEBUG:
                        save_frame_bundle_debug(
                            frame_bundle=frame_bundle,
                            step=step,
                            sample_idx=idx,
                            env_name=env_name
                        )

                    if step < DEBUG_SAVE_STEPS and SAVE_KEYFRAME_DEBUG and KEYFRAME_MODE in ("heuristic", "learned"):
                        save_keyframe_debug(
                            first=frame_bundle[0],
                            second=frame_bundle[1],
                            third=frame_bundle[2],
                            step=step,
                            sample_idx=idx,
                            env_name=env_name
                        )

                    if TREND_CONDITIONED:
                        trend_runtime = get_trend_state_runtime()
                        trend_state = trend_runtime.predict_state(
                            policy=policy,
                            processor=processor,
                            frame_bundle=frame_bundle,
                            text=text,
                            step=step,
                            max_step=MAX_STEP,
                        )

                        # Final action is generated by OpenFly/LoRA.
                        # Trend head only augments the instruction.
                        model_action = get_action(
                            policy=policy,
                            processor=processor,
                            frame_bundle=frame_bundle,
                            text=trend_state["conditioned_text"],
                            device="cuda:0",
                            dtype=torch.bfloat16
                        )
                        openfly_action = model_action

                        if TREND_STATE_DEBUG:
                            print(
                                "[TrendState] "
                                f"motion_hint={trend_state['aux_action_name']} "
                                f"wp_hint={trend_state['decoder_action_name']} "
                                f"distance={trend_state['distance_trend']} "
                                f"yaw={trend_state['yaw_trend']} "
                                f"vertical={trend_state['vertical_trend']} "
                                f"single_stop={trend_state['single_stop_prob']:.3f} "
                                f"temporal_stop={trend_state['stop_prob']:.3f} "
                                f"prestop={trend_state['prestop_prob']:.3f} "
                                f"temporal={trend_state['temporal_stop_trend']} "
                                f"hist={trend_state['stop_history_len']}",
                                flush=True,
                            )
                    else:
                        openfly_action = get_action(
                            policy=policy,
                            processor=processor,
                            frame_bundle=frame_bundle,
                            text=text,
                            device="cuda:0",
                            dtype=torch.bfloat16
                        )

                        model_action, waypoint_result = maybe_override_action_with_waypoint(
                            policy=policy,
                            processor=processor,
                            openfly_action=openfly_action,
                            frame_bundle=frame_bundle,
                            text=text,
                            step=step,
                            max_step=MAX_STEP,
                        )

                    acts.append(model_action)
                    action_sequence.append(int(model_action))

                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(
                        f"Environment: {env_name}, Sample: {idx}, Step: {step}, "
                        f"TrendConditioned: {TREND_CONDITIONED}, Action: {model_action}, New position: {new_pose}"
                    )

                    if "gs" in env_name:
                        env_bridge.set_camera_pose(
                            new_pose[0] / pos_ratio,
                            new_pose[1] / pos_ratio,
                            new_pose[2] / pos_ratio,
                            pitch,
                            np.rad2deg(new_pose[3]),
                            0,
                            str(Path(image_dir).resolve())
                        )
                    else:
                        env_bridge.set_camera_pose(
                            new_pose[0] / pos_ratio,
                            new_pose[1] / pos_ratio,
                            new_pose[2] / pos_ratio,
                            pitch,
                            np.rad2deg(new_pose[3]),
                            0
                        )

                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)

                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)

                    old_pose = new_pose

                    if model_action == 0:
                        sample_pred_stop = True
                        pred_stop_count += 1
                        break

                    step += 1

                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_postion)
            env_bridge.distance_to_goal.append(dis)

            sample_success = 1 if dis < 20 else 0
            sample_spl_raw, sample_spl_bounded = openfly_compute_spl_values(
                sample_success,
                env_bridge.traj_len,
                env_bridge.pass_len,
            )

            if sample_success:
                env_bridge.success.append(1)
                # Keep env_bridge.spl as paper-standard bounded SPL.
                env_bridge.spl.append(sample_spl_bounded)
                if not hasattr(env_bridge, "spl_raw"):
                    env_bridge.spl_raw = []
                if not hasattr(env_bridge, "spl_bounded"):
                    env_bridge.spl_bounded = []
                env_bridge.spl_raw.append(sample_spl_raw)
                env_bridge.spl_bounded.append(sample_spl_bounded)
                acc += 1
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0.0)
                if not hasattr(env_bridge, "spl_raw"):
                    env_bridge.spl_raw = []
                if not hasattr(env_bridge, "spl_bounded"):
                    env_bridge.spl_bounded = []
                env_bridge.spl_raw.append(0.0)
                env_bridge.spl_bounded.append(0.0)

            if flag_osr == 0:
                env_bridge.osr.append(0)

            sample_scene_category = openfly_scene_category(item, env_name)
            sample_successful_stop = bool(sample_pred_stop and dis < 20)
            sample_forced_end = bool((not sample_pred_stop) and step >= MAX_STEP)
            openfly_append_scene_metric(
                nav_episode_metrics,
                env_name=env_name,
                scene_category=sample_scene_category,
                sample_idx=idx,
                ne_m=float(env_bridge.distance_to_goal[-1]),
                sr=int(env_bridge.success[-1]),
                osr=int(env_bridge.osr[-1]),
                spl_raw=float(env_bridge.spl_raw[-1]),
                spl_bounded=float(env_bridge.spl_bounded[-1]),
                pred_stop=sample_pred_stop,
                successful_stop=sample_successful_stop,
                forced_end=sample_forced_end,
                image_error=image_error,
            )

            env_bridge.print_info()
            print(
                f"[EpisodeMetric] env={env_name} scene={sample_scene_category} sample={idx} "
                f"NE/m={float(env_bridge.distance_to_goal[-1]):.4f} "
                f"SR={int(env_bridge.success[-1])} OSR={int(env_bridge.osr[-1])} "
                f"SPL={float(env_bridge.spl_bounded[-1]):.6f} "
                f"SPL_raw={float(env_bridge.spl_raw[-1]):.6f} "
                f"SPL_bounded={float(env_bridge.spl_bounded[-1]):.6f}"
            )

            save_eval_task_summary(
                image_dir=image_dir,
                instruction=text,
                env_name=env_name,
                sample_idx=idx,
                baseline_method=baseline_method,
                start_position=start_postion,
                goal_position=end_position,
                start_yaw=start_yaw,
                final_position=new_pose,
                success=env_bridge.success[-1],
                osr=env_bridge.osr[-1],
                num_steps=len(image_paths),
                action_sequence=action_sequence,
                image_files=image_paths,
                stop_pred=sample_pred_stop,
                image_error=image_error,
                additional_info={
                    "scene_category": sample_scene_category,
                    "NE/m": float(env_bridge.distance_to_goal[-1]),
                    "SR": int(env_bridge.success[-1]),
                    "OSR": int(env_bridge.osr[-1]),
                    "SPL": float(env_bridge.spl_bounded[-1]),
                    "SPL_raw": float(env_bridge.spl_raw[-1]),
                    "SPL_bounded": float(env_bridge.spl_bounded[-1]),
                    "SR/%": 100.0 * float(env_bridge.success[-1]),
                    "OSR/%": 100.0 * float(env_bridge.osr[-1]),
                    "SPL/%": 100.0 * float(env_bridge.spl_bounded[-1]),
                    "SPL_raw/%": 100.0 * float(env_bridge.spl_raw[-1]),
                    "SPL_bounded/%": 100.0 * float(env_bridge.spl_bounded[-1]),
                },
            )

            # Stop metrics use valid samples only.
            # Navigation accuracy above is kept source-compatible.
            if image_error:
                image_error_count += 1
                continue

            valid_data_num += 1

            if sample_successful_stop:
                successful_stop_count += 1

            if sample_forced_end:
                forced_end_count += 1

        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        del env_bridge
        import gc
        gc.collect()

    # Final results.
    # Keep original navigation accuracy denominator unchanged for baseline comparability.
    final_acc = acc / data_num if data_num > 0 else 0

    stop_denom = valid_data_num if valid_data_num > 0 else data_num
    pred_stop_rate = pred_stop_count / stop_denom if stop_denom > 0 else 0
    successful_stop_rate = successful_stop_count / stop_denom if stop_denom > 0 else 0
    forced_end_rate = forced_end_count / stop_denom if stop_denom > 0 else 0
    image_error_rate = image_error_count / data_num if data_num > 0 else 0

    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Valid samples for stop metrics: {valid_data_num}")
    print(f"Image error count: {image_error_count}")
    print(f"Image error rate: {image_error_rate:.4f}")
    print(f"Final accuracy: {final_acc:.4f}")

    metrics_tag = os.environ.get(
        "EVAL_METRICS_TAG",
        f"{Path(__file__).stem}_{baseline_method}" if "baseline_method" in globals() or "baseline_method" in locals() else Path(__file__).stem,
    )
    openfly_print_and_save_metric_summary(nav_episode_metrics, metrics_tag=metrics_tag)

    print(f"Pred stop count: {pred_stop_count}")
    print(f"Pred stop rate: {pred_stop_rate:.4f}")
    print(f"Successful stop count: {successful_stop_count}")
    print(f"Successful stop rate: {successful_stop_rate:.4f}")
    print(f"Forced end count: {forced_end_count}")
    print(f"Forced end rate: {forced_end_rate:.4f}")


if __name__ == '__main__':
    main()

