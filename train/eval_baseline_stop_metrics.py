

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
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openfly_eval_utils import (
    OpenFlyMetricAccumulator,
    get_eval_split_name,
    load_eval_info,
    resolve_eval_config,
)


def get_eval_image_dir(root="images", baseline_method="openfly", env_name="unknown", sample_idx: int = 0):
    out_dir = Path(root) / "eval_baseline" / baseline_method / env_name / f"sample_{sample_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_eval_task_summary(
    image_dir,
    instruction,
    env_name,
    sample_idx,
    baseline_method,
    start_position,
    goal_position,
    start_yaw,
    final_position,
    success,
    osr,
    num_steps,
    action_sequence,
    image_files,
    stop_pred,
    image_error,
    additional_info=None,
):
    if isinstance(image_dir, (str, Path)):
        image_dir = Path(image_dir)

    summary = {
        "baseline_method": baseline_method,
        "env_name": env_name,
        "sample_idx": sample_idx,
        "instruction": instruction,
        "start_position": start_position,
        "goal_position": goal_position,
        "start_yaw": start_yaw,
        "final_position": final_position,
        "success": int(success),
        "osr": int(osr),
        "num_steps": num_steps,
        "action_sequence": action_sequence,
        "image_files": image_files,
        "predicted_stop": bool(stop_pred),
        "image_error": bool(image_error),
        "additional_info": additional_info or {},
    }

    summary_path = image_dir / "task_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    root_summary = image_dir.parents[2] / "summary.jsonl"
    root_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(root_summary, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])

class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(30)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
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

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)

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
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.5)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type = 'color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
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
    def __init__(self, ue_ip, ue_port, env_name):
        self.kill_failed_process()
        time.sleep(30)

        # port = self.find_available_port()

        port = random.randint(9000, 9100)
        print(f"Available port: {port}")
        self.modify_port_in_ini(port, env_name)
        ue_port = port

        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_ue_sim)
        self._sim_thread.start()
        time.sleep(15)

        self._client = Client((ue_ip, ue_port))
        self._connection_check()

        self._camera_init()

        # self._drone_init()  
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
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()

            if f'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        with open(ini_file, 'r') as file:
            lines = file.readlines()

        with open(ini_file, 'w') as file:
            for line in lines:
                if line.startswith("Port="):
                    file.write(f"Port={port}\n")
                else:
                    file.write(line)

    def kill_failed_process(self):
        result = subprocess.run(['pgrep', '-n', 'CrashReport'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

        result = subprocess.run(['pgrep', '-n', 'CitySample'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

    def _init_ue_sim(self):
        env_dir = "envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/CitySample.sh"]

        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)
        time.sleep(2)

    def __del__(self):
        self._client.disconnect()

    def _connection_check(self):
        '''Check if connected'''
        if self._client.connect():
            print('UnrealCV connected successfully')
        else:
            print('UnrealCV is not connected')
            exit()

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        '''Set camera position'''
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        print('camera_settings', camera_settings)

    def _camera_init(self):
        '''Camera initialization'''
        time.sleep(2)
        self._client.request('vset /cameras/spawn')
        self._client.request('vset /camera/1/size 1920 1080')
        time.sleep(2)
        self.set_camera_pose(150, 400, 15, 0, 0, 0)  # Initial position
        time.sleep(2)

    def get_camera_data(self, camera_type = 'lit'):
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
            return depth_np  # Return depth data

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
        # Official seen/unseen includes multiple GS scenes. Do not hard-code nwpu02.
        _gs_key = self.env_name.upper().replace("ENV_GS_", "").replace("-", "_")
        _gs_root = os.environ.get("OPENFLY_GS_DATA_ROOT") or os.environ.get("GS_DATA_ROOT")
        dataset_dir = (
            os.environ.get(f"OPENFLY_GS_DATASET_DIR_{_gs_key}")
            or os.environ.get(f"GS_DATASET_DIR_{_gs_key}")
            or os.environ.get("OPENFLY_GS_DATASET_DIR")
            or os.environ.get("GS_DATASET_DIR")
            or (str(Path(_gs_root) / self.env_name) if _gs_root else None)
            or (str(Path(_gs_root) / self.env_name.replace("env_gs_", "")) if _gs_root else None)
            or f"/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/{self.env_name.replace('env_gs_', '')}"
        )
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
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
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
            time.sleep(20)

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



def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result

def random_baseline_action():
    return random.randint(0, 9)


def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):
    if policy is None:
        action_id = random_baseline_action()
        print("Random baseline action:", action_id)
        return action_id

    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)
        
    prompt = text
    inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
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
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

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
    args = parse_eval_cli_args(default_baseline_method="openfly")
    eval_split = args.eval_split or get_eval_split_name()
    eval_config_arg = args.eval_json or os.environ.get("EVAL_CONFIG")
    if args.eval_json:
        os.environ["EVAL_CONFIG"] = args.eval_json
    all_eval_info, eval_info, eval_split = load_eval_info(split=eval_split, explicit_config=eval_config_arg)
    print(f"[OpenFly Eval] split={eval_split}")
    print(f"[OpenFly Eval] eval_config={eval_info}")
    print(f"[OpenFly Eval] num_samples={len(all_eval_info)}")
    
    baseline_method_raw = os.environ.get("BASELINE_METHOD", "openfly")
    baseline_method_key = baseline_method_raw.lower().strip()
    if baseline_method_key in {"openfly", "openvla", "random"}:
        baseline_method = baseline_method_key
    elif baseline_method_key.startswith("b0") or "openfly" in baseline_method_key:
        baseline_method = "openfly"
    elif "openvla" in baseline_method_key:
        baseline_method = "openvla"
    elif "random" in baseline_method_key:
        baseline_method = "random"
    else:
        raise ValueError(
            f"Unsupported BASELINE_METHOD={baseline_method_raw}; "
            "supported choices/tags include openfly, openvla, random, B0_original_openfly_baseline"
        )
    baseline_log_tag = baseline_method_raw.replace("/", "_").replace(" ", "_")
    print(f"[Baseline] tag = {baseline_method_raw}")
    print(f"[Baseline] method = {baseline_method}")

    if baseline_method == "random":
        policy = None
        processor = None
    else:
        if baseline_method == "openfly":
            local_model_path = os.environ.get(
                "OPENFLY_LOCAL_MODEL_PATH",
                "models/openfly-agent-7b",
            )
            remote_model_id = os.environ.get(
                "OPENFLY_REMOTE_MODEL_ID",
                "IPEC-COMMUNITY/openfly-agent-7b",
            )
        else:
            local_model_path = os.environ.get(
                "OPENVLA_LOCAL_MODEL_PATH",
                "models/openvla-7b-prismatic",
            )
            remote_model_id = os.environ.get(
                "OPENVLA_REMOTE_MODEL_ID",
                "openvla/openvla-7b",
            )

        allow_remote = os.environ.get("HF_FALLBACK_REMOTE", "0").lower() in {
            "1", "true", "yes", "y"
        }

        offline_env = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        )

        local_config = os.path.join(local_model_path, "config.json")
        if os.path.isdir(local_model_path) and os.path.isfile(local_config):
            model_name_or_path = local_model_path
            local_files_only = True
            print(f"[Model] Using local model: {os.path.abspath(model_name_or_path)}")
        elif allow_remote and not offline_env:
            model_name_or_path = remote_model_id
            local_files_only = False
            print(f"[Model] Local model not found, fallback to remote: {model_name_or_path}")
        else:
            raise FileNotFoundError(
                "\n[Model ERROR] Cannot find local model.\n"
                f"Expected directory: {os.path.abspath(local_model_path)}\n"
                f"Expected config:    {os.path.abspath(local_config)}\n"
                "Fix one of the following:\n"
                "  1) Put the model under the expected local path; or\n"
                "  2) export OPENFLY_LOCAL_MODEL_PATH or OPENVLA_LOCAL_MODEL_PATH with the correct path; or\n"
                "  3) if online is allowed: export HF_FALLBACK_REMOTE=1 and unset offline env vars.\n"
            )

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        policy = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to("cuda:0")

    # Test metrics
    acc = 0
    data_num = 0
    # Episode-level paper metrics for final and scene-wise reporting.
    nav_episode_metrics = []
    MAX_STEP = int(args.max_step if args.max_step is not None else os.environ.get("MAX_STEP", "100"))
    openfly_metrics = OpenFlyMetricAccumulator()

    # Stop-related metrics.
    # Keep navigation success definition unchanged.
    pred_stop_count = 0
    successful_stop_count = 0
    forced_end_count = 0
    image_error_count = 0
    valid_data_num = 0

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)
    
    # Process each environment type sequentially
    for env_name, eval_info in env_groups.items():
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_info)} data entries")
        time.sleep(5)
        
        # Create appropriate environment bridge based on environment type
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
        
        env_openfly_metrics = OpenFlyMetricAccumulator()

        # Evaluate all data for current environment
        for idx, item in enumerate(eval_info):
            acts = []  # Reset action list
            data_num += 1
            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")

            image_root = os.environ.get("EVAL_IMAGE_ROOT", "images")
            image_dir = get_eval_image_dir(image_root, baseline_log_tag, env_name, idx)
            image_paths = []
            action_sequence = []
            
            # stop_error is kept for source compatibility, but final stop metrics
            # are computed by pred_stop_count / successful_stop_count / forced_end_count.
            stop_error = 1
            image_error = False
            sample_pred_stop = False
            
            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            if "gs" in env_name:
                env_bridge.set_camera_pose(
                    start_postion[0]/pos_ratio,
                    start_postion[1]/pos_ratio,
                    start_postion[2]/pos_ratio,
                    pitch,
                    np.rad2deg(start_yaw),
                    0,
                    str(Path(image_dir).resolve()),
                )
            else:
                env_bridge.set_camera_pose(
                    start_postion[0]/pos_ratio,
                    start_postion[1]/pos_ratio,
                    start_postion[2]/pos_ratio,
                    pitch,
                    np.rad2deg(start_yaw),
                    0,
                )
            
            step = 0
            flag_osr = 0
            image_list = []
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
                    model_action = get_action(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    acts.append(model_action)
                    action_sequence.append(int(model_action))
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    if "gs" in env_name:
                        env_bridge.set_camera_pose(
                            new_pose[0]/pos_ratio,
                            new_pose[1]/pos_ratio,
                            new_pose[2]/pos_ratio,
                            pitch,
                            np.rad2deg(new_pose[3]),
                            0,
                            str(Path(image_dir).resolve()),
                        )
                    else:
                        env_bridge.set_camera_pose(
                            new_pose[0]/pos_ratio,
                            new_pose[1]/pos_ratio,
                            new_pose[2]/pos_ratio,
                            pitch,
                            np.rad2deg(new_pose[3]),
                            0,
                        )
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose

                    if model_action == 0:
                        stop_error = 0
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
            openfly_metrics.update_from_bridge_latest(env_bridge)
            env_openfly_metrics.update_from_bridge_latest(env_bridge)

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

            save_eval_task_summary(
                image_dir=image_dir,
                instruction=text,
                env_name=env_name,
                sample_idx=idx,
                baseline_method=baseline_log_tag,
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
                
        env_summary = env_openfly_metrics.summary()
        print(
            f"[OpenFlyMetrics][env={env_name}] "
            f"NE/m={env_summary['NE/m']:.4f}, "
            f"SR/%={env_summary['SR/%']:.2f}, "
            f"OSR/%={env_summary['OSR/%']:.2f}, "
            f"SPL/%={env_summary['SPL/%']:.2f}, "
            f"count={env_summary['count']}"
        )

        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        del env_bridge
        import gc
        gc.collect()
    
    # Final results
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
    print("\nOpenFly source-compatible metrics:")
    openfly_metrics.print_summary(prefix="OpenFly Metrics")
    print(f"eval_split: {eval_split}")
    print(f"eval_config: {eval_info}")


if __name__ == '__main__':
    main()
