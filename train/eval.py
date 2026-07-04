
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
    """
    try:
        image_path = str(item.get("image_path", ""))
    except Exception:
        image_path = ""

    parts = image_path.replace("\\", "/").split("/")
    if "astar_data" in parts:
        i = parts.index("astar_data")
        if i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]

    # fallback: use parent folder before file name
    if len(parts) >= 2 and parts[-2]:
        return parts[-2]

    return str(env_name)


def openfly_compute_spl_values(success, traj_len, pass_len):
    """
    SPL_raw follows OpenFly source: success * traj_len / pass_len.
    SPL_bounded follows standard reporting: success * traj_len / max(traj_len, pass_len).
    SPL/% in final summary uses SPL_bounded/%.
    """
    ref_len = max(float(traj_len), 1e-6)
    pred_len = max(float(pass_len), 1e-6)
    if int(success) == 1:
        spl_raw = ref_len / pred_len
        spl_bounded = ref_len / max(ref_len, pred_len)
        spl_bounded = max(0.0, min(1.0, float(spl_bounded)))
    else:
        spl_raw = 0.0
        spl_bounded = 0.0
    return float(spl_raw), float(spl_bounded)


def openfly_append_scene_metric(nav_episode_metrics, env_name, scene_category, sample_idx, ne_m, sr, osr, spl_raw, spl_bounded):
    nav_episode_metrics.append({
        "env": str(env_name),
        "scene": str(scene_category),
        "sample_idx": int(sample_idx),
        "NE/m": float(ne_m),
        "SR": int(sr),
        "OSR": int(osr),
        "SPL_raw": float(spl_raw),
        "SPL_bounded": float(spl_bounded),
        "SPL": float(spl_bounded),
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
        return {
            "samples": len(rows),
            "NE/m": _mean([r["NE/m"] for r in rows]),
            "SR/%": 100.0 * _mean([r["SR"] for r in rows]),
            "OSR/%": 100.0 * _mean([r["OSR"] for r in rows]),
            "SPL/%": 100.0 * _mean([r["SPL_bounded"] for r in rows]),
            "SPL_raw/%": 100.0 * _mean([r["SPL_raw"] for r in rows]),
            "SPL_bounded/%": 100.0 * _mean([r["SPL_bounded"] for r in rows]),
        }

    total = _summ(nav_episode_metrics)

    print(f"NE/m: {total['NE/m']:.4f}")
    print(f"SR/%: {total['SR/%']:.2f}")
    print(f"OSR/%: {total['OSR/%']:.2f}")
    print(f"SPL/%: {total['SPL/%']:.2f}")
    print(f"SPL_raw/%: {total['SPL_raw/%']:.2f}")
    print(f"SPL_bounded/%: {total['SPL_bounded/%']:.2f}")

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
            f"NE/m={info['NE/m']:.4f} SR/%={info['SR/%']:.2f} OSR/%={info['OSR/%']:.2f} "
            f"SPL/%={info['SPL/%']:.2f} SPL_raw/%={info['SPL_raw/%']:.2f} "
            f"SPL_bounded/%={info['SPL_bounded/%']:.2f}"
        )

    print("\nEnv-scene-wise metrics:")
    env_scene_rows = []
    for env, scene in sorted(by_env_scene.keys()):
        info = _summ(by_env_scene[(env, scene)])
        row = {"env": env, "scene": scene, **info}
        env_scene_rows.append(row)
        print(
            f"[EnvSceneMetrics] env={env} scene={scene} samples={info['samples']} "
            f"NE/m={info['NE/m']:.4f} SR/%={info['SR/%']:.2f} OSR/%={info['OSR/%']:.2f} "
            f"SPL/%={info['SPL/%']:.2f} SPL_raw/%={info['SPL_raw/%']:.2f} "
            f"SPL_bounded/%={info['SPL_bounded/%']:.2f}"
        )

    metrics_root = _Path(_os.environ.get("EVAL_METRICS_ROOT", "runs/eval_metrics"))
    metrics_root.mkdir(parents=True, exist_ok=True)
    safe_tag = str(metrics_tag).replace("/", "_").replace("\\", "_").replace(" ", "_")

    episode_csv = metrics_root / f"{safe_tag}_episode_metrics.csv"
    scene_csv = metrics_root / f"{safe_tag}_scene_metrics.csv"
    env_scene_csv = metrics_root / f"{safe_tag}_env_scene_metrics.csv"
    summary_json = metrics_root / f"{safe_tag}_metric_summary.json"

    episode_fields = ["env", "scene", "sample_idx", "NE/m", "SR", "OSR", "SPL/%", "SPL_raw/%", "SPL_bounded/%", "SPL", "SPL_raw", "SPL_bounded"]
    with open(episode_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=episode_fields)
        w.writeheader()
        for r in nav_episode_metrics:
            rr = dict(r)
            rr["SPL/%"] = 100.0 * rr["SPL_bounded"]
            rr["SPL_raw/%"] = 100.0 * rr["SPL_raw"]
            rr["SPL_bounded/%"] = 100.0 * rr["SPL_bounded"]
            w.writerow({k: rr.get(k, "") for k in episode_fields})

    scene_fields = ["scene", "samples", "NE/m", "SR/%", "OSR/%", "SPL/%", "SPL_raw/%", "SPL_bounded/%"]
    with open(scene_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=scene_fields)
        w.writeheader()
        for r in scene_rows:
            w.writerow({k: r.get(k, "") for k in scene_fields})

    env_scene_fields = ["env", "scene", "samples", "NE/m", "SR/%", "OSR/%", "SPL/%", "SPL_raw/%", "SPL_bounded/%"]
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
        time.sleep(10)

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
        time.sleep(1)
        
    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(1)

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
        time.sleep(20)

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
        # dataset_dir = "envs/gs/" + self.env_name  
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"  
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
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

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
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
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass



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

def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):

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

def main():
    eval_split = get_eval_split_name()
    eval_config_env = os.environ.get("EVAL_CONFIG")
    all_eval_info, eval_info, eval_split = load_eval_info(split=eval_split, explicit_config=eval_config_env)
    print(f"[OpenFly Eval] split={eval_split}")
    print(f"[OpenFly Eval] eval_config={eval_info}")
    
    # Load OpenFly model, aligned with eval_baseline_stop_metrics.py.
    local_model_path = os.environ.get(
        "OPENFLY_LOCAL_MODEL_PATH",
        "models/openfly-agent-7b",
    )
    remote_model_id = os.environ.get(
        "OPENFLY_REMOTE_MODEL_ID",
        "IPEC-COMMUNITY/openfly-agent-7b",
    )
    allow_remote = os.environ.get("HF_FALLBACK_REMOTE", "0").lower() in {"1", "true", "yes", "y"}
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
            "\n[Model ERROR] Cannot find local OpenFly model.\n"
            f"Expected directory: {os.path.abspath(local_model_path)}\n"
            f"Expected config:    {os.path.abspath(local_config)}\n"
            "Set OPENFLY_LOCAL_MODEL_PATH or enable HF_FALLBACK_REMOTE=1 when online."
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
    stop = 0
    data_num = 0
    MAX_STEP = int(os.environ.get("MAX_STEP", "100"))
    openfly_metrics = OpenFlyMetricAccumulator()

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
            pos_ratio = 5.15
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
            
            stop_error = 1
            image_error = False
            
            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_postion[0]/pos_ratio, 
                start_postion[1]/pos_ratio, 
                start_postion[2]/pos_ratio, 
                pitch, 
                np.rad2deg(start_yaw), 
                0
            )
            
            step = 0
            flag_osr = 0
            image_list = []
            env_bridge.pass_len = 1e-3
            old_pose = new_pose
            
            while step < MAX_STEP:
                try:
                    raw_image = env_bridge.get_camera_data()
                    cv2.imwrite("test/cur_img.jpg", raw_image)
                    image = raw_image
                    
                    image_list.append(image)
                    model_action = get_action(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    acts.append(model_action)
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0]/pos_ratio, 
                        new_pose[1]/pos_ratio, 
                        new_pose[2]/pos_ratio, 
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
                        stop_error = 0
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
            if flag_osr == 0:
                env_bridge.osr.append(0)

            sample_scene_category = openfly_scene_category(item, env_name)
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

            if image_error:
                continue
                
        
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
    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = 1 - stop / data_num if data_num > 0 else 0
    
    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")

    metrics_tag = os.environ.get(
        "EVAL_METRICS_TAG",
        f"{Path(__file__).stem}_{baseline_method}" if "baseline_method" in globals() or "baseline_method" in locals() else Path(__file__).stem,
    )
    openfly_print_and_save_metric_summary(nav_episode_metrics, metrics_tag=metrics_tag)
    print(f"Final stop rate: {final_stop:.4f}")
    print("\nOpenFly source-compatible metrics:")
    openfly_metrics.print_summary(prefix="OpenFly Metrics")
    print(f"eval_split: {eval_split}")
    print(f"eval_config: {eval_info}")


if __name__ == '__main__':
    main()
