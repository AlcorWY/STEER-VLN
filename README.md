# STEER-VLN（We are modifying the code.）

STEER-VLN is an experimental extension of the OpenFly Platform for embodied low-altitude UAV Vision-Language Navigation (VLN). It keeps the original OpenFly simulation, toolchain, training, and evaluation interfaces, and adds a temporal state-conditioned cybernetic adaptation pipeline for low-altitude UAV VLN experiments.

This GitHub-ready version places all STEER-VLN-specific source code directly under the repository root directory `STEER-VLN/`. The new STEER-VLN training, evaluation, ablation, runtime, and utility scripts should be launched from `STEER-VLN/`.

![cover](images/cover.png)

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Repository Layout](#repository-layout)
4. [Data and Model Preparation](#data-and-model-preparation)
5. [Simulation Preparation](#simulation-preparation)
6. [STEER-VLN Training](#steer-vln-training)
7. [STEER-VLN Evaluation](#steer-vln-evaluation)
8. [Offline Module Evaluation](#offline-module-evaluation)
9. [Troubleshooting](#troubleshooting)
10. [Acknowledgement](#acknowledgement)
11. [License](#license)

## Prerequisites

- Operating system: Ubuntu 22.04 is recommended.
- GPU: NVIDIA GPU with CUDA support. A 24 GB GPU such as RTX 3090 is recommended for the full STEER-VLN training pipeline.
- Python: 3.10 is recommended.
- ROS2: Humble, following the OpenFly environment setup.
- Conda environment: `openfly` is recommended for consistency with OpenFly.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <your-steer-vln-repository-url>.git
cd STEER-VLN
```

Create the Python environment:

```bash
conda create -n STEER-VLN python=3.10 -y
conda activate STEER-VLN
pip install -r requirements.txt
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

Install the OpenFly auxiliary dependency:

```bash
git clone https://github.com/kvablack/dlimp
cd dlimp
pip install -e .
cd ..
```

Install system dependencies used by the simulator and toolchain:

```bash
sudo apt update
sudo apt install -y xvfb libgoogle-glog-dev ros-humble-pcl-ros nlohmann-json3-dev
```

Build the ROS workspace if the toolchain is needed:

```bash
cd tool_ws
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
cd ..
```

## Repository Layout

```text
STEER-VLN/
├── STEER-VLN/                   # STEER-VLN training, evaluation, ablation, and utility scripts
│   ├── keyframe/                # learned keyframe cache/scorer components
│   ├── runtime/                 # runtime helper package
│   ├── utils/                   # training/evaluation helpers
│   ├── train_integrated_full.py # integrated STEER-VLN training entry
│   ├── eval_hd_lora.py          # simulator evaluation entry
│   └── run_steer_vln_*.sh       # one-command training/evaluation scripts
├── train/                       # OpenFly baseline training/evaluation scripts
├── scripts/                     # OpenFly simulation/toolchain scripts
├── configs/                     # scene and evaluation configs
├── envs/                        # simulator environment directory
├── scene_data/                  # scene point clouds and segmentation maps
└── requirements.txt
```

## Data and Model Preparation

STEER-VLN follows the OpenFly data layout. Prepare the dataset, annotations, simulator assets, and local OpenFly-Agent model before running experiments.

Recommended local paths:

```text
dataset/Annotation/train_airsim16.json
dataset/Annotation/test_seen.json
dataset/Annotation/test_unseen.json
dataset/hf_openfly_airsim16/traj/
models/openfly-agent-7b/
```

If your paths are different, pass them through script arguments or environment variables. The STEER-VLN scripts prefer local files and can be used in offline mode after the model and dataset are prepared.

Useful environment variables:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENFLY_MODEL_DIR=models/openfly-agent-7b
export STEER_VLN_FINAL_MODEL_DIR=runs/STEER-VLN/train/final_model
export STEER_VLN_TEST_SEEN_CONFIG=dataset/Annotation/test_seen.json
export STEER_VLN_TEST_UNSEEN_CONFIG=dataset/Annotation/test_unseen.json
```

## Simulation Preparation

Start the OpenFly environment bridge before simulator-based evaluation:

```bash
conda activate openfly
python scripts/sim/env_bridge.py --env env_airsim_16
```

Wait until the simulator reports that it is ready. AirSim/UE startup may need additional time after scene switching. The STEER-VLN evaluation scripts include waiting and retry logic for more stable AirSim connections.

To clean stale AirSim, UE, PX4, and STEER-VLN processes:

```bash
bash STEER-VLN/clean_steer_vln_airsim_ue.sh
```

## STEER-VLN Training

Run the integrated STEER-VLN training pipeline:

```bash
conda activate openfly
bash STEER-VLN/run_steer_vln_final_train.sh
```

The default output directory is:

```text
runs/STEER-VLN/train/final_model/
```

Typical final-model files include:

```text
keyframe_scorer_best.pt
simple_tokenizer_vocab.json
m3c_trend_head_best.pt
lora_adapter/
```

For custom training data or output paths, set environment variables before launching the script, for example:

```bash
export STEER_VLN_FINAL_MODEL_DIR=runs/STEER-VLN/train/final_model
export OPENFLY_MODEL_DIR=models/openfly-agent-7b
bash STEER-VLN/run_steer_vln_final_train.sh
```

## STEER-VLN Evaluation

Run all final seen/unseen simulator evaluations:

```bash
conda activate openfly
bash STEER-VLN/run_steer_vln_final_eval_seen_unseen_all_ordered_no_gs_no_gtav.sh
```

Run a single ablation setting:

```bash
# B0: OpenFly-style baseline comparison
bash STEER-VLN/run_steer_vln_final_eval_b0_only.sh

# E1/E2/E3: STEER-VLN ablation variants
bash STEER-VLN/run_steer_vln_final_eval_e1_only.sh
bash STEER-VLN/run_steer_vln_final_eval_e2_only.sh
bash STEER-VLN/run_steer_vln_final_eval_e3_only.sh
```

Seen/unseen split scripts are also provided:

```bash
bash STEER-VLN/run_steer_vln_final_eval_b0_seen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_b0_unseen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_e3_seen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_e3_unseen_ordered_no_gs_no_gtav.sh
```

Evaluation logs are saved under:

```text
runs/STEER-VLN/logs/eval_final_ablation/
```

Summarize final ablation logs:

```bash
python STEER-VLN/summarize_steer_vln_final_ablation_logs.py runs/STEER-VLN/logs/eval_final_ablation
```

## Offline Module Evaluation

Before running the simulator, you can check the offline components:

```bash
bash STEER-VLN/run_steer_vln_offline_module_eval.sh
```

You can also run individual checks:

```bash
python STEER-VLN/check_steer_vln_scheme.py
python STEER-VLN/offline_eval_keyframe_module.py
python STEER-VLN/offline_eval_policy_action_module.py
python STEER-VLN/offline_eval_integrated_modules.py
```

Offline reports are written to:

```text
runs/STEER-VLN/offline_module_eval/
```

## Troubleshooting

If AirSim or UE fails to connect, clean stale simulator processes and restart the environment bridge:

```bash
bash STEER-VLN/clean_steer_vln_airsim_ue.sh
python scripts/sim/env_bridge.py --env env_airsim_16
```

If final checkpoints are missing, run training first or set the checkpoint directory explicitly:

```bash
export STEER_VLN_FINAL_MODEL_DIR=/path/to/final_model
```

If Hugging Face downloads are blocked or unavailable, prepare the OpenFly-Agent model locally and enable offline mode:

```bash
export OPENFLY_MODEL_DIR=models/openfly-agent-7b
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Acknowledgement

This repository is built on top of the OpenFly Platform. Please also follow the original OpenFly setup instructions when preparing simulator assets, scene data, and benchmark annotations.

## License

This project follows the license terms included in this repository and the upstream OpenFly Platform components.
