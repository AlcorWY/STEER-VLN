#!/usr/bin/env bash
# clean_steer_vln_airsim_ue.sh
# 一键清理 OpenFly STEER-VLN / AirSim / UE 进程，并释放常见端口与残留进程端口

set +e

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "============================================================"
echo "[OpenFly STEER-VLN / AirSim / UE Cleanup]"
echo "============================================================"
echo "[ROOT] $ROOT_DIR"
echo "[TIME] $(date)"
echo ""

cd "$ROOT_DIR" || {
  echo "[ERROR] Cannot cd to $ROOT_DIR"
  exit 1
}

CURRENT_PID=$$
echo "[INFO] Current script PID: $CURRENT_PID"
echo ""

kill_by_pattern_soft() {
  local pattern="$1"
  echo "[SOFT KILL] $pattern"
  pkill -f "$pattern" 2>/dev/null || true
}

kill_by_pattern_force() {
  local pattern="$1"
  echo "[FORCE KILL] $pattern"
  pkill -9 -f "$pattern" 2>/dev/null || true
}

show_processes() {
  echo ""
  echo "[PROCESS CHECK]"
  ps -ef | grep -Ei \
    "openfly|OpenFly-Platform|STEER-VLN|run_steer_vln|eval_hd_lora|train_integrated|train_lora|AirVLN|LinuxNoEditor|AirSim|CitySample|City_UE52|CrashReport|Unreal|UE4Editor|UE5Editor|PX4|MAVLink" \
    | grep -v grep || true
}

show_ports() {
  echo ""
  echo "[PORT CHECK]"
  for p in "$@"; do
    echo ""
    echo "========== PORT $p =========="
    sudo ss -lntup 2>/dev/null | grep ":$p" || true
    sudo ss -lnuap 2>/dev/null | grep ":$p" || true
    sudo lsof -Pan -iTCP:"$p" -sTCP:LISTEN -n -P 2>/dev/null || true
    sudo lsof -Pan -iUDP:"$p" -n -P 2>/dev/null || true
  done
}

free_port() {
  local p="$1"
  echo "[FREE PORT] TCP/UDP $p"
  sudo fuser -k -n tcp "$p" 2>/dev/null || true
  sudo fuser -k -n udp "$p" 2>/dev/null || true
}

echo "============================================================"
echo "[1/8] Before cleanup: current related processes"
echo "============================================================"
show_processes

echo ""
echo "============================================================"
echo "[2/8] Detect ports from AirSim settings.json / UnrealCV config"
echo "============================================================"

DETECTED_PORTS=()

echo "[AirSim settings.json]"
find "$HOME/.airsim" "$ROOT_DIR" -name "settings.json" -print 2>/dev/null | while read -r f; do
  echo "---- $f"
  grep -nEi "ApiServerPort|ControlPort|Udp|Tcp|Port|MavLink|LocalHostIp" "$f" || true
done

while read -r port; do
  if [[ "$port" =~ ^[0-9]+$ ]]; then
    DETECTED_PORTS+=("$port")
  fi
done < <(
  find "$HOME/.airsim" "$ROOT_DIR" -name "settings.json" -print0 2>/dev/null \
  | xargs -0 grep -hEo '"[^"]*(Port|UdpPort|TcpPort|ApiServerPort|ControlPort|LocalPort|RemotePort)[^"]*"[[:space:]]*:[[:space:]]*[0-9]+' 2>/dev/null \
  | grep -Eo '[0-9]+$' \
  | sort -u
)

echo ""
echo "[UnrealCV config]"
find "$ROOT_DIR" \( -iname "unrealcv.ini" -o -iname "*UnrealCV*.ini" \) -print 2>/dev/null | while read -r f; do
  echo "---- $f"
  grep -nEi "Port|Enable|Width|Height" "$f" || true
done

while read -r port; do
  if [[ "$port" =~ ^[0-9]+$ ]]; then
    DETECTED_PORTS+=("$port")
  fi
done < <(
  find "$ROOT_DIR" \( -iname "unrealcv.ini" -o -iname "*UnrealCV*.ini" \) -print0 2>/dev/null \
  | xargs -0 grep -hEo 'Port[[:space:]]*=[[:space:]]*[0-9]+' 2>/dev/null \
  | grep -Eo '[0-9]+$' \
  | sort -u
)

echo ""
echo "============================================================"
echo "[3/8] Kill OpenFly STEER-VLN train/eval scripts"
echo "============================================================"

kill_by_pattern_soft "STEER-VLN/eval_hd_lora.py"
kill_by_pattern_soft "STEER-VLN/train_integrated_full.py"
kill_by_pattern_soft "STEER-VLN/train_lora_full.py"
kill_by_pattern_soft "STEER-VLN/train_lora_full_core.py"
kill_by_pattern_soft "run_steer_vln"


echo ""
echo "============================================================"
echo "[4/8] Kill AirSim / UE / AirVLN related processes"
echo "============================================================"

kill_by_pattern_soft "AirVLN-Linux-Shipping"
kill_by_pattern_soft "LinuxNoEditor"
kill_by_pattern_soft "AirSim"
kill_by_pattern_soft "CitySample"
kill_by_pattern_soft "City_UE52"
kill_by_pattern_soft "CrashReport"
kill_by_pattern_soft "UE4Editor"
kill_by_pattern_soft "UE5Editor"
kill_by_pattern_soft "PX4"
kill_by_pattern_soft "MAVLink"

sleep 2

echo ""
echo "============================================================"
echo "[5/8] Force kill remaining OpenFly STEER-VLN / AirSim / UE processes"
echo "============================================================"

kill_by_pattern_force "STEER-VLN/eval_hd_lora.py"
kill_by_pattern_force "STEER-VLN/train_integrated_full.py"
kill_by_pattern_force "STEER-VLN/train_lora_full.py"
kill_by_pattern_force "STEER-VLN/train_lora_full_core.py"
kill_by_pattern_force "run_steer_vln"


kill_by_pattern_force "AirVLN-Linux-Shipping"
kill_by_pattern_force "LinuxNoEditor"
kill_by_pattern_force "AirSim"
kill_by_pattern_force "CitySample"
kill_by_pattern_force "City_UE52"
kill_by_pattern_force "CrashReport"
kill_by_pattern_force "UE4Editor"
kill_by_pattern_force "UE5Editor"
kill_by_pattern_force "PX4"
kill_by_pattern_force "MAVLink"

echo ""
echo "============================================================"
echo "[6/8] Free common AirSim / UE / UnrealCV / PX4 ports"
echo "============================================================"

COMMON_PORTS=(
  41451   # AirSim RPC default
  9000    # UnrealCV common port
  14540   # PX4 / AirSim UDP common
  14550   # QGroundControl / MAVLink common
  14560   # MAVLink / PX4 common
  14580   # MAVLink / PX4 common
  14590   # MAVLink / PX4 backup
)

ALL_PORTS=("${COMMON_PORTS[@]}" "${DETECTED_PORTS[@]}")

# 去重
UNIQUE_PORTS=()
for p in "${ALL_PORTS[@]}"; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue

  exists=0
  for q in "${UNIQUE_PORTS[@]}"; do
    if [ "$p" = "$q" ]; then
      exists=1
      break
    fi
  done

  if [ "$exists" -eq 0 ]; then
    UNIQUE_PORTS+=("$p")
  fi
done

echo "[PORT LIST] ${UNIQUE_PORTS[*]}"

echo ""
echo "[BEFORE FREE PORTS]"
show_ports "${UNIQUE_PORTS[@]}"

for p in "${UNIQUE_PORTS[@]}"; do
  free_port "$p"
done

sleep 1

echo ""
echo "============================================================"
echo "[7/8] Detect remaining AirSim / UE PIDs and kill their ports"
echo "============================================================"

PATTERN="AirSim|AirVLN|AirVLN-Linux-Shipping|LinuxNoEditor|CitySample|City_UE52|UE4Editor|UE5Editor|PX4|MAVLink"

PIDS=$(pgrep -f "$PATTERN" | tr '\n' ' ')
echo "[PIDS] $PIDS"

if [ -n "$PIDS" ]; then
  for pid in $PIDS; do
    if [ "$pid" = "$CURRENT_PID" ]; then
      echo "[SKIP] current script PID $pid"
      continue
    fi

    echo ""
    echo "========== PID $pid =========="
    ps -p "$pid" -o pid,ppid,user,stat,etime,cmd || true

    echo ""
    echo "[lsof network]"
    sudo lsof -Pan -p "$pid" -i 2>/dev/null || true

    echo ""
    echo "[kill pid]"
    sudo kill -9 "$pid" 2>/dev/null || true
  done
else
  echo "[OK] No remaining AirSim / UE related process found."
fi

sleep 1

echo ""
echo "============================================================"
echo "[8/8] Final check"
echo "============================================================"

echo ""
echo "[CHECK] Remaining OpenFly / STEER-VLN / AirSim / UE processes:"
show_processes

echo ""
echo "[CHECK] Remaining AirSim / UE ports:"
show_ports "${UNIQUE_PORTS[@]}"

echo ""
echo "[CHECK] GPU:"
nvidia-smi || true

echo ""
echo "============================================================"
echo "[DONE] STEER-VLN / AirSim / UE cleanup finished."
echo "============================================================"
