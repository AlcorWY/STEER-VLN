import json
from pathlib import Path
for name in ["test_seen", "test_unseen"]:
    p = Path(f"dataset/Annotation/{name}.json")
    if not p.exists():
        print(f"[FAIL] missing {p}")
        raise SystemExit(1)
    data = json.load(open(p, encoding="utf-8"))
    print(f"[OK] {p}: episodes={len(data)}")
    if not isinstance(data, list) or not data:
        raise SystemExit(f"bad json: {p}")
    keys = set(data[0].keys())
    need = {"image_path", "gpt_instruction", "action", "index_list", "pos", "yaw"}
    missing = sorted(need - keys)
    if missing:
        raise SystemExit(f"{p} missing keys: {missing}")
