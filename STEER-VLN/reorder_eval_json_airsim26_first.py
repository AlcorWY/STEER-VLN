#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for reordering any OpenFly eval json."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--filter_gs", action="store_true")
    ap.add_argument("--filter_gtav", action="store_true")
    args = ap.parse_args()

    maker = Path(__file__).with_name("make_openfly_no_gs_no_gtav_eval_json.py")
    out = Path(args.output) if args.output else Path(args.input)

    # Use maker logic. It writes eval_test_ordered_no_gs_no_gtav.json into a temp output dir,
    # then copy/overwrite to requested output. This keeps all ordering rules in one file.
    tmp_dir = out.parent / ".tmp_reorder_eval_json"
    cmd = [
        sys.executable,
        str(maker),
        "--eval_test", args.input,
        "--only", "eval_test",
        "--out_dir", str(tmp_dir),
        "--no_legacy_alias",
    ]
    if not args.filter_gs:
        cmd.append("--keep_gs")
    if not args.filter_gtav:
        cmd.append("--keep_gtav")

    subprocess.check_call(cmd)
    generated = tmp_dir / "eval_test_ordered_no_gs_no_gtav.json"

    if args.backup and out == Path(args.input):
        bak = Path(args.input).with_suffix(Path(args.input).suffix + ".bak")
        bak.write_text(Path(args.input).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[BACKUP] {bak}")

    out.write_text(generated.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[WRITE] {out}")


if __name__ == "__main__":
    main()
