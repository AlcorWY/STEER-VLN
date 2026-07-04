"""
Auto-loaded by Python when running scripts from train/.

Purpose:
    Make OpenFly project root importable for scripts like:
        python train/evalv2_3_learned.py
        python train/evalv2_4_xxx.py

Why insert at index 1:
    sys.path[0] is usually the script directory, i.e. train/.
    Keep train/ first so imports like `from common import *` preserve original behavior.
    Put project root right after train/ so root-level packages like keyframe/ are importable.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    insert_at = 1 if len(sys.path) > 0 else 0
    sys.path.insert(insert_at, PROJECT_ROOT_STR)
