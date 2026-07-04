"""
STEER_VLN compatibility shim.

Real implementation:
    STEER-VLN/keyframe/openfly_feature_dual_dataset.py
"""

from keyframe.openfly_feature_dual_dataset import (
    OpenFlyFeatureDualDataset,
    collate_openfly_feature_dual_batch,
)

__all__ = [
    "OpenFlyFeatureDualDataset",
    "collate_openfly_feature_dual_batch",
]
