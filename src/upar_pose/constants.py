"""Competition keypoint schema and the public COCO evaluator defaults."""

from __future__ import annotations

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "head_bottom",
    "head_top",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# The organizer states that these slots are unannotated and ignored by scoring.
IGNORED_KEYPOINT_INDICES: tuple[int, ...] = (3, 4)

# pycocotools.cocoeval.COCOeval's default 17-slot keypoint sigmas.  The public
# competition page specifies COCOeval but the organizer's scoring bundle is not
# public, so the local-validation manifest records this as an explicit parity
# assumption rather than claiming private-scorer identity.
COCO_KEYPOINT_SIGMAS: tuple[float, ...] = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)

COMPETITION_ID = 18273
PHASE_ID = 30830
