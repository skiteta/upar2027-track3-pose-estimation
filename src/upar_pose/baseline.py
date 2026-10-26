"""A deliberately small, auditable top-down pose baseline.

The model learns only a mean pose in bounding-box-normalized coordinates.  It
is weak, but its fit boundary is unambiguous and its inference API mirrors the
competition's ``predict_image(sample)`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constants import KEYPOINT_NAMES


def _finite_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class NormalizedMeanPose:
    """Mean keypoint locations and a train-only box-aspect confidence prior."""

    normalized_xy: tuple[tuple[float, float], ...]
    observation_counts: tuple[int, ...]
    fit_annotation_count: int
    labelled_fit_annotation_count: int
    median_aspect_ratio: float
    log_aspect_mad: float

    @classmethod
    def fit(cls, datasets: Iterable[Mapping[str, Any]]) -> "NormalizedMeanPose":
        sums = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float64)
        counts = np.zeros(len(KEYPOINT_NAMES), dtype=np.int64)
        aspects: list[float] = []
        fit_annotation_count = 0
        labelled_fit_annotation_count = 0

        for dataset in datasets:
            for ann in dataset.get("annotations", []):
                if int(ann.get("category_id", 1)) != 1:
                    continue
                fit_annotation_count += 1
                bbox = ann.get("bbox", [])
                if len(bbox) != 4 or not _finite_positive(bbox[2]) or not _finite_positive(bbox[3]):
                    continue
                x0, y0, width, height = (float(value) for value in bbox)
                aspects.append(width / height)
                raw = ann.get("keypoints", [])
                if len(raw) != 3 * len(KEYPOINT_NAMES):
                    continue
                keypoints = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
                visible = keypoints[:, 2] > 0
                if np.any(visible):
                    labelled_fit_annotation_count += 1
                normalized = np.column_stack(
                    ((keypoints[:, 0] - x0) / width, (keypoints[:, 1] - y0) / height)
                )
                finite = np.isfinite(normalized).all(axis=1)
                usable = visible & finite
                sums[usable] += normalized[usable]
                counts[usable] += 1

        if labelled_fit_annotation_count == 0:
            raise ValueError("No labelled training annotations were available for fitting")
        if not aspects:
            raise ValueError("No valid training bounding boxes were available for fitting")

        # Missing slots (the two ears in this challenge) are placed at the mean
        # of the annotated head triplet.  They are retained only for the required
        # 17-slot output and are ignored by the official metric.
        normalized_xy = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float64)
        observed = counts > 0
        normalized_xy[observed] = sums[observed] / counts[observed, None]
        head_observed = [idx for idx in (0, 1, 2) if observed[idx]]
        fallback = (
            normalized_xy[head_observed].mean(axis=0)
            if head_observed
            else np.asarray([0.5, 0.2], dtype=np.float64)
        )
        normalized_xy[~observed] = fallback

        aspect_array = np.asarray(aspects, dtype=np.float64)
        median_aspect = float(np.median(aspect_array))
        log_distance = np.abs(np.log(aspect_array / median_aspect))
        log_aspect_mad = float(np.median(log_distance))

        return cls(
            normalized_xy=tuple((float(x), float(y)) for x, y in normalized_xy),
            observation_counts=tuple(int(value) for value in counts),
            fit_annotation_count=fit_annotation_count,
            labelled_fit_annotation_count=labelled_fit_annotation_count,
            median_aspect_ratio=median_aspect,
            log_aspect_mad=log_aspect_mad,
        )

    def _box_score(self, bbox: Sequence[float]) -> float:
        width, height = float(bbox[2]), float(bbox[3])
        if width <= 0 or height <= 0:
            return 0.01
        aspect = width / height
        # Train-only robust distance.  The floor prevents a near-zero MAD from
        # turning benign aspect variation into numerical underflow.
        scale = max(1.4826 * self.log_aspect_mad, 0.05)
        distance = abs(math.log(aspect / self.median_aspect_ratio)) / scale
        return float(np.clip(math.exp(-0.5 * distance), 0.01, 1.0))

    def predict_sample(self, sample: Mapping[str, Any]) -> dict[str, list[Any]]:
        """Predict in the same box order as the Codabench ingestion API."""

        boxes = sample.get("boxes", [])
        width = int(sample.get("width", 0))
        height = int(sample.get("height", 0))
        keypoints: list[list[list[float]]] = []
        scores: list[float] = []
        normalized = np.asarray(self.normalized_xy, dtype=np.float64)

        for bbox in boxes:
            if len(bbox) != 4 or not _finite_positive(bbox[2]) or not _finite_positive(bbox[3]):
                raise ValueError(f"Invalid xywh bounding box: {bbox!r}")
            x0, y0, box_width, box_height = (float(value) for value in bbox)
            coordinates = np.empty_like(normalized)
            coordinates[:, 0] = x0 + normalized[:, 0] * box_width
            coordinates[:, 1] = y0 + normalized[:, 1] * box_height
            if width > 0:
                coordinates[:, 0] = np.clip(coordinates[:, 0], 0.0, max(width - 1.0, 0.0))
            if height > 0:
                coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, max(height - 1.0, 0.0))
            keypoints.append(
                [[float(x), float(y), 1.0] for x, y in coordinates]
            )
            scores.append(self._box_score(bbox))
        return {"keypoints": keypoints, "scores": scores}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "normalized_mean_pose",
            "keypoint_names": list(KEYPOINT_NAMES),
            "normalized_xy": [list(pair) for pair in self.normalized_xy],
            "observation_counts": list(self.observation_counts),
            "fit_annotation_count": self.fit_annotation_count,
            "labelled_fit_annotation_count": self.labelled_fit_annotation_count,
            "confidence": {
                "type": "train_only_robust_box_aspect_prior",
                "median_aspect_ratio": self.median_aspect_ratio,
                "log_aspect_mad": self.log_aspect_mad,
            },
        }
