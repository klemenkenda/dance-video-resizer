from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import BBox, ReframeState


@dataclass(frozen=True)
class FrameGeometry:
    source_width: int
    source_height: int
    target_width: int
    target_height: int

    @property
    def target_aspect(self) -> float:
        return self.target_width / self.target_height


class Reframer:
    def __init__(
        self,
        geometry: FrameGeometry,
        margin_ratio: float = 0.15,
        smoothing_alpha: float = 0.05,
        background_darken: float = 0.25,
    ) -> None:
        self.geometry = geometry
        self.margin_ratio = max(0.0, margin_ratio)
        self.smoothing_alpha = min(max(smoothing_alpha, 1e-4), 1.0)
        self.background_darken = min(max(background_darken, 0.0), 1.0)

    def initial_state(self) -> ReframeState:
        return ReframeState(
            center_x=self.geometry.source_width / 2.0,
            center_y=self.geometry.source_height / 2.0,
            crop_height=float(self.geometry.source_height),
        )

    def compute_target_state(self, bbox: Optional[BBox]) -> ReframeState:
        min_crop_height = 32.0
        # Allow full source height — do not constrain by target aspect ratio here.
        max_crop_height = float(self.geometry.source_height)

        if bbox is None:
            return ReframeState(
                center_x=self.geometry.source_width / 2.0,
                center_y=self.geometry.source_height / 2.0,
                crop_height=max_crop_height,
            )

        x1, y1, x2, y2 = bbox
        subject_height = max(1.0, float(y2 - y1 + 1))
        desired_subject_fraction = max(0.15, 1.0 - self.margin_ratio * 2.0)
        needed_h = subject_height / desired_subject_fraction

        crop_height = max(min_crop_height, min(max_crop_height, needed_h))
        center_x = self.geometry.source_width / 2.0
        center_y = (y1 + y2) / 2.0
        return ReframeState(center_x=center_x, center_y=center_y, crop_height=crop_height)

    def smooth_state(self, current: ReframeState, target: ReframeState) -> ReframeState:
        a = self.smoothing_alpha
        return ReframeState(
            center_x=current.center_x * (1.0 - a) + target.center_x * a,
            center_y=current.center_y * (1.0 - a) + target.center_y * a,
            crop_height=current.crop_height * (1.0 - a) + target.crop_height * a,
        )

    def render_frame(self, frame_bgr: np.ndarray, state: ReframeState) -> np.ndarray:
        src_h, src_w = frame_bgr.shape[:2]

        # Always crop full source width; only vary vertical extent.
        crop_w = float(src_w)
        crop_h = max(1.0, min(float(src_h), state.crop_height))

        half_h = crop_h / 2.0
        center_x = src_w / 2.0
        center_y = min(max(state.center_y, half_h), src_h - half_h)

        x1 = 0
        x2 = src_w
        y1 = int(round(center_y - half_h))
        y2 = int(round(center_y + half_h))

        y1 = max(0, y1)
        y2 = min(src_h, y2)

        if y2 <= y1:
            x1, x2, y1, y2 = 0, src_w, 0, src_h

        crop = frame_bgr[y1:y2, x1:x2]
        bg = self._make_background(frame_bgr)

        fg_h, fg_w = crop.shape[:2]
        fg_aspect = fg_w / fg_h
        tgt_aspect = self.geometry.target_aspect

        if fg_aspect >= tgt_aspect:
            fitted_w = self.geometry.target_width
            fitted_h = max(1, int(round(fitted_w / fg_aspect)))
        else:
            fitted_h = self.geometry.target_height
            fitted_w = max(1, int(round(fitted_h * fg_aspect)))

        fitted = cv2.resize(crop, (fitted_w, fitted_h), interpolation=cv2.INTER_LINEAR)

        y_offset = (self.geometry.target_height - fitted_h) // 2
        x_offset = (self.geometry.target_width - fitted_w) // 2
        composed = bg.copy()
        composed[y_offset : y_offset + fitted_h, x_offset : x_offset + fitted_w] = fitted
        return composed

    def _make_background(self, frame_bgr: np.ndarray) -> np.ndarray:
        src_h, src_w = frame_bgr.shape[:2]
        target_w = self.geometry.target_width
        target_h = self.geometry.target_height

        # Preserve source aspect ratio for the background by scaling to cover,
        # then center-cropping to the target canvas.
        scale = max(target_w / src_w, target_h / src_h)
        scaled_w = max(1, int(round(src_w * scale)))
        scaled_h = max(1, int(round(src_h * scale)))
        scaled = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        x1 = max(0, (scaled_w - target_w) // 2)
        y1 = max(0, (scaled_h - target_h) // 2)
        bg = scaled[y1 : y1 + target_h, x1 : x1 + target_w]

        if bg.shape[1] != target_w or bg.shape[0] != target_h:
            bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if self.background_darken > 0:
            dark = np.zeros_like(bg)
            bg = cv2.addWeighted(bg, 1.0 - self.background_darken, dark, self.background_darken, 0)
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=12, sigmaY=12)
        return bg
