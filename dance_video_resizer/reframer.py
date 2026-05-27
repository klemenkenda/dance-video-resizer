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

    @property
    def source_aspect(self) -> float:
        return self.source_width / self.source_height


class Reframer:
    def __init__(
        self,
        geometry: FrameGeometry,
        margin_ratio: float = 0.15,
        smoothing_alpha: float = 0.05,
        background_darken: float = 0.25,
        portrait_rectangular_crop: bool = False,
        portrait_foreground_aspect: float = 1.0,
    ) -> None:
        self.geometry = geometry
        self.margin_ratio = max(0.0, margin_ratio)
        self.smoothing_alpha = min(max(smoothing_alpha, 1e-4), 1.0)
        self.background_darken = min(max(background_darken, 0.0), 1.0)
        self.portrait_rectangular_crop = portrait_rectangular_crop
        self.portrait_foreground_aspect = max(0.2, float(portrait_foreground_aspect))
        self._center_kalman: Optional[cv2.KalmanFilter] = None

    def _active_follow_aspect(self) -> float:
        if self._is_narrower_target() and self.portrait_rectangular_crop:
            return self.portrait_foreground_aspect
        return self.geometry.target_aspect

    def _init_center_kalman(self, center_x: float, center_y: float) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 60.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], dtype=np.float32)
        self._center_kalman = kf

    def initial_state(self) -> ReframeState:
        center_x = self.geometry.source_width / 2.0
        center_y = self.geometry.source_height / 2.0
        self._init_center_kalman(center_x, center_y)
        return ReframeState(
            center_x=center_x,
            center_y=center_y,
            crop_height=self._max_crop_height(),
            crop_width=float(self.geometry.source_width),
        )

    def _is_narrower_target(self) -> bool:
        return self.geometry.target_aspect < self.geometry.source_aspect

    def _is_wider_target(self) -> bool:
        return self.geometry.target_aspect > self.geometry.source_aspect

    def _max_crop_height(self) -> float:
        if self._is_wider_target():
            # In portrait-to-landscape mode we can use full source height.
            return float(self.geometry.source_height)
        follow_aspect = self._active_follow_aspect()
        max_h_from_width = float(self.geometry.source_width) / follow_aspect
        return min(float(self.geometry.source_height), max_h_from_width)

    def _compute_wider_target_crop(self, bbox: BBox) -> ReframeState:
        min_crop_height = 32.0
        max_crop_height = self._max_crop_height()
        min_crop_width = 32.0
        max_crop_width = float(self.geometry.source_width)

        x1, y1, x2, y2 = bbox
        subject_width = max(1.0, float(x2 - x1 + 1))
        subject_height = max(1.0, float(y2 - y1 + 1))

        # Tie extra space to the user-facing margin setting so it is tunable
        # from CLI, while keeping a small minimum breathing room.
        extra_pad_x = max(10.0, subject_width * max(0.06, self.margin_ratio * 0.7))
        extra_pad_y = max(16.0, subject_height * max(0.08, self.margin_ratio * 0.95))

        required_width = subject_width + extra_pad_x * 2.0
        required_height = subject_height + extra_pad_y * 2.0

        crop_height = max(required_height, required_width / self.geometry.source_aspect)
        crop_height = max(min_crop_height, min(max_crop_height, crop_height))
        crop_width = max(min_crop_width, min(max_crop_width, crop_height * self.geometry.source_aspect))

        return ReframeState(
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            crop_height=crop_height,
            crop_width=crop_width,
        )

    def compute_target_state(self, bbox: Optional[BBox]) -> ReframeState:
        min_crop_height = 32.0
        max_crop_height = self._max_crop_height()
        follow_aspect = self._active_follow_aspect()
        min_crop_width = 32.0
        max_crop_width = float(self.geometry.source_width)

        if bbox is None:
            return ReframeState(
                center_x=self.geometry.source_width / 2.0,
                center_y=self.geometry.source_height / 2.0,
                crop_height=max_crop_height,
                crop_width=max_crop_width,
            )

        x1, y1, x2, y2 = bbox
        subject_width = max(1.0, float(x2 - x1 + 1))
        subject_height = max(1.0, float(y2 - y1 + 1))

        if self._is_wider_target():
            return self._compute_wider_target_crop(bbox)
        
        # For portrait rectangular mode, use looser framing to avoid cutting heads.
        if self._is_narrower_target() and self.portrait_rectangular_crop:
            desired_subject_fraction = max(0.15, 1.0 - self.margin_ratio * 4.0)
        else:
            desired_subject_fraction = max(0.15, 1.0 - self.margin_ratio * 2.0)

        needed_h_from_height = subject_height / desired_subject_fraction
        needed_h_from_width = (subject_width / desired_subject_fraction) / follow_aspect
        needed_h = max(needed_h_from_height, needed_h_from_width)

        crop_height = max(min_crop_height, min(max_crop_height, needed_h))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return ReframeState(
            center_x=center_x,
            center_y=center_y,
            crop_height=crop_height,
            crop_width=max_crop_width,
        )

    def smooth_state(self, current: ReframeState, target: ReframeState) -> ReframeState:
        a = self.smoothing_alpha
        if self._is_narrower_target():
            if self._center_kalman is None:
                self._init_center_kalman(current.center_x, current.center_y)

            assert self._center_kalman is not None
            predicted = self._center_kalman.predict()
            predicted_x = float(predicted[0, 0])
            predicted_y = float(predicted[1, 0])

            # Gate abrupt detections to prevent one-frame jumps/flicker.
            max_gate_shift_x = self.geometry.source_width * 0.08
            max_gate_shift_y = self.geometry.source_height * 0.08
            gated_x = predicted_x + min(max(target.center_x - predicted_x, -max_gate_shift_x), max_gate_shift_x)
            gated_y = predicted_y + min(max(target.center_y - predicted_y, -max_gate_shift_y), max_gate_shift_y)

            measurement = np.array([[gated_x], [gated_y]], dtype=np.float32)
            corrected = self._center_kalman.correct(measurement)
            filtered_x = float(corrected[0, 0])
            filtered_y = float(corrected[1, 0])

            center_a = max(a, 0.15)
            zoom_a = max(a, 0.10)
            return ReframeState(
                center_x=current.center_x * (1.0 - center_a) + filtered_x * center_a,
                center_y=current.center_y * (1.0 - center_a) + filtered_y * center_a,
                crop_height=current.crop_height * (1.0 - zoom_a) + target.crop_height * zoom_a,
                crop_width=current.crop_width * (1.0 - zoom_a) + target.crop_width * zoom_a,
            )

        if self._is_wider_target():
            center_a = min(0.06, max(a, 0.03))
            zoom_a = min(0.02, max(a, 0.01))
            blended_crop_height = current.crop_height * (1.0 - zoom_a) + target.crop_height * zoom_a
            blended_crop_width = current.crop_width * (1.0 - zoom_a) + target.crop_width * zoom_a
            return ReframeState(
                center_x=current.center_x * (1.0 - center_a) + target.center_x * center_a,
                center_y=current.center_y * (1.0 - center_a) + target.center_y * center_a,
                crop_height=blended_crop_height,
                crop_width=blended_crop_width,
            )

        return ReframeState(
            center_x=current.center_x * (1.0 - a) + target.center_x * a,
            center_y=current.center_y * (1.0 - a) + target.center_y * a,
            crop_height=current.crop_height * (1.0 - a) + target.crop_height * a,
            crop_width=current.crop_width * (1.0 - a) + target.crop_width * a,
        )

    def ensure_subject_fits(self, state: ReframeState, bbox: Optional[BBox]) -> ReframeState:
        if bbox is None or not self._is_wider_target():
            return state

        x1, y1, x2, y2 = bbox
        subject_width = max(1.0, float(x2 - x1 + 1))
        subject_height = max(1.0, float(y2 - y1 + 1))

        min_pad_x = max(10.0, subject_width * max(0.06, self.margin_ratio * 0.7))
        min_pad_y = max(16.0, subject_height * max(0.08, self.margin_ratio * 0.95))

        min_required_w = subject_width + 2.0 * min_pad_x
        min_required_h = subject_height + 2.0 * min_pad_y

        crop_height = max(state.crop_height, min_required_h)
        crop_width = max(state.crop_width, min_required_w)

        crop_height = min(crop_height, float(self.geometry.source_height))
        crop_width = min(crop_width, float(self.geometry.source_width))

        # Keep a portrait strip shape for the foreground crop.
        crop_width = min(crop_width, crop_height * self.geometry.source_aspect)
        crop_height = min(float(self.geometry.source_height), crop_width / self.geometry.source_aspect)

        half_h = crop_height / 2.0
        half_w = crop_width / 2.0

        min_center_y = y2 - half_h
        max_center_y = y1 + half_h
        min_center_x = x2 - half_w
        max_center_x = x1 + half_w

        center_y = min(max(state.center_y, min_center_y), max_center_y)
        center_x = min(max(state.center_x, min_center_x), max_center_x)

        center_x = min(max(center_x, half_w), self.geometry.source_width - half_w)
        center_y = min(max(center_y, half_h), self.geometry.source_height - half_h)

        return ReframeState(
            center_x=center_x,
            center_y=center_y,
            crop_height=crop_height,
            crop_width=crop_width,
        )

    def render_frame(self, frame_bgr: np.ndarray, state: ReframeState) -> np.ndarray:
        src_h, src_w = frame_bgr.shape[:2]

        if self._is_narrower_target() and not self.portrait_rectangular_crop:
            return self._render_cropped_frame(frame_bgr, state)

        if self._is_wider_target():
            return self._render_wider_with_background(frame_bgr, state)

        return self._render_with_background(frame_bgr, state)

    def _render_wider_with_background(self, frame_bgr: np.ndarray, state: ReframeState) -> np.ndarray:
        src_h, src_w = frame_bgr.shape[:2]

        # Full source width always — only crop vertically around the couple.
        crop_h = max(1.0, min(float(src_h), state.crop_height))
        half_h = crop_h / 2.0
        center_y = min(max(state.center_y, half_h), src_h - half_h)

        y1 = max(0, int(round(center_y - half_h)))
        y2 = min(src_h, int(round(center_y + half_h)))

        if y2 <= y1:
            crop = frame_bgr
        else:
            crop = frame_bgr[y1:y2, 0:src_w]

        bg = self._make_background(frame_bgr)

        fg_h, fg_w = crop.shape[:2]
        fg_aspect = fg_w / max(1, fg_h)
        fitted_h = self.geometry.target_height
        fitted_w = max(1, int(round(fitted_h * fg_aspect)))
        fitted_w = min(fitted_w, self.geometry.target_width)

        fitted = cv2.resize(crop, (fitted_w, fitted_h), interpolation=cv2.INTER_LINEAR)

        x_offset = (self.geometry.target_width - fitted_w) // 2
        composed = bg.copy()
        composed[0:fitted_h, x_offset : x_offset + fitted_w] = fitted
        return composed

    def _compute_portrait_rect_crop_box(self, state: ReframeState, src_w: int, src_h: int) -> tuple[int, int, int, int] | None:
        """Compute portrait rectangular crop box from state. Returns (x1, y1, x2, y2) or None."""
        fg_aspect = self.portrait_foreground_aspect
        max_crop_h = min(float(src_h), float(src_w) / fg_aspect)
        crop_h = max(1.0, min(max_crop_h, state.crop_height))
        crop_w = max(1.0, min(float(src_w), crop_h * fg_aspect))
        crop_h = min(float(src_h), crop_w / fg_aspect)

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        center_x = state.center_x
        center_y = state.center_y

        x1 = int(round(center_x - half_w))
        x2 = int(round(center_x + half_w))
        y1 = int(round(center_y - half_h))
        y2 = int(round(center_y + half_h))

        x1 = max(0, min(x1, src_w - 1))
        x2 = max(1, min(x2, src_w))
        y1 = max(0, min(y1, src_h - 1))
        y2 = max(1, min(y2, src_h))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _render_cropped_frame(self, frame_bgr: np.ndarray, state: ReframeState) -> np.ndarray:
        src_h, src_w = frame_bgr.shape[:2]

        crop_h = max(1.0, min(float(src_h), state.crop_height))
        crop_w = max(1.0, min(float(src_w), crop_h * self.geometry.target_aspect))

        # Recompute height from width so the final crop always matches target aspect.
        crop_h = min(float(src_h), crop_w / self.geometry.target_aspect)

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        center_x = min(max(state.center_x, half_w), src_w - half_w)
        center_y = min(max(state.center_y, half_h), src_h - half_h)

        x1 = int(round(center_x - half_w))
        x2 = int(round(center_x + half_w))
        y1 = int(round(center_y - half_h))
        y2 = int(round(center_y + half_h))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(src_w, x2)
        y2 = min(src_h, y2)

        if x2 <= x1 or y2 <= y1:
            crop = frame_bgr
        else:
            crop = frame_bgr[y1:y2, x1:x2]

        return cv2.resize(
            crop,
            (self.geometry.target_width, self.geometry.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _render_with_background(self, frame_bgr: np.ndarray, state: ReframeState) -> np.ndarray:
        src_h, src_w = frame_bgr.shape[:2]

        if self._is_narrower_target() and self.portrait_rectangular_crop:
            crop_box = self._compute_portrait_rect_crop_box(state, src_w, src_h)
            if crop_box is None:
                crop = frame_bgr
            else:
                x1, y1, x2, y2 = crop_box
                crop = frame_bgr[y1:y2, x1:x2]
        else:
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
