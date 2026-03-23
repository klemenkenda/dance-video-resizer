from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception:  # noqa: BLE001
    mp = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # noqa: BLE001
    YOLO = None

from .types import BBox, FrameAnalysis


class DancerDetector:
    """Detector that estimates a bounding box around the front dance couple."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = None
        self._yolo = None
        self._backend = "hog+motion"

        if YOLO is not None:
            try:
                # Auto-downloads weights on first run if missing.
                self._yolo = YOLO("yolov8n-pose.pt")
                self._backend = "yolo-pose"
                print("Using detector backend: YOLO pose")
            except Exception:  # noqa: BLE001
                self._yolo = None

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=False,
        )

        if self._yolo is None and mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._backend = "mediapipe"
            print("Using detector backend: MediaPipe pose")
        else:
            if self._backend != "yolo-pose":
                print("Warning: ML pose backend unavailable. Falling back to HOG + motion detector.")

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()

    def detect(self, frame_bgr: np.ndarray) -> FrameAnalysis:
        if self._backend == "yolo-pose" and self._yolo is not None:
            yolo_bbox = self._detect_yolo_front_couple(frame_bgr)
            if yolo_bbox is not None:
                yolo_bbox = self._expand_for_full_body(frame_bgr.shape[1], frame_bgr.shape[0], yolo_bbox)
                return FrameAnalysis(bbox=yolo_bbox)

        if self._backend == "mediapipe" and self._pose is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)

            if not result.pose_landmarks:
                return FrameAnalysis(bbox=None)

            bbox = self._landmarks_to_bbox(result.pose_landmarks.landmark, frame_bgr.shape[1], frame_bgr.shape[0])
            return FrameAnalysis(bbox=bbox)

        hog_bbox = self._detect_hog_bbox(frame_bgr)
        motion_bbox = self._detect_motion_bbox(frame_bgr)
        bbox = self._select_fallback_bbox(frame_bgr.shape[1], frame_bgr.shape[0], hog_bbox, motion_bbox)
        if bbox is not None:
            bbox = self._expand_for_full_body(frame_bgr.shape[1], frame_bgr.shape[0], bbox)
        return FrameAnalysis(bbox=bbox)

    def _detect_yolo_front_couple(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        try:
            results = self._yolo(frame_bgr, verbose=False, conf=0.2, iou=0.5)
        except Exception:  # noqa: BLE001
            return None

        if not results:
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))

        candidates = []
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf

        for i in range(len(xyxy)):
            if int(cls[i]) != 0:
                continue

            x1, y1, x2, y2 = xyxy[i].tolist()
            x1 = max(0, min(frame_w - 1, int(round(x1))))
            y1 = max(0, min(frame_h - 1, int(round(y1))))
            x2 = max(0, min(frame_w - 1, int(round(x2))))
            y2 = max(0, min(frame_h - 1, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            area = float((x2 - x1) * (y2 - y1))
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            dist = float(np.hypot(center_x - cx, center_y - cy)) / frame_diag

            # Front couple tends to be large and nearer to image center.
            score = (area / (frame_w * frame_h)) * 1.3 + float(conf[i]) * 0.4 - dist * 0.7
            candidates.append((score, (x1, y1, x2, y2)))

        if not candidates:
            return None

        candidates.sort(key=lambda it: it[0], reverse=True)
        selected = [candidates[0][1]]

        # Include second dancer if present and reasonably close to the first.
        first = candidates[0][1]
        fx = (first[0] + first[2]) / 2.0
        fy = (first[1] + first[3]) / 2.0
        for _, box in candidates[1:]:
            bx = (box[0] + box[2]) / 2.0
            by = (box[1] + box[3]) / 2.0
            d = float(np.hypot(bx - fx, by - fy)) / frame_diag
            if d <= 0.35:
                selected.append(box)
                break

        x1 = min(b[0] for b in selected)
        y1 = min(b[1] for b in selected)
        x2 = max(b[2] for b in selected)
        y2 = max(b[3] for b in selected)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _detect_hog_bbox(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        rects, weights = self._hog.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        if len(rects) == 0:
            return None

        selected = []
        for i, (x, y, w, h) in enumerate(rects):
            score = float(weights[i]) if i < len(weights) else 1.0
            if score >= 0.25:
                selected.append((x, y, x + w, y + h))

        if not selected:
            return None

        x1 = max(0, min(r[0] for r in selected))
        y1 = max(0, min(r[1] for r in selected))
        x2 = max(r[2] for r in selected)
        y2 = max(r[3] for r in selected)

        frame_h, frame_w = frame_bgr.shape[:2]
        x2 = min(frame_w - 1, x2)
        y2 = min(frame_h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def _detect_motion_bbox(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        mask = self._bg_subtractor.apply(frame_bgr)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        min_area = frame_h * frame_w * 0.002
        large = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not large:
            return None

        large.sort(key=cv2.contourArea, reverse=True)
        # Keep top contours so both dance partners are included.
        selected = large[:3]

        x1 = frame_w
        y1 = frame_h
        x2 = 0
        y2 = 0
        for c in selected:
            x, y, w, h = cv2.boundingRect(c)
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + w)
            y2 = max(y2, y + h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w - 1, x2)
        y2 = min(frame_h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    @staticmethod
    def _select_fallback_bbox(
        frame_w: int,
        frame_h: int,
        hog_bbox: Optional[BBox],
        motion_bbox: Optional[BBox],
    ) -> Optional[BBox]:
        frame_area = float(frame_w * frame_h)

        if hog_bbox is not None:
            # Prefer HOG as the primary person box. If motion overlaps,
            # blend only when it does not explode to near-full-frame.
            if motion_bbox is None:
                return hog_bbox

            hx1, hy1, hx2, hy2 = hog_bbox
            mx1, my1, mx2, my2 = motion_bbox

            inter_x1 = max(hx1, mx1)
            inter_y1 = max(hy1, my1)
            inter_x2 = min(hx2, mx2)
            inter_y2 = min(hy2, my2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = float(inter_w * inter_h)

            hog_area = float(max(1, (hx2 - hx1) * (hy2 - hy1)))
            motion_area = float(max(1, (mx2 - mx1) * (my2 - my1)))

            if intersection <= 0:
                return hog_bbox

            union_x1 = min(hx1, mx1)
            union_y1 = min(hy1, my1)
            union_x2 = max(hx2, mx2)
            union_y2 = max(hy2, my2)
            union_area = float(max(1, (union_x2 - union_x1) * (union_y2 - union_y1)))

            motion_ratio = motion_area / frame_area
            union_ratio = union_area / frame_area
            iou = intersection / max(1.0, hog_area + motion_area - intersection)

            # Expand with motion only when reasonably consistent and not huge.
            if iou >= 0.1 and motion_ratio <= 0.55 and union_ratio <= 0.65:
                return (union_x1, union_y1, union_x2, union_y2)

            return hog_bbox

        if motion_bbox is None:
            return None

        mx1, my1, mx2, my2 = motion_bbox
        motion_area = float(max(1, (mx2 - mx1) * (my2 - my1)))
        motion_ratio = motion_area / frame_area

        # Reject near-full-frame motion boxes; they do not help zooming.
        if motion_ratio >= 0.7:
            return None

        return motion_bbox

    @staticmethod
    def _expand_for_full_body(frame_w: int, frame_h: int, bbox: BBox) -> BBox:
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

        # HOG often catches torso-only regions; expand to likely full body/couple extent.
        expand_x = 1.18
        expand_y = 1.45
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0 - h * 0.08

        new_w = w * expand_x
        new_h = h * expand_y

        nx1 = int(round(center_x - new_w / 2.0))
        nx2 = int(round(center_x + new_w / 2.0))
        ny1 = int(round(center_y - new_h / 2.0))
        ny2 = int(round(center_y + new_h / 2.0))

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(frame_w - 1, nx2)
        ny2 = min(frame_h - 1, ny2)

        if nx2 <= nx1 or ny2 <= ny1:
            return bbox

        return (nx1, ny1, nx2, ny2)

    @staticmethod
    def _landmarks_to_bbox(landmarks, frame_w: int, frame_h: int) -> Optional[BBox]:
        xs = []
        ys = []

        for lm in landmarks:
            # Ignore very low confidence landmarks.
            if lm.visibility < 0.3:
                continue
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            xs.append(x)
            ys.append(y)

        if not xs or not ys:
            return None

        x1 = max(0, min(xs))
        x2 = min(frame_w - 1, max(xs))
        y1 = max(0, min(ys))
        y2 = min(frame_h - 1, max(ys))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)
