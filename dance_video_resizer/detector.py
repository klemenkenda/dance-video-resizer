from __future__ import annotations

from typing import Optional, Tuple

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
        gender_focus: Optional[str] = None,
    ) -> None:
        self._pose = None
        self._gender_pose = None
        self._yolo = None
        self._backend = "hog+motion"
        self.gender_focus = gender_focus  # "male", "female", or None for auto
        self._track_center: Optional[Tuple[float, float]] = None
        self._track_velocity: Tuple[float, float] = (0.0, 0.0)
        self._track_bbox: Optional[BBox] = None
        self._track_misses = 0
        self._couple_tracks: list[Optional[dict]] = [None, None]
        self._couple_initialized = False
        self._pair_dist_ref: Optional[float] = None

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

        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                # Separate lightweight pose model for gender heuristics even when YOLO is active.
                self._gender_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            except Exception:  # noqa: BLE001
                self._gender_pose = None

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
        if self._gender_pose is not None and self._gender_pose is not self._pose:
            self._gender_pose.close()

    def detect(self, frame_bgr: np.ndarray) -> FrameAnalysis:
        if self._backend == "yolo-pose" and self._yolo is not None:
            yolo_result = self._detect_yolo_front_couple(frame_bgr)
            if yolo_result is not None:
                yolo_bbox, gender, confidence = yolo_result
                yolo_bbox = self._expand_for_full_body(frame_bgr.shape[1], frame_bgr.shape[0], yolo_bbox)
                yolo_bbox = self._stabilize_bbox(yolo_bbox, frame_bgr.shape[1], frame_bgr.shape[0])
                self._update_track_from_bbox(yolo_bbox)
                if gender is None:
                    gender, confidence = self._extract_gender_for_bbox(frame_bgr, yolo_bbox)
                return FrameAnalysis(
                    bbox=yolo_bbox,
                    gender=gender,
                    gender_confidence=confidence,
                    tracked_points=self._get_tracked_points(),
                )
            self._on_tracking_miss()
            tracked_bbox = self._predict_tracked_bbox(frame_bgr.shape[1], frame_bgr.shape[0])
            if tracked_bbox is not None:
                return FrameAnalysis(
                    bbox=tracked_bbox,
                    gender=None,
                    gender_confidence=0.0,
                    tracked_points=self._get_tracked_points(),
                )
            return FrameAnalysis(bbox=None)

        if self._backend == "mediapipe" and self._pose is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)

            if not result.pose_landmarks:
                self._on_tracking_miss()
                return FrameAnalysis(bbox=None)

            bbox = self._landmarks_to_bbox(result.pose_landmarks.landmark, frame_bgr.shape[1], frame_bgr.shape[0])
            if bbox is not None:
                bbox = self._stabilize_bbox(bbox, frame_bgr.shape[1], frame_bgr.shape[0])
                self._update_track_from_bbox(bbox)
                gender, confidence = self._classify_gender_from_landmarks(result.pose_landmarks.landmark)
                return FrameAnalysis(
                    bbox=bbox,
                    gender=gender,
                    gender_confidence=confidence,
                    tracked_points=self._get_tracked_points(),
                )
            self._on_tracking_miss()
            return FrameAnalysis(bbox=None)

        hog_bbox = self._detect_hog_bbox(frame_bgr)
        motion_bbox = self._detect_motion_bbox(frame_bgr)
        bbox = self._select_fallback_bbox(frame_bgr.shape[1], frame_bgr.shape[0], hog_bbox, motion_bbox)
        if bbox is not None:
            bbox = self._expand_for_full_body(frame_bgr.shape[1], frame_bgr.shape[0], bbox)
            bbox = self._stabilize_bbox(bbox, frame_bgr.shape[1], frame_bgr.shape[0])
            self._update_track_from_bbox(bbox)
            gender, confidence = self._extract_gender_for_bbox(frame_bgr, bbox)
            return FrameAnalysis(
                bbox=bbox,
                gender=gender,
                gender_confidence=confidence,
                tracked_points=self._get_tracked_points(),
            )
        self._on_tracking_miss()
        return FrameAnalysis(bbox=None)

    def detect_people_boxes(self, frame_bgr: np.ndarray) -> list[BBox]:
        """Return all detected person boxes for debug visualization."""
        frame_h, frame_w = frame_bgr.shape[:2]

        if self._backend == "yolo-pose" and self._yolo is not None:
            detections = self._extract_yolo_person_detections(frame_bgr)
            boxes: list[BBox] = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                x1 = max(0, min(frame_w - 1, int(x1)))
                y1 = max(0, min(frame_h - 1, int(y1)))
                x2 = max(0, min(frame_w - 1, int(x2)))
                y2 = max(0, min(frame_h - 1, int(y2)))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
            return boxes

        if self._backend == "mediapipe" and self._pose is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
            if result.pose_landmarks:
                bbox = self._landmarks_to_bbox(result.pose_landmarks.landmark, frame_w, frame_h)
                if bbox is not None:
                    return [bbox]
            return []

        rects, weights = self._hog.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes = []
        for i, (x, y, w, h) in enumerate(rects):
            score = float(weights[i]) if i < len(weights) else 1.0
            if score < 0.25:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1 = max(0, min(frame_w - 1, int(x1)))
            y1 = max(0, min(frame_h - 1, int(y1)))
            x2 = max(0, min(frame_w - 1, int(x2)))
            y2 = max(0, min(frame_h - 1, int(y2)))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _get_tracked_points(self) -> list[Tuple[int, int]]:
        points: list[Tuple[int, int]] = []
        for track in self._couple_tracks:
            if track is None:
                continue
            cx, cy = track["center"]
            points.append((int(round(cx)), int(round(cy))))
        return points

    def _detect_yolo_front_couple(self, frame_bgr: np.ndarray) -> Optional[tuple[BBox, Optional[str], float]]:
        detections = self._extract_yolo_person_detections(frame_bgr)
        frame_h, frame_w = frame_bgr.shape[:2]

        if not self._couple_initialized and len(detections) >= 2:
            self._initialize_couple_tracks(detections, frame_w, frame_h)

        self._update_couple_tracks(detections, frame_w, frame_h)

        couple_bbox = self._union_couple_bbox(frame_w, frame_h)
        if couple_bbox is None:
            return None

        gender, confidence = self._extract_gender_for_bbox(frame_bgr, couple_bbox)
        return couple_bbox, gender, confidence

    def _extract_yolo_person_detections(self, frame_bgr: np.ndarray) -> list[dict]:
        try:
            results = self._yolo(frame_bgr, verbose=False, conf=0.12, iou=0.6)
        except Exception:  # noqa: BLE001
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))

        detections: list[dict] = []
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

            w = float(x2 - x1)
            h = float(y2 - y1)
            if h <= 1.0:
                continue
            area_ratio = (w * h) / float(max(1, frame_w * frame_h))
            aspect = w / h

            # Reject likely merged multi-person boxes. A single dancer box is typically
            # taller than wide and should not dominate large fractions of the frame.
            if area_ratio > 0.24:
                continue
            if aspect > 1.15:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            dist_center = float(np.hypot(center_x - cx, center_y - cy)) / frame_diag
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "dist_center": dist_center,
                    "conf": float(conf[i]),
                    "w": w,
                    "h": h,
                }
            )

        return detections

    def _initialize_couple_tracks(self, detections: list[dict], frame_w: int, frame_h: int) -> None:
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))
        central = [d for d in detections if d["dist_center"] <= 0.28]
        if len(central) < 2:
            central = [d for d in detections if d["dist_center"] <= 0.38]
        pool_src = central if len(central) >= 2 else detections
        ordered = sorted(pool_src, key=lambda d: (d["dist_center"], -d["conf"]))[:8]

        best_pair = None
        best_score = float("inf")
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                ax, ay = a["center"]
                bx, by = b["center"]
                pair_dist = float(np.hypot(ax - bx, ay - by)) / frame_diag
                if pair_dist < 0.04 or pair_dist > 0.65:
                    continue
                # Strongly prioritize centrality, then confidence.
                score = (a["dist_center"] + b["dist_center"]) * 2.0 - (a["conf"] + b["conf"]) * 0.25
                if score < best_score:
                    best_score = score
                    best_pair = (a, b, pair_dist)

        if best_pair is not None:
            selected = [best_pair[0], best_pair[1]]
            self._pair_dist_ref = best_pair[2]
        else:
            selected = ordered[:2]
            if len(selected) == 2:
                ax, ay = selected[0]["center"]
                bx, by = selected[1]["center"]
                self._pair_dist_ref = float(np.hypot(ax - bx, ay - by)) / frame_diag

        for idx in range(2):
            if idx < len(selected):
                det = selected[idx]
                self._couple_tracks[idx] = {
                    "bbox": det["bbox"],
                    "center": det["center"],
                    "velocity": (0.0, 0.0),
                    "misses": 0,
                }
            else:
                self._couple_tracks[idx] = None
        self._couple_initialized = True

    def _update_couple_tracks(self, detections: list[dict], frame_w: int, frame_h: int) -> None:
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0
        prev_track_centers = [
            track["center"] if track is not None else None
            for track in self._couple_tracks
        ]
        prev_pair_dist = None
        if prev_track_centers[0] is not None and prev_track_centers[1] is not None:
            ax, ay = prev_track_centers[0]
            bx, by = prev_track_centers[1]
            prev_pair_dist = float(np.hypot(ax - bx, ay - by)) / frame_diag

        used_indices: set[int] = set()
        matched_meta: dict[int, tuple[int, float]] = {}

        for track_idx, track in enumerate(self._couple_tracks):
            if track is None:
                continue

            max_match_dist = 0.16

            cx, cy = track["center"]
            vx, vy = track["velocity"]
            pred_x = cx + vx
            pred_y = cy + vy

            best_idx = None
            best_dist = float("inf")
            tx1, ty1, tx2, ty2 = track["bbox"]
            tw = max(1.0, float(tx2 - tx1))
            th = max(1.0, float(ty2 - ty1))
            for det_idx, det in enumerate(detections):
                if det_idx in used_indices:
                    continue
                dx, dy = det["center"]
                dist = float(np.hypot(dx - pred_x, dy - pred_y)) / frame_diag

                # Keep tracked identities in the central dance region.
                if det["dist_center"] > 0.48:
                    continue

                # Keep identity by requiring rough size consistency.
                size_w_ratio = det["w"] / tw
                size_h_ratio = det["h"] / th
                if size_w_ratio < 0.55 or size_w_ratio > 1.85:
                    continue
                if size_h_ratio < 0.55 or size_h_ratio > 1.85:
                    continue

                # Keep pair geometry consistent with the initially selected central couple.
                other_track = self._couple_tracks[1 - track_idx]
                if other_track is not None and self._pair_dist_ref is not None:
                    ox, oy = other_track["center"]
                    pair_dist = float(np.hypot(dx - ox, dy - oy)) / frame_diag
                    min_pair = max(0.03, self._pair_dist_ref * 0.35)
                    max_pair = min(0.90, self._pair_dist_ref * 1.85)
                    if pair_dist < min_pair or pair_dist > max_pair:
                        continue

                if dist < best_dist:
                    best_dist = dist
                    best_idx = det_idx

            if best_idx is not None and best_dist <= max_match_dist:
                det = detections[best_idx]
                used_indices.add(best_idx)
                matched_meta[track_idx] = (best_idx, best_dist)
                det_cx, det_cy = det["center"]
                meas_vx = det_cx - cx
                meas_vy = det_cy - cy
                new_vx = vx * 0.70 + meas_vx * 0.30
                new_vy = vy * 0.70 + meas_vy * 0.30
                self._couple_tracks[track_idx] = {
                    "bbox": det["bbox"],
                    "center": (det_cx, det_cy),
                    "velocity": (new_vx, new_vy),
                    "misses": 0,
                }
                continue

            # Detection mismatch is too large: keep continuity by predicting from history.
            pred_vx = vx * 0.85
            pred_vy = vy * 0.85
            pred_cx = cx + pred_vx
            pred_cy = cy + pred_vy
            pred_bbox = self._shift_bbox(track["bbox"], pred_vx, pred_vy, frame_w, frame_h)
            misses = int(track["misses"]) + 1
            if misses > 120:
                self._couple_tracks[track_idx] = None
            else:
                self._couple_tracks[track_idx] = {
                    "bbox": pred_bbox,
                    "center": (pred_cx, pred_cy),
                    "velocity": (pred_vx, pred_vy),
                    "misses": misses,
                }

        # Enforce pair consistency: if one matched point makes pair distance jump too much,
        # drop that match and keep prediction for that identity.
        if prev_pair_dist is not None and self._couple_tracks[0] is not None and self._couple_tracks[1] is not None:
            a = self._couple_tracks[0]["center"]
            b = self._couple_tracks[1]["center"]
            cur_pair_dist = float(np.hypot(a[0] - b[0], a[1] - b[1])) / frame_diag
            max_delta = max(0.12, prev_pair_dist * 0.42)

            if abs(cur_pair_dist - prev_pair_dist) > max_delta:
                cands = [(idx, matched_meta[idx][1]) for idx in matched_meta.keys()]
                if cands:
                    # Revert the weaker (larger-distance) reassignment.
                    revert_idx = max(cands, key=lambda it: it[1])[0]
                    prev = prev_track_centers[revert_idx]
                    if prev is not None:
                        track = self._couple_tracks[revert_idx]
                        assert track is not None
                        vx, vy = track["velocity"]
                        pred_bbox = self._shift_bbox(track["bbox"], vx * 0.85, vy * 0.85, frame_w, frame_h)
                        self._couple_tracks[revert_idx] = {
                            "bbox": pred_bbox,
                            "center": prev,
                            "velocity": (vx * 0.85, vy * 0.85),
                            "misses": int(track["misses"]) + 1,
                        }

        # If both tracks drift too far from center, reset to current central detections.
        if self._couple_tracks[0] is not None and self._couple_tracks[1] is not None:
            a = self._couple_tracks[0]["center"]
            b = self._couple_tracks[1]["center"]
            mid_x = (a[0] + b[0]) / 2.0
            mid_y = (a[1] + b[1]) / 2.0
            mid_dist = float(np.hypot(mid_x - frame_cx, mid_y - frame_cy)) / frame_diag
            if mid_dist > 0.42:
                central = [d for d in detections if d["dist_center"] <= 0.30]
                if len(central) >= 2:
                    ordered = sorted(central, key=lambda d: (d["dist_center"], -d["conf"]))
                    for idx in range(2):
                        det = ordered[idx]
                        self._couple_tracks[idx] = {
                            "bbox": det["bbox"],
                            "center": det["center"],
                            "velocity": (0.0, 0.0),
                            "misses": 0,
                        }
                    ax, ay = self._couple_tracks[0]["center"]
                    bx, by = self._couple_tracks[1]["center"]
                    self._pair_dist_ref = float(np.hypot(ax - bx, ay - by)) / frame_diag

        # Do not recruit new people into couple tracks once initialized.

    def _shift_bbox(self, bbox: BBox, vx: float, vy: float, frame_w: int, frame_h: int) -> BBox:
        x1, y1, x2, y2 = bbox
        nx1 = int(round(x1 + vx))
        ny1 = int(round(y1 + vy))
        nx2 = int(round(x2 + vx))
        ny2 = int(round(y2 + vy))

        w = max(1, nx2 - nx1)
        h = max(1, ny2 - ny1)
        nx1 = max(0, min(frame_w - 1, nx1))
        ny1 = max(0, min(frame_h - 1, ny1))
        nx2 = max(nx1 + 1, min(frame_w - 1, nx1 + w))
        ny2 = max(ny1 + 1, min(frame_h - 1, ny1 + h))
        return (nx1, ny1, nx2, ny2)

    def _union_couple_bbox(self, frame_w: int, frame_h: int) -> Optional[BBox]:
        active_tracks = [t for t in self._couple_tracks if t is not None]
        if not active_tracks:
            return None

        x1 = min(t["bbox"][0] for t in active_tracks)
        y1 = min(t["bbox"][1] for t in active_tracks)
        x2 = max(t["bbox"][2] for t in active_tracks)
        y2 = max(t["bbox"][3] for t in active_tracks)

        # Preserve couple width if one track is temporarily missing.
        if len(active_tracks) == 1 and self._track_bbox is not None:
            tx1, ty1, tx2, ty2 = self._track_bbox
            hist_w = max(1.0, float(tx2 - tx1))
            hist_h = max(1.0, float(ty2 - ty1))
            cur_w = max(1.0, float(x2 - x1))
            cur_h = max(1.0, float(y2 - y1))
            keep_w = max(cur_w, hist_w * 0.94)
            keep_h = max(cur_h, hist_h * 0.88)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            x1 = int(round(cx - keep_w / 2.0))
            x2 = int(round(cx + keep_w / 2.0))
            y1 = int(round(cy - keep_h / 2.0))
            y2 = int(round(cy + keep_h / 2.0))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w - 1, x2)
        y2 = min(frame_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _predict_track_center(self) -> Optional[Tuple[float, float]]:
        if self._track_center is None:
            return None
        return (
            self._track_center[0] + self._track_velocity[0],
            self._track_center[1] + self._track_velocity[1],
        )

    def _update_track_from_bbox(self, bbox: BBox) -> None:
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        if self._track_center is not None:
            measured_vx = center_x - self._track_center[0]
            measured_vy = center_y - self._track_center[1]
            self._track_velocity = (
                self._track_velocity[0] * 0.65 + measured_vx * 0.35,
                self._track_velocity[1] * 0.65 + measured_vy * 0.35,
            )

        self._track_center = (center_x, center_y)
        self._track_bbox = bbox
        self._track_misses = 0

    def _stabilize_bbox(self, bbox: BBox, frame_w: int, frame_h: int) -> BBox:
        if self._track_bbox is None:
            return bbox

        x1, y1, x2, y2 = bbox
        tx1, ty1, tx2, ty2 = self._track_bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(8.0, float(x2 - x1))
        h = max(8.0, float(y2 - y1))

        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0
        tw = max(8.0, float(tx2 - tx1))
        th = max(8.0, float(ty2 - ty1))

        # Bound per-frame movement to avoid jitter while keeping open-position flexibility.
        max_shift_x = frame_w * 0.12
        max_shift_y = frame_h * 0.12
        cx = tcx + min(max(cx - tcx, -max_shift_x), max_shift_x)
        cy = tcy + min(max(cy - tcy, -max_shift_y), max_shift_y)

        max_scale_w = tw * 0.30
        max_scale_h = th * 0.30
        w = tw + min(max(w - tw, -max_scale_w), max_scale_w)
        h = th + min(max(h - th, -max_scale_h), max_scale_h)

        # Exponential smoothing against previous tracked box.
        a = 0.24
        sx = tcx * (1.0 - a) + cx * a
        sy = tcy * (1.0 - a) + cy * a
        sw = tw * (1.0 - a) + w * a
        sh = th * (1.0 - a) + h * a

        nx1 = int(round(sx - sw / 2.0))
        nx2 = int(round(sx + sw / 2.0))
        ny1 = int(round(sy - sh / 2.0))
        ny2 = int(round(sy + sh / 2.0))

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(frame_w - 1, nx2)
        ny2 = min(frame_h - 1, ny2)

        if nx2 <= nx1 or ny2 <= ny1:
            return bbox
        return (nx1, ny1, nx2, ny2)

    def _predict_tracked_bbox(self, frame_w: int, frame_h: int) -> Optional[BBox]:
        if self._track_bbox is None or self._track_center is None or self._track_misses > 12:
            return None

        tx1, ty1, tx2, ty2 = self._track_bbox
        w = max(1.0, float(tx2 - tx1))
        h = max(1.0, float(ty2 - ty1))

        cx = self._track_center[0] + self._track_velocity[0]
        cy = self._track_center[1] + self._track_velocity[1]

        x1 = int(round(cx - w / 2.0))
        x2 = int(round(cx + w / 2.0))
        y1 = int(round(cy - h / 2.0))
        y2 = int(round(cy + h / 2.0))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w - 1, x2)
        y2 = min(frame_h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _on_tracking_miss(self) -> None:
        self._track_misses += 1
        self._track_velocity = (self._track_velocity[0] * 0.8, self._track_velocity[1] * 0.8)
        if self._track_misses > 20:
            self._track_center = None
            self._track_velocity = (0.0, 0.0)
            self._track_bbox = None
        if self._track_misses > 45:
            self._couple_tracks = [None, None]
            self._couple_initialized = False
            self._pair_dist_ref = None

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
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))

        for i, (x, y, w, h) in enumerate(rects):
            score = float(weights[i]) if i < len(weights) else 1.0
            if score >= 0.25:
                x1, y1, x2, y2 = (x, y, x + w, y + h)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                dist = float(np.hypot(center_x - frame_cx, center_y - frame_cy)) / frame_diag
                area = float(max(1, (x2 - x1) * (y2 - y1))) / float(max(1, frame_w * frame_h))
                central_score = dist * 2.0 - score * 0.2 - area * 0.2
                selected.append((central_score, dist, (x1, y1, x2, y2)))

        if not selected:
            return None

        central_selected = [it for it in selected if it[1] <= 0.36]
        pool = central_selected if central_selected else selected
        pool.sort(key=lambda it: it[0])

        best = pool[0][2]
        x1, y1, x2, y2 = best

        # Merge with one nearby partner when available.
        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        for _, _, candidate in pool[1:]:
            cx = (candidate[0] + candidate[2]) / 2.0
            cy = (candidate[1] + candidate[3]) / 2.0
            center_dist = float(np.hypot(cx - bx, cy - by)) / frame_diag
            if center_dist <= 0.30:
                x1 = min(x1, candidate[0])
                y1 = min(y1, candidate[1])
                x2 = max(x2, candidate[2])
                y2 = max(y2, candidate[3])
                break

        x1 = max(0, x1)
        y1 = max(0, y1)
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

        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))

        scored = []
        for contour in large:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            dist = float(np.hypot(center_x - frame_cx, center_y - frame_cy)) / frame_diag
            area_ratio = float(max(1, w * h)) / float(max(1, frame_w * frame_h))
            score = dist * 2.0 - area_ratio * 0.35
            scored.append((score, dist, (x, y, w, h)))

        central_scored = [it for it in scored if it[1] <= 0.38]
        pool = central_scored if central_scored else scored
        pool.sort(key=lambda it: it[0])

        selected = [pool[0][2]]
        best_x, best_y, best_w, best_h = pool[0][2]
        best_cx = best_x + best_w / 2.0
        best_cy = best_y + best_h / 2.0

        for _, _, (x, y, w, h) in pool[1:]:
            cx = x + w / 2.0
            cy = y + h / 2.0
            pair_dist = float(np.hypot(cx - best_cx, cy - best_cy)) / frame_diag
            if pair_dist <= 0.30:
                selected.append((x, y, w, h))
                break

        x1 = frame_w
        y1 = frame_h
        x2 = 0
        y2 = 0
        for x, y, w, h in selected:
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

    def _classify_gender_from_landmarks(self, landmarks) -> tuple:
        """
        Classify gender using shoulder-to-hip ratio from MediaPipe keypoints.
        Returns (gender, confidence) where gender is "male", "female", or None.
        
        Keypoint indices:
        - Shoulders: 11 (right), 12 (left)
        - Hips: 23 (right), 24 (left)
        """
        try:
            # Extract shoulder and hip landmarks
            if len(landmarks) < 25:
                return None, 0.0

            r_shoulder = landmarks[11]  # Right shoulder
            l_shoulder = landmarks[12]  # Left shoulder
            r_hip = landmarks[23]       # Right hip
            l_hip = landmarks[24]       # Left hip

            # Check visibility
            min_visibility = 0.3
            if any(lm.visibility < min_visibility for lm in [r_shoulder, l_shoulder, r_hip, l_hip]):
                return None, 0.0

            # Calculate widths
            shoulder_width = abs(r_shoulder.x - l_shoulder.x)
            hip_width = abs(r_hip.x - l_hip.x)

            if shoulder_width < 0.01 or hip_width < 0.01:
                return None, 0.0

            ratio = shoulder_width / hip_width

            # Classify based on ratio
            # Males: broader shoulders (ratio > 1.05)
            # Females: broader hips (ratio < 0.95)
            if ratio > 1.08:
                confidence = min(1.0, (ratio - 1.08) / 0.2)
                return "male", confidence
            elif ratio < 0.92:
                confidence = min(1.0, (0.92 - ratio) / 0.15)
                return "female", confidence
            else:
                # Uncertain range
                return None, 0.0

        except Exception:  # noqa: BLE001
            return None, 0.0

    def _extract_gender_for_bbox(self, frame_bgr: np.ndarray, bbox: BBox) -> tuple:
        """Extract gender classification for a bounding box using MediaPipe."""
        pose_model = self._gender_pose if self._gender_pose is not None else self._pose
        if pose_model is None:
            return None, 0.0

        try:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                return None, 0.0

            if (x2 - x1) < 24 or (y2 - y1) < 24:
                return None, 0.0

            # Crop the region
            crop = frame_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = pose_model.process(rgb)

            if result.pose_landmarks:
                return self._classify_gender_from_landmarks(result.pose_landmarks.landmark)

            return None, 0.0
        except Exception:  # noqa: BLE001
            return None, 0.0
