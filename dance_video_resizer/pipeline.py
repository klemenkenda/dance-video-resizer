from __future__ import annotations

from collections import deque
import shutil
import tempfile
import time
from pathlib import Path

import cv2

from .config import ProcessingConfig
from .detector import DancerDetector
from .ffmpeg_utils import merge_audio, resolve_ffmpeg
from .reframer import FrameGeometry, Reframer


def _merge_or_move_audio(
    input_path: str,
    silent_output: str,
    output_path: str,
    ffmpeg_path: str,
) -> None:
    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    resolved_ffmpeg, resolution_source = resolve_ffmpeg(ffmpeg_path)
    if resolved_ffmpeg:
        print(f"FFmpeg: using '{resolved_ffmpeg}' (source: {resolution_source}).")
        merged_ok, merge_error = merge_audio(
            ffmpeg_path=resolved_ffmpeg,
            input_video_with_audio=input_path,
            processed_video_no_audio=silent_output,
            output_path=output_path,
        )
        if not merged_ok:
            print(f"Warning: Audio merge failed. {merge_error}")
            shutil.move(silent_output, str(final_output))
    else:
        print(f"Warning: No working FFmpeg binary found for '{ffmpeg_path}'. Writing processed video without audio.")
        shutil.move(silent_output, str(final_output))


def process_detection_overlay_video(input_path: str, output_path: str, config: ProcessingConfig) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dry_run_frame_limit = None
    if config.dry_run_seconds is not None:
        dry_run_frame_limit = max(1, int(round(config.dry_run_seconds * fps)))
        print(f"Dry-run enabled: processing up to {config.dry_run_seconds:.1f}s (~{dry_run_frame_limit} frames).")

    effective_total_frames = frame_count
    if dry_run_frame_limit is not None:
        if effective_total_frames > 0:
            effective_total_frames = min(effective_total_frames, dry_run_frame_limit)
        else:
            effective_total_frames = dry_run_frame_limit

    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError("Invalid source video resolution.")

    detector = DancerDetector(
        min_detection_confidence=config.pose_min_detection_confidence,
        min_tracking_confidence=config.pose_min_tracking_confidence,
        gender_focus=config.gender_focus,
    )

    tmp_dir = tempfile.TemporaryDirectory(prefix="dance_detect_")
    silent_output = str(Path(tmp_dir.name) / "detection_overlay_silent.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_output, fourcc, fps, (src_w, src_h))

    if not writer.isOpened():
        cap.release()
        detector.close()
        tmp_dir.cleanup()
        raise RuntimeError("Failed to open video writer.")

    frame_idx = 0
    start_time = time.perf_counter()
    progress_step = int(max(1.0, config.progress_interval_seconds * fps))
    lag_frames = 3
    window = lag_frames * 2 + 1
    frame_buffer = deque()
    last_written_boxes = [None, None]

    geometry = FrameGeometry(
        source_width=src_w,
        source_height=src_h,
        target_width=config.target_width,
        target_height=config.target_height,
    )
    smoothing_alpha = 1.0 - 0.5 ** (1.0 / max(1.0, config.transition_seconds * fps))
    reframer = Reframer(
        geometry=geometry,
        margin_ratio=config.margin_ratio,
        smoothing_alpha=smoothing_alpha,
        background_darken=config.background_darken,
        portrait_rectangular_crop=config.portrait_rectangular_crop,
        portrait_foreground_aspect=config.portrait_foreground_aspect,
    )
    current_state = reframer.initial_state()
    target_state = current_state
    last_valid_subject_box = None

    frame_area = float(max(1, src_w * src_h))
    frame_cx = src_w / 2.0
    frame_cy = src_h / 2.0
    frame_diag = float(max(1.0, (src_w * src_w + src_h * src_h) ** 0.5))

    def _score_box(box: tuple[int, int, int, int], max_height: float) -> float:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5) / frame_diag
        area_ratio = float(max(1, (x2 - x1) * (y2 - y1))) / frame_area
        height = float(max(1, y2 - y1))
        height_ratio = height / float(max(1, src_h))
        min_target_height = max(src_h * 0.22, max_height * 0.62)
        edge_penalty = max(0.0, dist - 0.34) * 2.0
        small_penalty = max(0.0, (min_target_height - height) / float(max(1, src_h))) * 6.0
        return dist * 2.6 - height_ratio * 1.25 - area_ratio * 0.25 + edge_penalty + small_penalty

    def _is_inside_allowed_area(box: tuple[int, int, int, int], max_height: float) -> bool:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5) / frame_diag
        height = float(max(1, y2 - y1))
        min_height = max(src_h * 0.16, max_height * 0.42)
        horizontal_ok = src_w * 0.06 <= cx <= src_w * 0.94
        vertical_ok = src_h * 0.08 <= cy <= src_h * 0.94
        return dist <= 0.43 and height >= min_height and horizontal_ok and vertical_ok

    def _best_fallback_box(
        all_boxes: list[tuple[int, int, int, int]],
        prev_box: tuple[int, int, int, int] | None,
    ):
        if not all_boxes:
            return prev_box

        heights = [max(1, y2 - y1) for (x1, y1, x2, y2) in all_boxes]
        max_height = float(max(heights)) if heights else 1.0
        candidates = [b for b in all_boxes if _is_inside_allowed_area(b, max_height)]
        if not candidates:
            return prev_box

        best = None
        best_score = float("inf")
        for box in candidates:
            score = _score_box(box, max_height)
            if prev_box is not None:
                x1, y1, x2, y2 = box
                px1, py1, px2, py2 = prev_box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                continuity = float(((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5) / frame_diag
                if continuity > 0.10:
                    continue
                score += continuity * 0.9
            if score < best_score:
                best_score = score
                best = box
        return best if best is not None else prev_box

    def _union_boxes(boxes: list[tuple[int, int, int, int] | None]):
        valid = [b for b in boxes if b is not None]
        if not valid:
            return None
        x1 = min(b[0] for b in valid)
        y1 = min(b[1] for b in valid)
        x2 = max(b[2] for b in valid)
        y2 = max(b[3] for b in valid)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _subject_box_is_allowed(box: tuple[int, int, int, int], prev_box: tuple[int, int, int, int] | None) -> bool:
        x1, y1, x2, y2 = box
        bw = float(max(1, x2 - x1))
        bh = float(max(1, y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist_center = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5) / frame_diag

        if not (src_w * 0.07 <= cx <= src_w * 0.93 and src_h * 0.06 <= cy <= src_h * 0.95):
            return False
        if dist_center > 0.46:
            return False
        if bh < src_h * 0.16:
            return False

        if prev_box is not None:
            px1, py1, px2, py2 = prev_box
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            pbw = float(max(1, px2 - px1))
            pbh = float(max(1, py2 - py1))
            jump = float(((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5) / frame_diag
            if jump > 0.10:
                return False

            width_ratio = bw / pbw
            height_ratio = bh / pbh
            if width_ratio < 0.55 or width_ratio > 1.8:
                return False
            if height_ratio < 0.55 or height_ratio > 1.8:
                return False

        return True

    def _state_to_source_crop_box(state):
        if geometry.target_aspect < geometry.source_aspect and not config.portrait_rectangular_crop:
            crop_h = max(1.0, min(float(src_h), state.crop_height))
            crop_w = max(1.0, min(float(src_w), crop_h * geometry.target_aspect))
            crop_h = min(float(src_h), crop_w / geometry.target_aspect)

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
        elif geometry.target_aspect < geometry.source_aspect and config.portrait_rectangular_crop:
            # Use reframer's shared method for consistency
            crop_box = reframer._compute_portrait_rect_crop_box(state, src_w, src_h)
            if crop_box is None:
                return None
            x1, y1, x2, y2 = crop_box
        else:
            # Wider target (portrait-to-landscape): full source width, Y-axis crop only.
            crop_h = max(1.0, min(float(src_h), state.crop_height))
            half_h = crop_h / 2.0
            center_y = min(max(state.center_y, half_h), src_h - half_h)
            x1 = 0
            x2 = src_w
            y1 = max(0, int(round(center_y - half_h)))
            y2 = min(src_h, int(round(center_y + half_h)))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2 - 1, y2 - 1)

    def _expand_crop_box_to_cover_subject(
        crop_box: tuple[int, int, int, int] | None,
        subject_box: tuple[int, int, int, int] | None,
    ):
        if crop_box is None or subject_box is None:
            return crop_box

        cx1, cy1, cx2, cy2 = crop_box
        sx1, sy1, sx2, sy2 = subject_box
        subject_w = max(1, sx2 - sx1 + 1)
        subject_h = max(1, sy2 - sy1 + 1)
        pad_x = max(8, int(round(subject_w * max(0.05, config.margin_ratio * 0.55))))
        pad_y = max(12, int(round(subject_h * max(0.06, config.margin_ratio * 0.75))))

        nx1 = min(cx1, sx1 - pad_x)
        ny1 = min(cy1, sy1 - pad_y)
        nx2 = max(cx2, sx2 + pad_x)
        ny2 = max(cy2, sy2 + pad_y)

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(src_w - 1, nx2)
        ny2 = min(src_h - 1, ny2)
        if nx2 <= nx1 or ny2 <= ny1:
            return crop_box
        return (nx1, ny1, nx2, ny2)

    def _select_central_boxes(person_boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not person_boxes:
            return []

        heights = [max(1, y2 - y1) for (x1, y1, x2, y2) in person_boxes]
        max_height = float(max(heights)) if heights else 1.0

        scored = []
        for idx, box in enumerate(person_boxes):
            score = _score_box(box, max_height)
            scored.append((score, idx))

        scored.sort(key=lambda it: it[0])
        selected = []
        for _, idx in scored[:2]:
            selected.append(person_boxes[idx])
        return selected

    def _smooth_rank_box(rank: int, entries: list[tuple], center_idx: int):
        # Weighted average over a symmetric window to use both past and future frames.
        sigma = 1.5
        wx = wy = ww = wh = wsum = 0.0
        for i, (_, boxes, _, _) in enumerate(entries):
            if rank >= len(boxes):
                continue
            x1, y1, x2, y2 = boxes[rank]
            dx = abs(i - center_idx)
            weight = 1.0 / (1.0 + (dx / sigma) ** 2)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            wx += cx * weight
            wy += cy * weight
            ww += bw * weight
            wh += bh * weight
            wsum += weight

        if wsum <= 0:
            return None

        cx = wx / wsum
        cy = wy / wsum
        bw = max(1.0, ww / wsum)
        bh = max(1.0, wh / wsum)

        prev_box = last_written_boxes[rank]
        if prev_box is not None:
            px1, py1, px2, py2 = prev_box
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            pbw = max(1.0, px2 - px1)
            pbh = max(1.0, py2 - py1)

            max_shift_x = src_w * 0.03
            max_shift_y = src_h * 0.03
            cx = pcx + min(max(cx - pcx, -max_shift_x), max_shift_x)
            cy = pcy + min(max(cy - pcy, -max_shift_y), max_shift_y)

            max_scale_w = pbw * 0.10
            max_scale_h = pbh * 0.10
            bw = pbw + min(max(bw - pbw, -max_scale_w), max_scale_w)
            bh = pbh + min(max(bh - pbh, -max_scale_h), max_scale_h)

        x1 = int(round(cx - bw / 2.0))
        x2 = int(round(cx + bw / 2.0))
        y1 = int(round(cy - bh / 2.0))
        y2 = int(round(cy + bh / 2.0))

        x1 = max(0, min(src_w - 1, x1))
        y1 = max(0, min(src_h - 1, y1))
        x2 = max(0, min(src_w - 1, x2))
        y2 = max(0, min(src_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            candidate = None
        else:
            candidate = (x1, y1, x2, y2)

        current_all_boxes = entries[center_idx][2]
        if current_all_boxes:
            heights = [max(1, by2 - by1) for (_, by1, _, by2) in current_all_boxes]
            current_max_height = float(max(heights)) if heights else 1.0
            if candidate is None or not _is_inside_allowed_area(candidate, current_max_height):
                if prev_box is not None:
                    return prev_box
                return _best_fallback_box(current_all_boxes, prev_box)

        return candidate

    def _render_overlay(frame, smoothed_boxes, person_count, crop_box):
        boxed = frame.copy()
        if smoothed_boxes:
            for rank, box in enumerate(smoothed_boxes, start=1):
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                cv2.rectangle(boxed, (x1, y1), (x2, y2), (30, 30, 240), 3)
                cv2.putText(
                    boxed,
                    f"central-{rank}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 30, 240),
                    2,
                    cv2.LINE_AA,
                )
            if crop_box is not None:
                cx1, cy1, cx2, cy2 = crop_box
                cv2.rectangle(boxed, (cx1, cy1), (cx2, cy2), (240, 120, 30), 2)
                cv2.putText(
                    boxed,
                    "tracked crop",
                    (cx1, max(20, cy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (240, 120, 30),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                boxed,
                f"persons: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 250, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                boxed,
                "central couple: red (temporal smooth)",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                boxed,
                "tracked crop: orange",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 120, 30),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                boxed,
                "persons: none",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 240),
                2,
                cv2.LINE_AA,
            )
        return boxed

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        person_boxes = detector.detect_people_boxes(frame)
        selected = _select_central_boxes(person_boxes)
        frame_buffer.append((frame.copy(), selected, person_boxes, len(person_boxes)))

        if len(frame_buffer) >= window:
            entries = list(frame_buffer)
            center_idx = lag_frames
            smoothed = []
            for rank in range(2):
                sb = _smooth_rank_box(rank, entries, center_idx)
                smoothed.append(sb)
            for rank in range(2):
                last_written_boxes[rank] = smoothed[rank]

            subject_box = _union_boxes(smoothed)
            if subject_box is not None and _subject_box_is_allowed(subject_box, last_valid_subject_box):
                target_state = reframer.compute_target_state(subject_box)
                last_valid_subject_box = subject_box
            else:
                target_state = current_state
            current_state = reframer.smooth_state(current_state, target_state)
            current_state = reframer.ensure_subject_fits(current_state, subject_box)
            crop_box = _state_to_source_crop_box(current_state)
            crop_box = _expand_crop_box_to_cover_subject(crop_box, subject_box)

            out_frame = _render_overlay(entries[center_idx][0], smoothed, entries[center_idx][3], crop_box)
            writer.write(out_frame)
            frame_buffer.popleft()

        frame_idx += 1
        if dry_run_frame_limit is not None and frame_idx >= dry_run_frame_limit:
            break

        if frame_idx % progress_step == 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            speed_fps = frame_idx / elapsed
            if effective_total_frames > 0:
                pct = min(100.0, (frame_idx / effective_total_frames) * 100.0)
                frames_left = max(0, effective_total_frames - frame_idx)
                eta_seconds = frames_left / max(1e-6, speed_fps)
                print(f"Overlay: {pct:.1f}% | speed: {speed_fps:.1f} fps | ETA: {eta_seconds:.1f}s")
            else:
                print(f"Overlay frames: {frame_idx} | speed: {speed_fps:.1f} fps")

    # Flush remaining buffered frames with available context.
    remaining = list(frame_buffer)
    for i in range(len(remaining)):
        left = max(0, i - lag_frames)
        right = min(len(remaining), i + lag_frames + 1)
        local = remaining[left:right]
        center_idx = i - left
        smoothed = []
        for rank in range(2):
            sb = _smooth_rank_box(rank, local, center_idx)
            smoothed.append(sb)
        for rank in range(2):
            last_written_boxes[rank] = smoothed[rank]

        subject_box = _union_boxes(smoothed)
        if subject_box is not None and _subject_box_is_allowed(subject_box, last_valid_subject_box):
            target_state = reframer.compute_target_state(subject_box)
            last_valid_subject_box = subject_box
        else:
            target_state = current_state
        current_state = reframer.smooth_state(current_state, target_state)
        current_state = reframer.ensure_subject_fits(current_state, subject_box)
        crop_box = _state_to_source_crop_box(current_state)
        crop_box = _expand_crop_box_to_cover_subject(crop_box, subject_box)

        out_frame = _render_overlay(remaining[i][0], smoothed, remaining[i][3], crop_box)
        writer.write(out_frame)

    cap.release()
    writer.release()
    detector.close()

    _merge_or_move_audio(
        input_path=input_path,
        silent_output=silent_output,
        output_path=output_path,
        ffmpeg_path=config.ffmpeg_path,
    )

    tmp_dir.cleanup()
    total_frames = max(1, frame_idx)
    duration = total_frames / fps
    total_elapsed = max(1e-6, time.perf_counter() - start_time)
    avg_speed = total_frames / total_elapsed
    print(f"Done overlay. Processed {total_frames} frames (~{duration:.1f}s) at {avg_speed:.1f} fps.")


def process_video(input_path: str, output_path: str, config: ProcessingConfig) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dry_run_frame_limit = None
    if config.dry_run_seconds is not None:
        dry_run_frame_limit = max(1, int(round(config.dry_run_seconds * fps)))
        print(f"Dry-run enabled: processing up to {config.dry_run_seconds:.1f}s (~{dry_run_frame_limit} frames).")

    effective_total_frames = frame_count
    if dry_run_frame_limit is not None:
        if effective_total_frames > 0:
            effective_total_frames = min(effective_total_frames, dry_run_frame_limit)
        else:
            effective_total_frames = dry_run_frame_limit

    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError("Invalid source video resolution.")

    if src_w < 64 or src_h < 64:
        cap.release()
        raise RuntimeError("Source video resolution is too small for robust processing.")

    geometry = FrameGeometry(
        source_width=src_w,
        source_height=src_h,
        target_width=config.target_width,
        target_height=config.target_height,
    )

    # Convert transition_seconds to a per-frame alpha so speed is time-based
    # regardless of fps. After transition_seconds the state is ~50% blended.
    smoothing_alpha = 1.0 - 0.5 ** (1.0 / max(1.0, config.transition_seconds * fps))

    reframer = Reframer(
        geometry=geometry,
        margin_ratio=config.margin_ratio,
        smoothing_alpha=smoothing_alpha,
        background_darken=config.background_darken,
        portrait_rectangular_crop=config.portrait_rectangular_crop,
        portrait_foreground_aspect=config.portrait_foreground_aspect,
    )

    detector = DancerDetector(
        min_detection_confidence=config.pose_min_detection_confidence,
        min_tracking_confidence=config.pose_min_tracking_confidence,
        gender_focus=config.gender_focus,
    )

    tmp_dir = tempfile.TemporaryDirectory(prefix="dance_resize_")
    silent_output = str(Path(tmp_dir.name) / "processed_silent.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_output, fourcc, fps, (config.target_width, config.target_height))

    if not writer.isOpened():
        cap.release()
        detector.close()
        tmp_dir.cleanup()
        raise RuntimeError("Failed to open video writer.")

    lag_frames = 3
    window = lag_frames * 2 + 1
    frame_buffer = deque()
    last_written_boxes = [None, None]
    current_state = reframer.initial_state()
    target_state = current_state
    last_valid_subject_box = None

    frame_idx = 0
    start_time = time.perf_counter()
    progress_step = int(max(1.0, config.progress_interval_seconds * fps))
    frame_cx = src_w / 2.0
    frame_cy = src_h / 2.0
    frame_diag = float(max(1.0, (src_w * src_w + src_h * src_h) ** 0.5))

    frame_area = float(max(1, src_w * src_h))

    def _score_box(box: tuple[int, int, int, int], max_height: float) -> float:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5) / frame_diag
        area_ratio = float(max(1, (x2 - x1) * (y2 - y1))) / frame_area
        height = float(max(1, y2 - y1))
        height_ratio = height / float(max(1, src_h))
        min_target_height = max(src_h * 0.22, max_height * 0.62)
        edge_penalty = max(0.0, dist - 0.34) * 2.0
        small_penalty = max(0.0, (min_target_height - height) / float(max(1, src_h))) * 6.0
        return dist * 2.6 - height_ratio * 1.25 - area_ratio * 0.25 + edge_penalty + small_penalty

    def _is_inside_allowed_area(box: tuple[int, int, int, int], max_height: float) -> bool:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5) / frame_diag
        height = float(max(1, y2 - y1))
        min_height = max(src_h * 0.16, max_height * 0.42)
        horizontal_ok = src_w * 0.06 <= cx <= src_w * 0.94
        vertical_ok = src_h * 0.08 <= cy <= src_h * 0.94
        return dist <= 0.43 and height >= min_height and horizontal_ok and vertical_ok

    def _best_fallback_box(
        all_boxes: list[tuple[int, int, int, int]],
        prev_box: tuple[int, int, int, int] | None,
    ):
        if not all_boxes:
            return prev_box

        heights = [max(1, y2 - y1) for (x1, y1, x2, y2) in all_boxes]
        max_height = float(max(heights)) if heights else 1.0
        candidates = [b for b in all_boxes if _is_inside_allowed_area(b, max_height)]
        if not candidates:
            return prev_box

        best = None
        best_score = float("inf")
        for box in candidates:
            score = _score_box(box, max_height)
            if prev_box is not None:
                x1, y1, x2, y2 = box
                px1, py1, px2, py2 = prev_box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                continuity = float(((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5) / frame_diag
                if continuity > 0.10:
                    continue
                score += continuity * 0.9
            if score < best_score:
                best_score = score
                best = box
        return best if best is not None else prev_box

    def _union_boxes(boxes: list[tuple[int, int, int, int] | None]):
        valid = [b for b in boxes if b is not None]
        if not valid:
            return None
        x1 = min(b[0] for b in valid)
        y1 = min(b[1] for b in valid)
        x2 = max(b[2] for b in valid)
        y2 = max(b[3] for b in valid)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _subject_box_is_allowed(box: tuple[int, int, int, int], prev_box: tuple[int, int, int, int] | None) -> bool:
        x1, y1, x2, y2 = box
        bw = float(max(1, x2 - x1))
        bh = float(max(1, y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist_center = float(((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5) / frame_diag

        if not (src_w * 0.07 <= cx <= src_w * 0.93 and src_h * 0.06 <= cy <= src_h * 0.95):
            return False
        if dist_center > 0.46:
            return False
        if bh < src_h * 0.16:
            return False

        if prev_box is not None:
            px1, py1, px2, py2 = prev_box
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            pbw = float(max(1, px2 - px1))
            pbh = float(max(1, py2 - py1))
            jump = float(((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5) / frame_diag
            if jump > 0.10:
                return False

            width_ratio = bw / pbw
            height_ratio = bh / pbh
            if width_ratio < 0.55 or width_ratio > 1.8:
                return False
            if height_ratio < 0.55 or height_ratio > 1.8:
                return False

        return True

    def _select_central_boxes(person_boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not person_boxes:
            return []

        heights = [max(1, y2 - y1) for (x1, y1, x2, y2) in person_boxes]
        max_height = float(max(heights)) if heights else 1.0

        scored = []
        for idx, box in enumerate(person_boxes):
            score = _score_box(box, max_height)
            scored.append((score, idx))

        scored.sort(key=lambda it: it[0])
        selected = []
        for _, idx in scored[:2]:
            selected.append(person_boxes[idx])
        return selected

    def _smooth_rank_box(rank: int, entries: list[tuple], center_idx: int):
        sigma = 1.5
        wx = wy = ww = wh = wsum = 0.0
        for i, (_, boxes, _, _) in enumerate(entries):
            if rank >= len(boxes):
                continue
            x1, y1, x2, y2 = boxes[rank]
            dx = abs(i - center_idx)
            weight = 1.0 / (1.0 + (dx / sigma) ** 2)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            wx += cx * weight
            wy += cy * weight
            ww += bw * weight
            wh += bh * weight
            wsum += weight

        if wsum <= 0:
            return None

        cx = wx / wsum
        cy = wy / wsum
        bw = max(1.0, ww / wsum)
        bh = max(1.0, wh / wsum)

        prev_box = last_written_boxes[rank]
        if prev_box is not None:
            px1, py1, px2, py2 = prev_box
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            pbw = max(1.0, px2 - px1)
            pbh = max(1.0, py2 - py1)

            max_shift_x = src_w * 0.03
            max_shift_y = src_h * 0.03
            cx = pcx + min(max(cx - pcx, -max_shift_x), max_shift_x)
            cy = pcy + min(max(cy - pcy, -max_shift_y), max_shift_y)

            max_scale_w = pbw * 0.10
            max_scale_h = pbh * 0.10
            bw = pbw + min(max(bw - pbw, -max_scale_w), max_scale_w)
            bh = pbh + min(max(bh - pbh, -max_scale_h), max_scale_h)

        x1 = int(round(cx - bw / 2.0))
        x2 = int(round(cx + bw / 2.0))
        y1 = int(round(cy - bh / 2.0))
        y2 = int(round(cy + bh / 2.0))

        x1 = max(0, min(src_w - 1, x1))
        y1 = max(0, min(src_h - 1, y1))
        x2 = max(0, min(src_w - 1, x2))
        y2 = max(0, min(src_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            candidate = None
        else:
            candidate = (x1, y1, x2, y2)

        current_all_boxes = entries[center_idx][2]
        if current_all_boxes:
            heights = [max(1, by2 - by1) for (_, by1, _, by2) in current_all_boxes]
            current_max_height = float(max(heights)) if heights else 1.0
            if candidate is None or not _is_inside_allowed_area(candidate, current_max_height):
                if prev_box is not None:
                    return prev_box
                return _best_fallback_box(current_all_boxes, prev_box)

        return candidate

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        person_boxes = detector.detect_people_boxes(frame)
        selected = _select_central_boxes(person_boxes)
        frame_buffer.append((frame.copy(), selected, person_boxes, len(person_boxes)))

        if len(frame_buffer) >= window:
            entries = list(frame_buffer)
            center_idx = lag_frames
            smoothed = []
            for rank in range(2):
                sb = _smooth_rank_box(rank, entries, center_idx)
                smoothed.append(sb)
            for rank in range(2):
                last_written_boxes[rank] = smoothed[rank]

            subject_box = _union_boxes(smoothed)
            if subject_box is not None and _subject_box_is_allowed(subject_box, last_valid_subject_box):
                target_state = reframer.compute_target_state(subject_box)
                last_valid_subject_box = subject_box
            else:
                target_state = current_state
            current_state = reframer.smooth_state(current_state, target_state)
            current_state = reframer.ensure_subject_fits(current_state, subject_box)

            out = reframer.render_frame(entries[center_idx][0], current_state)
            writer.write(out)
            frame_buffer.popleft()

        frame_idx += 1
        if dry_run_frame_limit is not None and frame_idx >= dry_run_frame_limit:
            break

        if frame_idx % progress_step == 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            speed_fps = frame_idx / elapsed
            if effective_total_frames > 0:
                pct = min(100.0, (frame_idx / effective_total_frames) * 100.0)
                frames_left = max(0, effective_total_frames - frame_idx)
                eta_seconds = frames_left / max(1e-6, speed_fps)
                print(f"Processing: {pct:.1f}% | speed: {speed_fps:.1f} fps | ETA: {eta_seconds:.1f}s")
            else:
                print(f"Processing frames: {frame_idx} | speed: {speed_fps:.1f} fps")

    # Flush remaining buffered frames with available context.
    remaining = list(frame_buffer)
    for i in range(len(remaining)):
        left = max(0, i - lag_frames)
        right = min(len(remaining), i + lag_frames + 1)
        local = remaining[left:right]
        center_idx = i - left
        smoothed = []
        for rank in range(2):
            sb = _smooth_rank_box(rank, local, center_idx)
            smoothed.append(sb)
        for rank in range(2):
            last_written_boxes[rank] = smoothed[rank]

        subject_box = _union_boxes(smoothed)
        if subject_box is not None and _subject_box_is_allowed(subject_box, last_valid_subject_box):
            target_state = reframer.compute_target_state(subject_box)
            last_valid_subject_box = subject_box
        else:
            target_state = current_state
        current_state = reframer.smooth_state(current_state, target_state)
        current_state = reframer.ensure_subject_fits(current_state, subject_box)

        out = reframer.render_frame(remaining[i][0], current_state)
        writer.write(out)

    cap.release()
    writer.release()
    detector.close()

    _merge_or_move_audio(
        input_path=input_path,
        silent_output=silent_output,
        output_path=output_path,
        ffmpeg_path=config.ffmpeg_path,
    )

    tmp_dir.cleanup()
    total_frames = max(1, frame_idx)
    duration = total_frames / fps
    total_elapsed = max(1e-6, time.perf_counter() - start_time)
    avg_speed = total_frames / total_elapsed
    print(f"Done. Processed {total_frames} frames (~{duration:.1f}s) at {avg_speed:.1f} fps.")
