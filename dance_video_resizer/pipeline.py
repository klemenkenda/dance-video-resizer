from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import cv2

from .config import ProcessingConfig
from .detector import DancerDetector
from .ffmpeg_utils import merge_audio, resolve_ffmpeg_path
from .reframer import FrameGeometry, Reframer


def _merge_or_move_audio(
    input_path: str,
    silent_output: str,
    output_path: str,
    ffmpeg_path: str,
) -> None:
    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    resolved_ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    if resolved_ffmpeg:
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
        print("Warning: No working FFmpeg binary found. Writing processed video without audio.")
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        analysis = detector.detect(frame)
        boxed = frame.copy()
        if analysis.bbox is not None:
            x1, y1, x2, y2 = analysis.bbox
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (20, 240, 20), 3)
            cv2.putText(
                boxed,
                "front-couple",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 240, 20),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                boxed,
                "front-couple: not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 240),
                2,
                cv2.LINE_AA,
            )

        writer.write(boxed)

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
    )

    detector = DancerDetector(
        min_detection_confidence=config.pose_min_detection_confidence,
        min_tracking_confidence=config.pose_min_tracking_confidence,
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

    segment_size = max(1, int(round(config.segment_seconds * fps)))
    current_state = reframer.initial_state()
    target_state = current_state

    frame_idx = 0
    start_time = time.perf_counter()
    progress_step = int(max(1.0, config.progress_interval_seconds * fps))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % segment_size == 0:
            analysis = detector.detect(frame)
            if analysis.bbox is not None:
                target_state = reframer.compute_target_state(analysis.bbox)

        current_state = reframer.smooth_state(current_state, target_state)
        out = reframer.render_frame(frame, current_state)
        writer.write(out)

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
