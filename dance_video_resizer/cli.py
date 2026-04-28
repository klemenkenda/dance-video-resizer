from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

from .config import (
    PRESET_YOUTUBE_LANDSCAPE,
    SUPPORTED_PRESETS,
    ProcessingConfig,
    build_preset_config,
)
from .ffmpeg_utils import resolve_ffmpeg


def _load_workspace_env() -> None:
    if load_dotenv is None:
        return

    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _default_ffmpeg_path() -> str:
    return os.getenv("DANCE_VIDEO_RESIZER_FFMPEG_PATH") or os.getenv("FFMPEG_PATH") or "ffmpeg"


def _print_ffmpeg_resolution(ffmpeg_path: str) -> int:
    resolved_ffmpeg, resolution_source = resolve_ffmpeg(ffmpeg_path)
    if resolved_ffmpeg:
        print(f"FFmpeg resolved: {resolved_ffmpeg}")
        print(f"FFmpeg source: {resolution_source}")
        return 0

    print(f"FFmpeg not found for configured value: {ffmpeg_path}")
    return 1


def _log_startup(config: ProcessingConfig, preset: str, debug_detection_box: bool) -> None:
    mode = "detection-overlay" if debug_detection_box else "reframe"
    print(
        "Starting processing: "
        f"mode={mode}, preset={preset}, target={config.target_width}x{config.target_height}, "
        f"segment={config.segment_seconds:.2f}s, transition={config.transition_seconds:.2f}s, "
        f"margin={config.margin_ratio:.2f}, background_darken={config.background_darken:.2f}, "
        f"portrait_rect_crop={config.portrait_rectangular_crop}, "
        f"portrait_fg_aspect={config.portrait_foreground_aspect:.3f}"
    )

    resolved_ffmpeg, resolution_source = resolve_ffmpeg(config.ffmpeg_path)
    if resolved_ffmpeg:
        print(f"FFmpeg startup resolution: {resolved_ffmpeg} (source: {resolution_source})")
    else:
        print(f"FFmpeg startup resolution: not found for '{config.ffmpeg_path}'. Audio merge may be skipped.")


def parse_args() -> argparse.Namespace:
    _load_workspace_env()

    parser = argparse.ArgumentParser(description="Reframe dance videos to a target aspect ratio with couple-aware tracking.")
    parser.add_argument("--input", help="Path to input video file.")
    parser.add_argument("--output", help="Path to output video file.")
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default=PRESET_YOUTUBE_LANDSCAPE, help="Named export preset.")
    parser.add_argument("--target-width", type=int, help="Output width. Overrides preset when set.")
    parser.add_argument("--target-height", type=int, help="Output height. Overrides preset when set.")
    parser.add_argument("--segment-seconds", type=float, help="How often to refresh dancer framing. Overrides preset when set.")
    parser.add_argument("--transition-seconds", type=float, help="Duration in seconds to smoothly transition between zoom levels. Overrides preset when set.")
    parser.add_argument("--margin-ratio", type=float, help="Horizontal margin around dancer bbox. Overrides preset when set.")
    parser.add_argument("--background-darken", type=float, help="Darkening strength for blurred background fill. Overrides preset when set.")
    parser.add_argument("--gender-focus", choices=["male", "female"], help="Focus on male or female dancer. Default is auto (both equally).")
    parser.add_argument(
        "--portrait-rectangular-crop",
        action="store_true",
        help="For portrait outputs, keep a rectangular foreground crop and fill top/bottom with darkened zoomed background.",
    )
    parser.add_argument(
        "--portrait-foreground-aspect",
        type=float,
        help="Foreground crop aspect ratio when --portrait-rectangular-crop is enabled. Default: 1.0 (square).",
    )
    parser.add_argument("--debug-detection-box", action="store_true", help="Write non-resized output with front-couple detection boxes.")
    parser.add_argument("--check-ffmpeg", action="store_true", help="Resolve the active FFmpeg binary, print it, and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Process only a short clip for quick testing.")
    parser.add_argument("--dry-run-seconds", type=float, default=15.0, help="Duration for dry-run mode in seconds.")
    parser.add_argument("--progress-interval", type=float, help="How often progress and ETA are printed, in seconds. Overrides preset when set.")
    parser.add_argument(
        "--ffmpeg-path",
        default=_default_ffmpeg_path(),
        help="FFmpeg executable path. Overrides DANCE_VIDEO_RESIZER_FFMPEG_PATH or FFMPEG_PATH from .env when set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check_ffmpeg:
        return _print_ffmpeg_resolution(args.ffmpeg_path)

    if not args.input:
        print("Error: --input is required unless --check-ffmpeg is used.")
        return 1

    if not args.output:
        print("Error: --output is required unless --check-ffmpeg is used.")
        return 1

    preset_config = build_preset_config(args.preset)

    target_width = args.target_width if args.target_width is not None else preset_config.target_width
    target_height = args.target_height if args.target_height is not None else preset_config.target_height
    segment_seconds = args.segment_seconds if args.segment_seconds is not None else preset_config.segment_seconds
    transition_seconds = args.transition_seconds if args.transition_seconds is not None else preset_config.transition_seconds
    margin_ratio = args.margin_ratio if args.margin_ratio is not None else preset_config.margin_ratio
    background_darken = args.background_darken if args.background_darken is not None else preset_config.background_darken
    progress_interval = args.progress_interval if args.progress_interval is not None else preset_config.progress_interval_seconds

    if target_width <= 0 or target_height <= 0:
        print("Error: target dimensions must be positive.")
        return 1

    if segment_seconds <= 0:
        print("Error: segment-seconds must be positive.")
        return 1

    if transition_seconds <= 0:
        print("Error: transition-seconds must be positive.")
        return 1

    if margin_ratio < 0:
        print("Error: margin-ratio must be non-negative.")
        return 1

    if not 0.0 <= background_darken <= 1.0:
        print("Error: background-darken must be between 0 and 1.")
        return 1

    if args.dry_run_seconds <= 0:
        print("Error: dry-run-seconds must be positive.")
        return 1

    if progress_interval <= 0:
        print("Error: progress-interval must be positive.")
        return 1

    portrait_foreground_aspect = 1.0 if args.portrait_foreground_aspect is None else args.portrait_foreground_aspect
    if portrait_foreground_aspect <= 0:
        print("Error: portrait-foreground-aspect must be positive.")
        return 1

    config = ProcessingConfig(
        target_width=target_width,
        target_height=target_height,
        segment_seconds=segment_seconds,
        transition_seconds=transition_seconds,
        margin_ratio=margin_ratio,
        background_darken=background_darken,
        ffmpeg_path=args.ffmpeg_path,
        dry_run_seconds=args.dry_run_seconds if args.dry_run else None,
        progress_interval_seconds=progress_interval,
        gender_focus=args.gender_focus,
        portrait_rectangular_crop=args.portrait_rectangular_crop,
        portrait_foreground_aspect=portrait_foreground_aspect,
    )

    _log_startup(config, args.preset, args.debug_detection_box)

    try:
        from .pipeline import process_detection_overlay_video, process_video

        if args.debug_detection_box:
            process_detection_overlay_video(args.input, args.output, config)
        else:
            process_video(args.input, args.output, config)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
