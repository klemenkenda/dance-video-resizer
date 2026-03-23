from __future__ import annotations

import argparse
import sys

from .config import ProcessingConfig
from .pipeline import process_detection_overlay_video, process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize dance videos to 16:9 with dancer-aware reframing.")
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", required=True, help="Path to output video file.")
    parser.add_argument("--target-width", type=int, default=1920, help="Output width.")
    parser.add_argument("--target-height", type=int, default=1080, help="Output height.")
    parser.add_argument("--segment-seconds", type=float, default=3.0, help="How often to refresh dancer framing.")
    parser.add_argument("--transition-seconds", type=float, default=1.5, help="Duration in seconds to smoothly transition between zoom levels.")
    parser.add_argument("--margin-ratio", type=float, default=0.15, help="Horizontal margin around dancer bbox.")
    parser.add_argument("--background-darken", type=float, default=0.25, help="Darkening strength for blurred background fill.")
    parser.add_argument("--debug-detection-box", action="store_true", help="Write non-resized output with front-couple detection boxes.")
    parser.add_argument("--dry-run", action="store_true", help="Process only a short clip for quick testing.")
    parser.add_argument("--dry-run-seconds", type=float, default=15.0, help="Duration for dry-run mode in seconds.")
    parser.add_argument("--progress-interval", type=float, default=2.0, help="How often progress and ETA are printed, in seconds.")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="FFmpeg executable path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.target_width <= 0 or args.target_height <= 0:
        print("Error: target dimensions must be positive.")
        return 1

    if args.segment_seconds <= 0:
        print("Error: segment-seconds must be positive.")
        return 1

    if args.dry_run_seconds <= 0:
        print("Error: dry-run-seconds must be positive.")
        return 1

    if args.progress_interval <= 0:
        print("Error: progress-interval must be positive.")
        return 1

    config = ProcessingConfig(
        target_width=args.target_width,
        target_height=args.target_height,
        segment_seconds=args.segment_seconds,
        transition_seconds=args.transition_seconds,
        margin_ratio=args.margin_ratio,
        background_darken=args.background_darken,
        ffmpeg_path=args.ffmpeg_path,
        dry_run_seconds=args.dry_run_seconds if args.dry_run else None,
        progress_interval_seconds=args.progress_interval,
    )

    try:
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
