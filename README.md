# Dance Video Resizer

Resize portrait dance videos into YouTube-friendly 16:9 landscape videos while keeping dancers as large as possible in frame.

## Features

- Reframes to 16:9 while preserving source content.
- Uses ML-based keyframe detection (YOLO pose/person, fallback to MediaPipe/HOG) to keep the front dance couple in view.
- Updates framing every 3 seconds and applies smooth transitions.
- Adds a dark, enlarged background fill (full frame) when side padding is needed.
- Optionally merges original audio back with FFmpeg.
- Handles edge cases like already-landscape videos and tiny resolutions.

## Requirements

- Python 3.10+
- FFmpeg available in PATH (for audio merge)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Compiling Manual

This project is a Python CLI script (no binary build step), but you can "compile"
it by validating all modules with Python bytecode compilation.

1. Create and activate environment (Windows + conda):

```bash
conda create -n dance-video-resizer python=3.11 -y
conda activate dance-video-resizer
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Compile-check all source files:

```bash
python -m compileall dance_video_resizer
```

4. Run a smoke test (short dry-run):

```bash
python -m dance_video_resizer.cli \
  --input "SPH - prelims - slow.mp4" \
  --output output_smoke_test.mp4 \
  --dry-run \
  --dry-run-seconds 12 \
  --segment-seconds 3.0 \
  --transition-seconds 2.5 \
  --margin-ratio 0.12 \
  --background-darken 0.6 \
  --progress-interval 1.0
```

5. Run full export:

```bash
python -m dance_video_resizer.cli \
  --input "SPH - prelims - slow.mp4" \
  --output output_full.mp4 \
  --segment-seconds 3.0 \
  --transition-seconds 2.5 \
  --margin-ratio 0.12 \
  --background-darken 0.6 \
  --progress-interval 1.0
```

If your FFmpeg in conda is unstable on Windows, pass a known working path:

```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --ffmpeg-path "C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe"
```

## Usage

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_full.mp4 --segment-seconds 3.0 --transition-seconds 2.5 --margin-ratio 0.12 --background-darken 0.6 --progress-interval 1.0
```

Recommended dry-run:

```bash
python -m dance_video_resizer.cli \
  --input "SPH - prelims - slow.mp4" \
  --output output_dry_run.mp4 \
  --dry-run \
  --dry-run-seconds 12 \
  --segment-seconds 3.0 \
  --transition-seconds 2.5 \
  --margin-ratio 0.12 \
  --background-darken 0.6 \
  --progress-interval 1.0
```

Recommended full run:

```bash
python -m dance_video_resizer.cli \
  --input "SPH - prelims - slow.mp4" \
  --output output_full.mp4 \
  --segment-seconds 3.0 \
  --transition-seconds 2.5 \
  --margin-ratio 0.12 \
  --background-darken 0.6 \
  --progress-interval 1.0
```

## Notes

- If FFmpeg is not available, the app still writes a processed silent video and logs a warning.
- MediaPipe may miss detections on some frames; the app falls back to center framing.
- Progress logs include estimated time remaining (ETA) when total frame count is known.
