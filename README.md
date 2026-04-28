# Dance Video Resizer

Reframe dance videos between portrait and landscape formats while keeping the main couple as large as possible in frame.

The tool supports both directions:

- Portrait to landscape for YouTube-style delivery.
- Landscape to portrait for social-media delivery.

## Features

- Reframes to portrait or landscape target sizes.
- Uses ML-based keyframe detection (YOLO pose/person, fallback to MediaPipe/HOG) to keep the front dance couple in view.
- Updates framing every 3 seconds and applies smooth transitions.
- For landscape-to-portrait exports, crops a moving portrait window that follows the detected couple.
- Optional portrait mode: keep a rectangular foreground crop and fill top/bottom with a darkened, zoomed background.
- Adds a dark, enlarged background fill (full frame) when side padding is needed.
- Optionally merges original audio back with FFmpeg.
- Handles edge cases like already-landscape videos and tiny resolutions.

## Requirements

- Python 3.10+
- FFmpeg available in PATH (for audio merge)
- A Python environment with the packages from `requirements.txt` installed.

Install dependencies:

```bash
pip install -r requirements.txt
```

One-liner:

```bash
pip install -r requirements.txt
```

## FFmpeg Path

The CLI checks FFmpeg in this order:

1. `--ffmpeg-path`
2. `DANCE_VIDEO_RESIZER_FFMPEG_PATH` from `.env` or the process environment
3. `FFMPEG_PATH` from `.env` or the process environment
4. `ffmpeg` from `PATH`
5. Common Windows and Conda install locations
6. The bundled `imageio-ffmpeg` executable if available

During processing, the tool now logs which FFmpeg binary was selected and where it came from.

Check the active FFmpeg resolution directly from the CLI:

```bash
python -m dance_video_resizer.cli --check-ffmpeg
```

One-liner:

```bash
python -m dance_video_resizer.cli --check-ffmpeg
```

Find FFmpeg on Windows PowerShell:

```powershell
Get-Command ffmpeg | Select-Object -ExpandProperty Source
```

One-liner:

```powershell
Get-Command ffmpeg | Select-Object -ExpandProperty Source
```

If `ffmpeg` is not on `PATH`, search common Conda locations:

```powershell
Get-ChildItem "$env:USERPROFILE\miniconda3","$env:USERPROFILE\anaconda3" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 10 -ExpandProperty FullName
```

One-liner:

```powershell
Get-ChildItem "$env:USERPROFILE\miniconda3","$env:USERPROFILE\anaconda3" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 10 -ExpandProperty FullName
```

If you know the current conda env root, this is the usual Windows location pattern:

```text
<conda-env>\Library\bin\ffmpeg.exe
```

Example:

```text
C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe
```

### Save FFmpeg Path In .env

Create a `.env` file in the project root:

```env
DANCE_VIDEO_RESIZER_FFMPEG_PATH=C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe
```

One-liner:

```env
DANCE_VIDEO_RESIZER_FFMPEG_PATH=C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe
```

Then run the CLI without passing `--ffmpeg-path` every time:

```bash
python -m dance_video_resizer.cli --input input.mp4 --output output.mp4 --preset youtube-landscape
```

One-liner:

```bash
python -m dance_video_resizer.cli --input input.mp4 --output output.mp4 --preset youtube-landscape
```

If you want to check a specific configured path without processing a video:

```bash
python -m dance_video_resizer.cli --check-ffmpeg --ffmpeg-path "C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe"
```

One-liner:

```bash
python -m dance_video_resizer.cli --check-ffmpeg --ffmpeg-path "C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe"
```

## Compiling Manual

This project is a Python CLI script (no binary build step), but you can "compile"
it by validating all modules with Python bytecode compilation.

1. Create and activate environment (Windows + conda):

```bash
conda create -n dance-video-resizer python=3.11 -y
conda activate dance-video-resizer
```

One-liner:

```bash
conda create -n dance-video-resizer python=3.11 -y && conda activate dance-video-resizer
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

One-liner:

```bash
python -m pip install -r requirements.txt
```

3. Compile-check all source files:

```bash
python -m compileall dance_video_resizer
```

One-liner:

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

One-liner:

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_smoke_test.mp4 --dry-run --dry-run-seconds 12 --segment-seconds 3.0 --transition-seconds 2.5 --margin-ratio 0.12 --background-darken 0.6 --progress-interval 1.0
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

One-liner:

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_full.mp4 --segment-seconds 3.0 --transition-seconds 2.5 --margin-ratio 0.12 --background-darken 0.6 --progress-interval 1.0
```

If your FFmpeg in conda is unstable on Windows, pass a known working path:

```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --ffmpeg-path "C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe"
```

One-liner:

```bash
python -m dance_video_resizer.cli --input input.mp4 --output output.mp4 --ffmpeg-path "C:/Users/Klemen/miniconda3/envs/dance-video-resizer/Library/bin/ffmpeg.exe"
```

Or store the same path in `.env` and omit the flag.

## Usage

Available presets:

- `social-portrait`: 1080x1920, faster retargeting for portrait social clips.
- `youtube-landscape`: 1920x1080, slower transitions with stronger background fill for landscape delivery.

Preset defaults:

| Preset | Target size | Segment seconds | Transition seconds | Margin ratio | Background darken |
| --- | --- | --- | --- | --- | --- |
| `social-portrait` | `1080x1920` | `1.5` | `1.0` | `0.18` | `0.0` |
| `youtube-landscape` | `1920x1080` | `3.0` | `2.5` | `0.12` | `0.6` |

The CLI defaults to `youtube-landscape` when `--preset` is omitted.

Explicit flags override preset values. For example, `--preset social-portrait --transition-seconds 0.8` keeps the portrait preset but uses a faster transition.

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_full.mp4 --preset youtube-landscape
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_full.mp4 --preset youtube-landscape
```

Portrait export example for social media:

```bash
python -m dance_video_resizer.cli \
  --input "landscape_input.mp4" \
  --output output_portrait.mp4 \
  --preset social-portrait
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "landscape_input.mp4" --output output_portrait.mp4 --preset social-portrait
```

Portrait export with square (1:1) foreground crop plus darkened/zoomed top and bottom fill:

```bash
python -m dance_video_resizer.cli \
  --input "landscape_input.mp4" \
  --output output_portrait_rect_fill.mp4 \
  --preset social-portrait \
  --portrait-rectangular-crop \
  --portrait-foreground-aspect 1.0 \
  --background-darken 0.35
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "landscape_input.mp4" --output output_portrait_rect_fill.mp4 --preset social-portrait --portrait-rectangular-crop --portrait-foreground-aspect 1.0 --background-darken 0.35
```

Note: when portrait rectangular mode is enabled, the default foreground aspect is already 1.0 (square). You can change it with --portrait-foreground-aspect.

Preset values can still be overridden explicitly:

```bash
python -m dance_video_resizer.cli \
  --input "landscape_input.mp4" \
  --output output_portrait_custom.mp4 \
  --preset social-portrait \
  --transition-seconds 0.8 \
  --margin-ratio 0.16
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "landscape_input.mp4" --output output_portrait_custom.mp4 --preset social-portrait --transition-seconds 0.8 --margin-ratio 0.16
```

Recommended dry-run:

```bash
python -m dance_video_resizer.cli \
  --input "SPH - prelims - slow.mp4" \
  --output output_dry_run.mp4 \
  --preset youtube-landscape \
  --dry-run \
  --dry-run-seconds 12 \
  --progress-interval 1.0
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_dry_run.mp4 --preset youtube-landscape --dry-run --dry-run-seconds 12 --progress-interval 1.0
```

Recommended full run:

```bash
python -m dance_video_resizer.cli \
  --input "SPH - prelims - slow.mp4" \
  --output output_full.mp4 \
  --preset youtube-landscape
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "SPH - prelims - slow.mp4" --output output_full.mp4 --preset youtube-landscape
```

Debug front-couple detection without reframing:

```bash
python -m dance_video_resizer.cli \
  --input "input.mp4" \
  --output output_detection_overlay.mp4 \
  --debug-detection-box
```

One-liner:

```bash
python -m dance_video_resizer.cli --input "input.mp4" --output output_detection_overlay.mp4 --debug-detection-box
```

## CLI Reference

### Required Arguments

- `--input <path>`: Path to the input video file. Required unless using `--check-ffmpeg`.
- `--output <path>`: Path to the output video file. Required unless using `--check-ffmpeg`.

### Preset Arguments

- `--preset {social-portrait|youtube-landscape}`: Named export preset. Defaults to `youtube-landscape`. Sets sensible defaults for target size, segment interval, transition speed, margin, and background fill. Can be partially or fully overridden with explicit flags.

### Target Dimension Arguments

- `--target-width <int>`: Output video width in pixels. Overrides preset when set. Must be positive.
- `--target-height <int>`: Output video height in pixels. Overrides preset when set. Must be positive.

Example: `--target-width 1080 --target-height 1920` produces a 1080×1920 portrait output.

### Framing Arguments

- `--segment-seconds <float>`: How often (in seconds) the detector refreshes the target framing. Lower values = more responsive tracking but higher CPU usage. Must be positive. Defaults vary by preset (1.5 for social-portrait, 3.0 for youtube-landscape). For landscape-to-portrait exports, effectively fixed to every frame for better responsiveness.

- `--transition-seconds <float>`: Duration (in seconds) for smooth transitions between framing targets. Higher values = more gradual movements, lower values = snappier tracking. Must be positive. Defaults: 1.0 for social-portrait, 2.5 for youtube-landscape.

- `--margin-ratio <float>`: Extra horizontal margin around the detected couple as a fraction of bbox width. Range: `[0.0, 1.0]`. Higher values give more padding room but reduce couple size. Defaults: 0.18 for social-portrait, 0.12 for youtube-landscape.

- `--background-darken <float>`: Darkening strength for the blurred background fill (only used in landscape mode). Range: `[0.0, 1.0]`. `0.0` = no darkening (original brightness), `1.0` = pure black. Mainly relevant for wider targets. Defaults: 0.0 for social-portrait, 0.6 for youtube-landscape.

Example: `--segment-seconds 2.0 --transition-seconds 1.5 --margin-ratio 0.15 --background-darken 0.5`

### Gender Targeting Arguments

- `--gender-focus {male|female}`: Prioritize tracking a specific gender. Uses pose-based heuristics (shoulder-to-hip width ratio) to classify dancers:
  - `male`: Broader shoulders → ratio > 1.08
  - `female`: Broader hips → ratio < 0.92
  - `None` (omitted): Auto-select without gender preference
  
  When set, the detector boosts the score of candidates matching the focused gender, increasing selection probability. If only one gender is detected, the system still tracks them. Useful when multiple couples are visible and you want to follow a specific dancer.

Example: `--gender-focus female` tracks the female dancer(s) preferentially.

Example: `--gender-focus male` tracks the male dancer(s) preferentially.

### Processing Arguments

- `--dry-run`: Process only a short test clip instead of the entire video. Enables quick feedback loops before running full exports on long files. When set, combined with `--dry-run-seconds` to define the clip length.

- `--dry-run-seconds <float>`: Duration (in seconds) to process when `--dry-run` is enabled. Defaults to `15.0`. Must be positive. Useful range: 8-30 seconds for quick validation.

- `--progress-interval <float>`: Interval (in seconds) for printing progress and ETA updates. Must be positive. Defaults: 1.0 for both presets. Lower values print more frequently.

- `--debug-detection-box`: Render detection boxes on the original video instead of reframing. Shows which regions are detected as the "front couple" without applying zoom/crop transformations. Useful for debugging detection failures or tuning parameters.

- `--ffmpeg-path <path>`: Explicit path to FFmpeg executable. Overrides environment variable lookup. Useful if you have multiple FFmpeg installations or want to use a specific version.

### Diagnostic Arguments

- `--check-ffmpeg`: Resolve and print the active FFmpeg binary path, then exit. Does not process any video. Useful for verifying FFmpeg setup before running a long export. Returns 0 on success, 1 if FFmpeg not found.

Example: `python -m dance_video_resizer.cli --check-ffmpeg`

### Environment Variables

These can be set in `.env` or the system environment to avoid typing `--ffmpeg-path` repeatedly:

- `DANCE_VIDEO_RESIZER_FFMPEG_PATH`: Preferred persistent FFmpeg path. Takes precedence over `FFMPEG_PATH`.
- `FFMPEG_PATH`: Fallback FFmpeg path variable.

### Parameter Defaults and Preset Comparison

| Parameter | Social-Portrait | YouTube-Landscape |
| --- | --- | --- |
| `--target-width` | 1080 | 1920 |
| `--target-height` | 1920 | 1080 |
| `--segment-seconds` | 1.5 | 3.0 |
| `--transition-seconds` | 1.0 | 2.5 |
| `--margin-ratio` | 0.18 | 0.12 |
| `--background-darken` | 0.0 | 0.6 |
| `--progress-interval` | 1.0 | 1.0 |
| `--gender-focus` | None (auto) | None (auto) |

### Common Parameter Combinations

**Fast, responsive portrait tracking:**
```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --preset social-portrait \
  --segment-seconds 1.0 \
  --transition-seconds 0.7 \
  --margin-ratio 0.15
```

**Smooth, gradual landscape tracking with strong background:**
```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --preset youtube-landscape \
  --segment-seconds 4.0 \
  --transition-seconds 3.5 \
  --background-darken 0.75
```

**Female-focused portrait tracking:**
```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --preset social-portrait \
  --gender-focus female
```

**Male-focused landscape tracking with custom timing:**
```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --preset youtube-landscape \
  --gender-focus male \
  --segment-seconds 2.0 \
  --transition-seconds 1.5
```

**Conservative, non-jittery tracking (good for smooth video):**
```bash
python -m dance_video_resizer.cli \
  --input input.mp4 \
  --output output.mp4 \
  --preset social-portrait \
  --segment-seconds 2.0 \
  --transition-seconds 1.5 \
  --margin-ratio 0.20
```

**Quick validation before full export:**
```bash
python -m dance_video_resizer.cli \
  --input "long_video.mp4" \
  --output output_test.mp4 \
  --preset social-portrait \
  --dry-run \
  --dry-run-seconds 15 \
  --progress-interval 1.0
```

### Typical Workflows

1. **Quick validation** (8-15 seconds):
   ```bash
   python -m dance_video_resizer.cli --input input.mp4 --output output_test.mp4 --preset social-portrait --dry-run --dry-run-seconds 12
   ```

2. **Full export with default preset**:
   ```bash
   python -m dance_video_resizer.cli --input input.mp4 --output output.mp4 --preset youtube-landscape
   ```

3. **Custom tracking with gender preference**:
   ```bash
   python -m dance_video_resizer.cli --input input.mp4 --output output.mp4 --preset social-portrait --gender-focus female --transition-seconds 0.8
   ```

4. **Check FFmpeg before processing**:
   ```bash
   python -m dance_video_resizer.cli --check-ffmpeg
   ```

5. **Debug detection on problematic clip**:
   ```bash
   python -m dance_video_resizer.cli --input input.mp4 --output output_debug.mp4 --debug-detection-box --dry-run --dry-run-seconds 10
   ```

## Notes

- If FFmpeg is not available, the app still writes a processed silent video and logs a warning.
- MediaPipe may miss detections on some frames; the app falls back to center framing.
- Progress logs include estimated time remaining (ETA) when total frame count is known.
- Gender classification uses pose heuristics and may have lower accuracy on dancers in unusual poses or at oblique angles.
- For best results with `--gender-focus`, ensure dancers are visible in their full upper bodies (shoulders and hips visible).
