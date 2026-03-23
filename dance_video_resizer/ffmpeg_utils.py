from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    import imageio_ffmpeg
except Exception:  # noqa: BLE001
    imageio_ffmpeg = None


def _is_working_ffmpeg(executable: str) -> bool:
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001
        return False
    return result.returncode == 0


def is_ffmpeg_available(ffmpeg_path: str = "ffmpeg") -> bool:
    return resolve_ffmpeg_path(ffmpeg_path) is not None


def resolve_ffmpeg_path(ffmpeg_path: str = "ffmpeg") -> Optional[str]:
    candidates = []

    if ffmpeg_path:
        candidates.append(ffmpeg_path)

    which = shutil.which(ffmpeg_path)
    if which:
        candidates.append(which)

    if imageio_ffmpeg is not None:
        try:
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:  # noqa: BLE001
            pass

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)

        if Path(candidate).exists() or shutil.which(candidate):
            if _is_working_ffmpeg(candidate):
                return candidate

    return None


def merge_audio(
    ffmpeg_path: str,
    input_video_with_audio: str,
    processed_video_no_audio: str,
    output_path: str,
) -> Tuple[bool, str]:
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        processed_video_no_audio,
        "-i",
        input_video_with_audio,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, ""

    message = (result.stderr or result.stdout or "Unknown FFmpeg error").strip()
    return False, message
