from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

try:
    import imageio_ffmpeg
except Exception:  # noqa: BLE001
    imageio_ffmpeg = None


FfmpegResolution = Tuple[Optional[str], Optional[str]]


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


def _iter_conda_roots() -> Iterator[Path]:
    seen = set()

    for raw_path in (os.getenv("CONDA_PREFIX"), sys.prefix):
        if not raw_path:
            continue

        path = Path(raw_path)
        normalized = str(path).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        yield path

    user_profile = os.getenv("USERPROFILE")
    if not user_profile:
        return

    for suffix in ("miniconda3", "anaconda3"):
        root = Path(user_profile) / suffix
        normalized = str(root).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        yield root


def _iter_common_ffmpeg_candidates() -> Iterator[Tuple[str, str]]:
    seen = set()

    def emit(path: Path, source: str) -> Iterator[Tuple[str, str]]:
        normalized = str(path).lower()
        if normalized in seen:
            return
        seen.add(normalized)
        yield (str(path), source)

    for root in _iter_conda_roots():
        for relative in (
            Path("Library/bin/ffmpeg.exe"),
            Path("Scripts/ffmpeg.exe"),
            Path("bin/ffmpeg"),
        ):
            yield from emit(root / relative, f"conda:{root}")

        envs_dir = root / "envs"
        if envs_dir.is_dir():
            for env_dir in envs_dir.iterdir():
                if not env_dir.is_dir():
                    continue
                yield from emit(env_dir / "Library/bin/ffmpeg.exe", f"conda-env:{env_dir.name}")

    program_files = [os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)"), os.getenv("LocalAppData")]
    for base in program_files:
        if not base:
            continue

        base_path = Path(base)
        for relative in (
            Path("ffmpeg/bin/ffmpeg.exe"),
            Path("FFmpeg/bin/ffmpeg.exe"),
            Path("Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-*/bin/ffmpeg.exe"),
            Path("Microsoft/WinGet/Links/ffmpeg.exe"),
        ):
            pattern = relative.as_posix()
            if "*" in pattern:
                for matched in sorted(base_path.glob(pattern)):
                    yield from emit(matched, f"windows:{matched.parent}")
            else:
                yield from emit(base_path / relative, f"windows:{base_path}")


def resolve_ffmpeg(ffmpeg_path: str = "ffmpeg") -> FfmpegResolution:
    candidates = []

    if ffmpeg_path:
        candidates.append((ffmpeg_path, "configured"))

        which = shutil.which(ffmpeg_path)
        if which:
            candidates.append((which, "PATH"))

    candidates.extend(_iter_common_ffmpeg_candidates())

    if imageio_ffmpeg is not None:
        try:
            candidates.append((imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg"))
        except Exception:  # noqa: BLE001
            pass

    seen = set()
    for candidate, source in candidates:
        if not candidate:
            continue

        normalized = str(candidate).lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        if Path(candidate).exists() or shutil.which(candidate):
            if _is_working_ffmpeg(candidate):
                return candidate, source

    return None, None


def is_ffmpeg_available(ffmpeg_path: str = "ffmpeg") -> bool:
    resolved_ffmpeg, _ = resolve_ffmpeg(ffmpeg_path)
    return resolved_ffmpeg is not None


def resolve_ffmpeg_path(ffmpeg_path: str = "ffmpeg") -> Optional[str]:
    resolved_ffmpeg, _ = resolve_ffmpeg(ffmpeg_path)
    return resolved_ffmpeg


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
