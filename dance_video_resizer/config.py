from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProcessingConfig:
    target_width: int = 1920
    target_height: int = 1080
    segment_seconds: float = 3.0
    transition_seconds: float = 1.5
    margin_ratio: float = 0.15
    background_darken: float = 0.25
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    ffmpeg_path: str = "ffmpeg"
    dry_run_seconds: Optional[float] = None
    progress_interval_seconds: float = 2.0
    gender_focus: Optional[str] = None  # "male", "female", or None
    portrait_rectangular_crop: bool = False
    portrait_foreground_aspect: float = 1.0


PRESET_SOCIAL_PORTRAIT = "social-portrait"
PRESET_YOUTUBE_LANDSCAPE = "youtube-landscape"
SUPPORTED_PRESETS = (PRESET_SOCIAL_PORTRAIT, PRESET_YOUTUBE_LANDSCAPE)


def build_preset_config(preset_name: str) -> ProcessingConfig:
    if preset_name == PRESET_SOCIAL_PORTRAIT:
        return ProcessingConfig(
            target_width=1080,
            target_height=1920,
            segment_seconds=1.5,
            transition_seconds=1.0,
            margin_ratio=0.18,
            background_darken=0.0,
            progress_interval_seconds=1.0,
        )

    if preset_name == PRESET_YOUTUBE_LANDSCAPE:
        return ProcessingConfig(
            target_width=1920,
            target_height=1080,
            segment_seconds=3.0,
            transition_seconds=2.5,
            margin_ratio=0.12,
            background_darken=0.6,
            progress_interval_seconds=1.0,
        )

    raise ValueError(f"Unsupported preset: {preset_name}")
