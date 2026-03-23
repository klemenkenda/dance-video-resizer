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
