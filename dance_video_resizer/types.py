from dataclasses import dataclass
from typing import Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class FrameAnalysis:
    bbox: Optional[BBox]
    gender: Optional[str] = None  # "male", "female", or None
    gender_confidence: float = 0.0
    tracked_points: Optional[list[Tuple[int, int]]] = None


@dataclass
class ReframeState:
    center_x: float
    center_y: float
    crop_height: float
