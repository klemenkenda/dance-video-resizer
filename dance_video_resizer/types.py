from dataclasses import dataclass
from typing import Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class FrameAnalysis:
    bbox: Optional[BBox]


@dataclass
class ReframeState:
    center_x: float
    center_y: float
    crop_height: float
