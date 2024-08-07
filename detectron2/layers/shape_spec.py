from dataclasses import dataclass
from typing import Optional

@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None