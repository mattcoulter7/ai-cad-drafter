import typing as T
from .base import BaseObjectExtractor

from shapely import Polygon, LineString

from ..entity import (
    LineExtractor,
    LWPolyLineExtractor
)


class WindowExtractor(BaseObjectExtractor):
    def __init__(self, layers) -> None:
        super().__init__(
            layers=layers
        )

    def extract(self, drawing) -> T.List[Polygon]:
        window_blocks = self.extract_blocks(drawing)

        def calc_window_bounds(drawing, block) -> Polygon:
            lines = []
            lines.extend(
                LWPolyLineExtractor().extract_lines(block._entities)
            )

            if not lines:
                return None

            x_values = []
            y_values = []

            for line in lines:
                for point in line.xy:
                    x_values.append(point[0])
                    y_values.append(point[1])

            min_x = min(x_values)
            min_y = min(y_values)
            max_x = max(x_values)
            max_y = max(y_values)

            return Polygon((
                (min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y),
                (min_x, min_y),
            ))

        polygons = []
        for block in window_blocks:
            poly = calc_window_bounds(drawing, block)
            if poly:
                polygons.append(poly)

        return polygons
