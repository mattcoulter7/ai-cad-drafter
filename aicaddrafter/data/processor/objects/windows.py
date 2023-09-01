import typing as T
import math
from .base import BaseObjectExtractor

from collections import defaultdict
from shapely import Polygon, LineString

from ..entity import (
    LineExtractor,
    LWPolyLineExtractor
)


def group_touching_lines(lines):
    endpoint_to_lines = defaultdict(list)

    # Populate the dictionary with lines grouped by endpoints
    for line in lines:
        endpoints = [line.coords[0], line.coords[-1]]
        for endpoint in endpoints:
            endpoint_to_lines[endpoint].append(line)

    # Keep track of lines that have been assigned to a group
    assigned_lines = set()

    grouped_lines = []

    for line in lines:
        if line in assigned_lines:
            continue

        group = []
        stack = [line]
        assigned_lines.add(line)

        while stack:
            current_line = stack.pop()
            group.append(current_line)

            endpoints = [current_line.coords[0], current_line.coords[-1]]

            # Look for lines sharing the same endpoints and add them to the stack
            for endpoint in endpoints:
                for connected_line in endpoint_to_lines[endpoint]:
                    if connected_line not in assigned_lines:
                        stack.append(connected_line)
                        assigned_lines.add(connected_line)

        grouped_lines.append(group)

    return grouped_lines


def round_line_coordinates(line, precision=0.333):
    rounded_coords = [(math.ceil(x / precision) * precision, round(y / precision) * precision) for x, y in line.coords]
    return LineString(rounded_coords)


def round_lines_coordinates(lines, precision=0.333):
    return [
        round_line_coordinates(line, precision) for line in lines
    ]


class WindowExtractor(BaseObjectExtractor):
    def __init__(self, layers) -> None:
        super().__init__(
            layers=layers
        )

    def extract(self, drawing) -> T.List[Polygon]:
        def calc_window_bounds(drawing, lines) -> Polygon:
            x_values = []
            y_values = []

            for line in lines:
                for x_val, y_val in line.coords:
                    x_values.append(x_val)
                    y_values.append(y_val)

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

        # extract the lines
        entities = self.extract_entities(drawing)
        return [Polygon((*ent.points,ent.points[0])) for ent in entities if ent.dxftype == "LWPOLYLINE"]
        lines = [
            *LWPolyLineExtractor().extract_lines(entities),
            *LineExtractor().extract_lines(entities)
        ]

        lines = round_lines_coordinates(lines)

        # group the lines
        grouped_lines = group_touching_lines(lines)
        polygons = []
        for group in grouped_lines:
            poly = calc_window_bounds(drawing, group)
            if poly:
                polygons.append(poly)

        return polygons
