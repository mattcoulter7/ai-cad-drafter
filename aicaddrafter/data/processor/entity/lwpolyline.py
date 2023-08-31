import typing as T
from .base import BaseEntityExtractor

from shapely import LineString


class LWPolyLineExtractor(BaseEntityExtractor):
    def __init__(self) -> None:
        super().__init__(
            type="LWPOLYLINE"
        )

    def extract_lines(self, entities) -> T.List[LineString]:
        entities = self.filter_entities(entities)
        lines = []

        def explode(entity) -> T.List[LineString]:
            lines = []
            for i in range(len(entity.points) - 1):
                j = i + 1
                if j > len(entity.points) - 1:
                    j = 0
                lines.append(
                    LineString((
                        (entity.points[i][0], entity.points[i][1]),
                        (entity.points[j][0], entity.points[j][1])
                    ))
                )
            return lines

        for ent in entities:
            lines.extend(explode(ent))

        return lines
