import typing as T
from .base import BaseEntityExtractor, Line


class LWPolyLineExtractor(BaseEntityExtractor):
    def __init__(self) -> None:
        super().__init__(
            type="LWPOLYLINE"
        )

    def extract_lines(self, entities) -> T.List[Line]:
        entities = self.filter_entities(entities)
        lines = []

        def explode(entity) -> T.List[Line]:
            lines = []
            for i in range(len(entity.points)):
                j = i + 1
                if j > len(entity.points) - 1:
                    j = 0
                lines.append(
                    Line(
                        x1=entity.points[i][0],
                        y1=entity.points[i][1],
                        x2=entity.points[j][0],
                        y2=entity.points[j][1],
                    )
                )
            return lines

        for ent in entities:
            lines.extend(explode(ent))

        return lines
