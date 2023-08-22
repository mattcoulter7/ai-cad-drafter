import typing as T
from .base import BaseEntityExtractor, Line


class LineExtractor(BaseEntityExtractor):
    def __init__(self) -> None:
        super().__init__(
            type="LINE"
        )

    def extract_lines(self, entities) -> T.List[Line]:
        entities = self.filter_entities(entities)
        return [
            Line(
                x1=ent.start[0],
                y1=ent.start[1],
                x2=ent.end[0],
                y2=ent.end[1],
            ) for ent in entities
        ]
