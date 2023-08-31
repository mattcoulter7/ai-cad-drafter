import typing as T
from .base import BaseEntityExtractor

from shapely import LineString


class LineExtractor(BaseEntityExtractor):
    def __init__(self) -> None:
        super().__init__(
            type="LINE"
        )

    def extract_lines(self, entities) -> T.List[LineString]:
        entities = self.filter_entities(entities)
        return [
            LineString((
                (ent.start[0], ent.start[1]),
                (ent.end[0], ent.end[1])
            )) for ent in entities
        ]
