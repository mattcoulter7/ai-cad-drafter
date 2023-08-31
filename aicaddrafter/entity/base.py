import typing as T
from shapely import LineString


class BaseEntityExtractor(object):
    type: str

    def __init__(self, type: str) -> None:
        self.type = type

    def filter_entities(self, entities) -> T.List[any]:
        return [ent for ent in entities if ent.dxftype == self.type]

    def extract_lines(self, entities) -> T.List[LineString]:
        raise NotImplementedError()
