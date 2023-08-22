import typing as T


class Line(object):
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class BaseEntityExtractor(object):
    type: str

    def __init__(self, type: str) -> None:
        self.type = type

    def filter_entities(self, entities) -> T.List[any]:
        return [ent for ent in entities if ent.dxftype == self.type]

    def extract_lines(self, entities) -> T.List[Line]:
        raise NotImplementedError()
