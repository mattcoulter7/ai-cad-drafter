import typing as T
from shapely import LineString


class BaseObjectExtractor(object):
    layers: T.List[str]

    def __init__(self, layers: T.List[str]) -> None:
        self.layers = layers

    def extract_blocks(self, drawing):
        def block_on_layer(block) -> bool:
            block_entities_on_layer = [ent for ent in block._entities if ent.layer in self.layers]
            return len(block_entities_on_layer) > 0

        blocks = [block for block in drawing.blocks._blocks.values() if block_on_layer(block)]
        return blocks

    def extract_lines(self, entities) -> T.List[LineString]:
        raise NotImplementedError()
