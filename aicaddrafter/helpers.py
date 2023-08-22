def get_entity(handle: str, drawing):
    entities = [
        ent for ent in drawing.entities._entities if ent.handle == handle
    ]
    if not entities:
        return None
    return entities[0]


def get_block(handle: str, drawing):
    blocks = [
        block for block in drawing.blocks._blocks.values() if block.handle == handle
    ]
    if not blocks:
        return None
    return blocks[0]
