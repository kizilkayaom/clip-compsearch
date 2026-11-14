import json
from dataclasses import dataclass
from pathlib import Path

CAPTION_PREPROCESSING = "collapse whitespace runs to one space and strip the ends"


@dataclass(frozen=True)
class Caption:
    caption_id: int
    image_id: int
    text: str


def load_captions(path: Path) -> list[Caption]:
    with path.open() as f:
        annotations = json.load(f)["annotations"]
    captions = [
        Caption(entry["id"], entry["image_id"], " ".join(entry["caption"].split()))
        for entry in annotations
    ]
    return sorted(captions, key=lambda caption: (caption.image_id, caption.caption_id))
