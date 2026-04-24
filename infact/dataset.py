import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    id: str
    track: str
    category: str
    level: int
    source_dataset: str
    video: str
    question: str
    options: list
    answer_idx: int
    order_sensitive: bool = False
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, d):
        return cls(
            id=d["id"],
            track=d["track"],
            category=d["category"],
            level=int(d.get("level", 0)),
            source_dataset=d.get("source_dataset", "unknown"),
            video=d["video"],
            question=d["question"],
            options=list(d["options"]),
            answer_idx=int(d["answer_idx"]),
            order_sensitive=bool(d.get("order_sensitive", False)),
            meta=d.get("meta", {}),
        )


def load_split(root, split):
    path = Path(root) / "questions" / f"{split}.jsonl"
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Item.from_json(json.loads(line)))
    return items
