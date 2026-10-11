"""Small grammar helpers for the generated change descriptions (the names come from open-vocabulary
models: "chair", "luggage and bags", "office chair", ...)."""
from __future__ import annotations

NOT_PLURAL = ("ss", "us", "is", "ics")       # glass, cactus, chassis, electronics are singular


def plural(label: str) -> bool:
    return " and " in label or (label.endswith("s") and not label.endswith(NOT_PLURAL))


def was(label: str) -> str:
    return "were" if plural(label) else "was"


def is_(label: str) -> str:
    return "are" if plural(label) else "is"


def article(label: str) -> str:
    """'A chair', 'An umbrella', 'Luggage and bags' (no article for plurals)."""
    if plural(label):
        return label[:1].upper() + label[1:]
    return f"{'An' if label[:1].lower() in 'aeiou' else 'A'} {label}"


def head(label: str) -> str:
    return label.split()[-1] if label else label


def without_self(relations: list[str], label: str) -> list[str]:
    """Drop relations to the object itself under another name ("chair" next to "office chair":
    segmentation often splits one object into nodes that are named differently)."""
    return [r for r in relations if head(r) != head(label)]


def it(label: str) -> str:
    return "they" if plural(label) else "it"


def It(label: str) -> str:  # noqa: N802 - sentence start
    return it(label).capitalize()


def stands(label: str) -> str:
    return "stand" if plural(label) else "stands"
