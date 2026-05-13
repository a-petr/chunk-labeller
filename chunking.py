import re

_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?:\n{2,})"
    r"|(?=\n\s*[-*•]\s)"
)

_BULLET_STRIP = re.compile(r"^[\s\-*•]+")


def split_sentences(text: str) -> list[str]:
    raw = _SENT_SPLIT.split(text or "")
    return [
        s_clean
        for s in raw
        if len((s_clean := _BULLET_STRIP.sub("", s).strip()).split()) >= 5
    ]
