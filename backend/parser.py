import re

COMMON_MAP = {
    "tmr": "tomorrow",
    "idk": "I do not know",
    "lmk": "let me know",
    "asap": "as soon as possible",
    "haven": "have not",
    "i tot": "I thought",
    "prob": "probably",
    "wru": "where are you",
    "u": "you",
    "ur": "your",
    "pls": "please",
}

FILLERS = {"lah", "leh", "lor", "ah", "bro", "sia"}


def preprocess_text(text: str) -> str:
    """Expand common abbreviations but keep raw text otherwise."""
    result = text
    # Sort by length descending so multi-word phrases match first
    for slang, expansion in sorted(COMMON_MAP.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(slang) + r'\b', re.IGNORECASE)
        result = pattern.sub(expansion, result)
    return result


def parse_chat(raw_chat: str, members: list[str]) -> list[dict]:
    """
    Split chat into messages list.
    Detect 'Name: message' pattern line-by-line.
    If no speaker detected, assign 'Unknown'.
    Returns list of {speaker, raw_text, timestamp}.
    """
    messages = []
    lines = raw_chat.strip().split("\n")

    # Build a pattern to match known members or generic "Name:" prefix
    member_pattern = re.compile(r'^([A-Za-z][A-Za-z0-9\s]{0,20}):\s*(.+)$')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = member_pattern.match(line)
        if match:
            speaker = match.group(1).strip()
            raw_text = match.group(2).strip()
        else:
            speaker = "Unknown"
            raw_text = line

        messages.append({
            "speaker": speaker,
            "raw_text": raw_text,
            "timestamp": None,
        })

    return messages
