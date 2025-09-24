import json
from typing import Dict, List
from config import DATA_PATH

def load_books() -> List[Dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_summary_by_title(title: str) -> str:
    """
    Return the full summary for an EXACT title (case-insensitive).
    Raises ValueError if not found.
    """
    data = load_books()
    idx = {b["title"].strip().lower(): b for b in data}
    key = (title or "").strip().lower()
    if key not in idx:
        raise ValueError(f"Title not found: {title}")
    return idx[key]["summary"]
