# rag_core.py
from typing import List, Dict, Any
import json
from openai import OpenAI

from ingest import ensure_index
from tools import get_summary_by_title

client = OpenAI()  # expects OPENAI_API_KEY in env

SYSTEM_RECOMMENDER = (
    "You are Smart Librarian. You receive the user's preferences and a small list of "
    "CANDIDATES (title + snippet) retrieved from a local collection. "
    "Rules:\n"
    "- Recommend exactly ONE to THREE titles from CANDIDATES. Do NOT invent titles.\n"
    "- Briefly state 1–2 reasons aligned with the request.\n"
    "- If there aren't any matches, or the match is weak, request clarification(e.g. category, period, vibe) and suggest the title of the book that matches the description the best.\n"
    "- After deciding, CALL the function get_summary_by_title with the exact chosen title.\n"
    "- Keep the tone concise and helpful."
)

def _score_from_distance(dist: float | None) -> float:
    # Chroma uses cosine space; distance in [0,2]. A simple normalized similarity:
    # If distance is None, return 0.0; otherwise map 0 -> 1.0 (best), ~1 -> ~0.0
    if dist is None:
        return 0.0
    # Many setups return cosine distance in [0,2]; clamp and convert to similarity-ish score.
    d = max(0.0, min(2.0, float(dist)))
    return round(1.0 - d/2.0, 3)

def retrieve_candidates(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Runs semantic search in Chroma and returns:
    [
      {"title": "The Hobbit", "document": "Title: ...\nSummary: ...", "score": 0.71},
      ...
    ]
    """
    col = ensure_index()
    res = col.query(query_texts=[query], n_results=top_k)
    items: List[Dict[str, Any]] = []

    docs      = (res.get("documents") or [[]])[0]
    metas     = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        doc  = docs[i]
        dist = distances[i] if i < len(distances) else None
        items.append({
            "title": meta.get("title", ""),
            "document": doc,
            "score": _score_from_distance(dist),
        })
    return items

def llm_recommend_and_call_tool(user_query: str, candidates: List[Dict[str, str]]) -> str:
    """
    1) Give the model the user query + retrieved candidates.
    2) Model MUST pick exactly one title and call get_summary_by_title(title).
    3) We execute the tool call and ask the model to compose the final answer (reason + full summary).
    """
    # Prepare compact candidate block for grounding (titles + short snippet)
    def _short(doc: str, n=400) -> str:
        return doc[:n] + ("..." if len(doc) > n else "")

    candidates_block = "\n".join([
        f"- {c['title']}: {_short(c['document'])}" for c in candidates
        if c.get("title")
    ]) or "NO CANDIDATES"

    tools = [{
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Return the FULL summary for the exact book title (case-insensitive).",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    }]

    messages = [
        {"role": "system", "content": SYSTEM_RECOMMENDER},
        {"role": "user", "content": json.dumps({
            "query": user_query,
            "candidates": [{"title": c["title"], "snippet": _short(c["document"]), "score": c.get("score", 0.0)} for c in candidates]
        }, ensure_ascii=False)},
        {"role": "user", "content": f"CANDIDATES (verbatim context):\n{candidates_block}"},
    ]

    # First pass: let the model decide and (ideally) call the tool
    first = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        tools=tools,
        tool_choice="auto",  # if debugging, you can flip to "required"
        messages=messages,
    )
    choice = first.choices[0]
    msg = choice.message

    # If the model issued a tool call, run it and do the second pass to compose the final reply
    if getattr(msg, "tool_calls", None):
        tool_msgs = []
        for tc in msg.tool_calls:
            if tc.type == "function" and tc.function and tc.function.name == "get_summary_by_title":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                title = (args.get("title") or "").strip()
                try:
                    summary = get_summary_by_title(title)
                except Exception as e:
                    summary = f"ERROR: {e}"
                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": summary,
                })

        # Second pass: provide the assistant tool-call message + the tool result(s)
        second = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=messages + [
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                },
                *tool_msgs,
            ],
        )
        return second.choices[0].message.content.strip()

    # No tool call? Be strict: ask again forcing the tool call (rare edge case).
    retry = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        tools=tools,
        tool_choice="required",
        messages=messages,
    )
    rmsg = retry.choices[0].message
    if getattr(rmsg, "tool_calls", None):
        # Re-run same execution path once
        tool_msgs = []
        for tc in rmsg.tool_calls:
            if tc.type == "function" and tc.function and tc.function.name == "get_summary_by_title":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                title = (args.get("title") or "").strip()
                try:
                    summary = get_summary_by_title(title)
                except Exception as e:
                    summary = f"ERROR: {e}"
                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": summary,
                })

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=messages + [
                {
                    "role": "assistant",
                    "content": rmsg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in rmsg.tool_calls
                    ],
                },
                *tool_msgs,
            ],
        )
        return final.choices[0].message.content.strip()

    # Still nothing: fall back to the first content instead of crashing
    return (msg.content or "Sorry, I couldn't produce a recommendation.").strip()
