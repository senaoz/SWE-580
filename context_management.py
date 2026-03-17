import json
from typing import Callable, Dict, List, Optional, Tuple


Message = Dict[str, object]


def estimate_tokens(messages: List[Message]) -> int:
    """Rough token estimate: ~4 characters per token.

    This is intentionally approximate and fast; it helps decide when to compact.
    """
    total_chars = 0
    for message in messages:
        content = message.get("content", "") or ""
        total_chars += len(str(content))

        # Tool call structures can be non-trivial in size; include them in the rough estimate.
        tool_calls = message.get("tool_calls")
        if tool_calls:
            try:
                total_chars += len(json.dumps(tool_calls))
            except Exception:
                total_chars += len(str(tool_calls))

    return total_chars // 4


def _format_message_for_summary(message: Message) -> str:
    role = str(message.get("role", "") or "").upper()
    content = message.get("content", "") or ""

    tool_calls = message.get("tool_calls")
    if tool_calls and role == "ASSISTANT":
        try:
            names = []
            for tc in tool_calls:
                fn = (tc or {}).get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name")
                if name:
                    names.append(str(name))
            if names:
                return f"{role} (tool_calls={', '.join(names)}): {content}".strip()
        except Exception:
            pass

    return f"{role}: {content}".strip()


def _split_for_compaction_by_user_turns(
    history: List[Message], keep_recent_user_turns: int
) -> Tuple[Optional[Message], List[Message], List[Message]]:
    if not history:
        return None, [], []

    system = history[0]
    if keep_recent_user_turns <= 0:
        return system, history[1:], []

    user_turns_found = 0
    recent_start = 1
    for idx in range(len(history) - 1, 0, -1):
        if history[idx].get("role") == "user":
            user_turns_found += 1
            if user_turns_found >= keep_recent_user_turns:
                recent_start = idx
                break

    if user_turns_found < keep_recent_user_turns:
        return system, [], history[1:]

    old = history[1:recent_start]
    recent = history[recent_start:]
    return system, old, recent


def compact_history(
    history: List[Message],
    summarize_chat: Callable[[List[Message]], Message],
    *,
    token_threshold: int = 500,
    keep_recent_user_turns: int = 4,
    summary_system_prompt: str = (
        "Summarize the following conversation into a brief paragraph. Preserve all key facts, "
        "names, preferences, and decisions. Be concise. Do not call tools."
    ),
    debug: bool = True,
) -> List[Message]:
    """Summarize old messages to stay within a token budget.

    Preserves:
      - the original system prompt at history[0]
      - the last `keep_recent_user_turns` user turns (including any tool chatter after them)

    Everything in-between is summarized into a single system message.
    """
    if estimate_tokens(history) < token_threshold:
        return history

    system, old, recent = _split_for_compaction_by_user_turns(history, keep_recent_user_turns)
    if system is None or not old:
        return history

    old_text = "\n".join(_format_message_for_summary(m) for m in old)
    summary_request = [
        {"role": "system", "content": summary_system_prompt},
        {"role": "user", "content": old_text},
    ]

    summary_reply = summarize_chat(summary_request)
    summary_content = str(summary_reply.get("content", "") or "").strip()
    if not summary_content:
        summary_content = "(Summary unavailable — model returned no content.)"

    if debug:
        print(f" [Compacted {len(old)} messages into summary]")

    summary_message: Message = {
        "role": "system",
        "content": f"Summary of earlier conversation: {summary_content}",
    }

    return [system, summary_message] + recent

