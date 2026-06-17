import re
import ollama
from datetime import datetime

_BOOKING_KEYWORDS = [
    "availab",  # matches available / availability / availabilities / etc.
    "book", "slot", "appointment", "free on", "clean on",
    "schedule", "reserve", "open on", "any slot",
    "how about", "what about", "clean at", "cleaning at", "hour clean",
    "hours clean", "session at", "time slot", "cleaning session",
    "what time", "which time", "what slots", "when can",
]

_NULL_RESPONSES = {"null", "none", "no", "n/a", "not a booking", "not applicable"}

_TIME_RE = re.compile(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b', re.IGNORECASE)
_TIME_24H_RE = re.compile(r'\b(\d{1,2}:\d{2})\b')
_DURATION_WORD_MAP = {"one": 1, "two": 2, "three": 3, "four": 4, "half": 0.5}
_DURATION_RE = re.compile(
    r'\b(one|two|three|four|half|\d+(?:\.\d+)?)\s*(?:-\s*)?(hour|hr)s?\b',
    re.IGNORECASE,
)


def _extract_time(text: str) -> str | None:
    m = _TIME_RE.search(text)
    if m:
        raw = m.group(1).strip().replace(" ", "").upper()  # e.g. "6:30PM", "5PM"
        for fmt in ("%I:%M%p", "%I%p"):
            try:
                return datetime.strptime(raw, fmt).strftime("%H:%M")
            except ValueError:
                pass
    m = _TIME_24H_RE.search(text)
    if m:
        return m.group(1)
    return None


def _extract_duration(text: str) -> float | None:
    m = _DURATION_RE.search(text)
    if not m:
        return None
    word = m.group(1).lower()
    if word in _DURATION_WORD_MAP:
        return float(_DURATION_WORD_MAP[word])
    try:
        return float(word)
    except ValueError:
        return None


def _extract_date_via_llm(message: str, today: str, context_block: str) -> str | None:
    prompt = f"""Today is {today}.
{context_block}
Extract the date from the latest message. Use the conversation above if the date is implied (e.g. "same day" or a follow-up question).
Return ONLY the date as YYYY-MM-DD. If no date can be determined, return null.

Latest message: {message}"""
    try:
        result = ollama.chat(
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        raw = result["message"]["content"].strip()
        if raw.lower().rstrip(".!") in _NULL_RESPONSES:
            return None
        m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', raw)
        return m.group(1) if m else None
    except Exception:
        return None


def _date_from_history(conversation_history: list[dict]) -> str | None:
    """Find the most recent date from stored booking intents or ISO dates in content."""
    for msg in reversed(conversation_history or []):
        # Prefer the exact ISO date stored alongside the bot's booking reply
        stored = msg.get("booking_intent") or {}
        if stored.get("date"):
            return stored["date"]
        m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', msg.get("content", ""))
        if m:
            return m.group(1)
    return None


def _duration_from_history(conversation_history: list[dict]) -> float | None:
    """Find the most recent duration from stored booking intents or text."""
    for msg in reversed(conversation_history or []):
        stored = msg.get("booking_intent") or {}
        if stored.get("duration_hours"):
            return stored["duration_hours"]
        dur = _extract_duration(msg.get("content", ""))
        if dur is not None:
            return dur
    return None


def extract_booking_intent(
    message: str, today: str, conversation_history: list[dict] | None = None
) -> dict | None:
    msg_lower = message.lower()
    if not any(kw in msg_lower for kw in _BOOKING_KEYWORDS):
        return None

    context_block = ""
    if conversation_history:
        recent = [m for m in conversation_history[-6:] if m.get("role") in ("user", "assistant")]
        if recent:
            lines = [
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in recent
            ]
            context_block = "Recent conversation:\n" + "\n".join(lines)

    # Regex extraction (reliable for time and duration)
    time_str = _extract_time(message)
    duration = _extract_duration(message)

    # LLM extraction for date (handles natural language)
    date_str = _extract_date_via_llm(message, today, context_block)

    # Inherit missing fields from conversation history
    if date_str is None:
        date_str = _date_from_history(conversation_history)
    if time_str is None:
        time_str = _extract_time(context_block)
    if duration is None:
        duration = _duration_from_history(conversation_history)

    if date_str is None:
        return None

    # time_str and duration may be None — caller handles those cases
    return {"date": date_str, "start_time": time_str, "duration_hours": duration}
