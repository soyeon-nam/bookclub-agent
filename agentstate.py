from typing import TypedDict


class BookAgentState(TypedDict, total=False):
    messages: list[str]
    book: str
    book_summary: str
    participants_name: list[str]
    user_name: str
    topics_history: list[str]
    max_messages: int
    _session: dict
    _routing: dict
    next_speaker: str
    turns_since_president: int
    force_question: bool
    last_audio_path: str
