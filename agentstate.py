from typing import Optional, TypedDict
from langchain_core.messages import AnyMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages

class BookAgentState(TypedDict, total=False):
    messages: list[str]
    book: str
    book_summary: str
    participants_name: list[str]
    user_name: str
    topics_history: list[str]
    max_messages: int
    _session: dict
