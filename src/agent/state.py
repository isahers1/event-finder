
from typing import Annotated, List, Optional, TypedDict
from src.utils.schemas import EventBase, Events
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict, total=False):
    messages: Annotated[Optional[List[BaseMessage]], add_messages]
    events: Optional[List[EventBase]]
    reflect: Optional[bool]

class CustomReactState(TypedDict):
    events: str
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    structured_response: Events