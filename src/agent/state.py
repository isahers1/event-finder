
from typing import Annotated, List
from utils.schemas import EventBase
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    events: list[EventBase]
    reflect: bool