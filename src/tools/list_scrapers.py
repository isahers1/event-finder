from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from src.utils.schemas import Events
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from typing_extensions import Annotated

model = ChatOpenAI(model="gpt-4o")

@tool
def luma_scraper(
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Scrape events for the upcoming weekend from Luma."""
    loader = WebBaseLoader("https://lu.ma/sf")
    scraper_model = model.with_structured_output(
        Events,
    )
    
    docs = loader.load()
    response = scraper_model.invoke(f"Please extract the events from the following website: {docs[0].page_content}")
    return Command(
        update={
            "events": response.events,
            "messages": [ToolMessage(content="Succesfully scraped Luma events", tool_call_id=tool_call_id)]
        }
    )