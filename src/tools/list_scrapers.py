from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from utils.schemas import Events
from langgraph.types import Command
from lanchain_core.messages import ToolMessage

model = ChatOpenAI(model="gpt-4o")

def luma_scraper():
    """Scrape events for the upcoming weekend from Luma."""
    loader = WebBaseLoader("https://lu.ma/sf")
    scraper_model = model.with_structured_output(
        Events,
    )

    docs = loader.load()
    response = scraper_model.invoke(f"Please extract the events from the following website: {docs[0].page_content}")
    return Command(
        {
            "events": response.content,
            "messages": [ToolMessage(content=response.content)]
        }
    )