from langgraph.prebuilt import create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o")
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True
)

def distance_planner(start_location: str, end_location: str):
    """Get the distance in minutes to drive between two cities."""
    agent = create_react_agent(model=model, tools=[search_tool])
    return agent.invoke({"messages": [HumanMessage(content=f"How long is it to drive from {start_location} to {end_location}?")]}).content

def route_planner(location: str, distance: int):
    """Find routes in a specific city."""
    agent = create_react_agent(model=model, tools=[search_tool])
    return agent.invoke({"messages": [HumanMessage(content=f"Please find some routes in {location} that are around {distance} miles long")]}).content

def bike_planner(location: str, drive_distance: int = 60, bike_distance: int = 30):
    """Plan a bike route."""
    agent = create_react_agent(model=model, tools=[distance_planner, route_planner])
    return agent.invoke({"messages": [HumanMessage(content=f"Please plan a bike route for me to take in and around {location}. I am willing to drive {drive_distance} minutes and I want the route to be roughly {bike_distance} minutes.")]})['messages'][-1].content