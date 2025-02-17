from typing import Dict

from utils.schemas import Events
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from state import State
from langgraph.types import interrupt
from langgraph.store.memory import InMemoryStore
from langchain_openai import ChatOpenAI
from langmem import create_memory_store_manager
from tools import luma_scraper, bike_planner

model = ChatOpenAI(model="gpt-4o")

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
) 

memory_manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories",),  # 
)

async def ask_schedule(state: State) -> Dict:
    """Ask user for their schedule passively."""
    # Send an email in real life
    weekend_schedule = interrupt(
        {
            "question": "What is your schedule for this weekend? What do you want to do?"
        }
    )
    # Save any preferences to memory
    await memory_manager.ainvoke(state['messages']+[HumanMessage(content=weekend_schedule)])
    return {"messages": [HumanMessage(content=weekend_schedule)]}

async def plan_weekend(state: State) -> Dict:
    """Process user's schedule response and save to memory."""
    schedule = state["messages"][-1].content
    agent = create_react_agent(model=model, tools=[luma_scraper, bike_planner], response_mode=Events)
    response = agent.invoke(schedule)
    
    return {"events": response.events}
    
async def confirm_with_user(state: State) -> Dict:
    """Ask user to confirm the planned schedule."""
    # Send an email in real life
    confirm = interrupt(
        {
            "question": "Is this what you want to do this weekend?"
        }
    )
    return {"reflect": confirm}

def decide_whether_to_reflect(state: State) -> str:
    if state['refect']:
        return "plan_weekend"
    else:
        return "__end__"


# Create the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("ask_schedule", ask_schedule)
graph_builder.add_node("plan_weekend", plan_weekend)
graph_builder.add_node("confirm_with_user", confirm_with_user)

# Add edges
graph_builder.add_edge("__start__", "ask_schedule")
graph_builder.add_edge("ask_schedule", "plan_weekend")
graph_builder.add_edge("plan_weekend", "confirm_with_user")
graph_builder.add_conditional_edges("confirm_with_user", decide_whether_to_reflect)

# Compile graph with interrupts and checkpointing
graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    store=store
)