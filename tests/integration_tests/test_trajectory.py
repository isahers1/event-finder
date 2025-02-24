from src.agent import graph
import pytest
from langgraph.types import Command
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_simple_trajectory():
    thread_config = {"configurable": {"thread_id": "some_id"}}

    # Sample run through of agent
    await graph.ainvoke({"messages": []}, config=thread_config)
    # First resume
    await graph.ainvoke(Command(resume={
        "answer": "I would like to go on a 50 mile bike ride on Saturday morning. I am willing to drive up to 30 minutes from SF for this ride. I would also like to attend an event Saturday/Sunday related to AI. I cannot do Sunday brunch time as I already have an appointment then."
    }), config=thread_config)
    # Second resume
    await graph.ainvoke(Command(resume=False), config=thread_config)

    history = graph.get_state(thread_config)

    # Create evaluator
    evaluator = create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT, model="openai:o3-mini")
    eval_result = evaluator(
        outputs=history.values,
    )

    # For demo purposes
    print(eval_result)

    assert eval_result['score'] == 1



    
