import pytest
from src.tools import luma_scraper
from src.utils.schemas import EventBase
from openevals.json import create_json_match_evaluator

@pytest.mark.langsmith
def test_luma():
    """Test that the luma tool works accurately."""
    ans = luma_scraper.invoke(input={"tool_call_id": "1"})
    events = ans.update['events']
    reference_events = [
        EventBase(
            name="In-person - The Politics of Steel: Nippon Steel’s Fight to Acquire US Steel",
            description="Discussion of Nippon Steel's acquisition attempt of US Steel",
            date="2025-02-20",
            location="San Francisco, CA",
            url="https://lu.ma/3eh88ac6"
        ),
         EventBase(
            name="In-person - The Politics of Steel: Nippon Steel’s Fight to Acquire US Steel",
            description="Discussion of Nippon Steel's acquisition attempt of US Steel",
            date="2025-02-20",
            location="San Francisco, CA",
            url="https://lu.ma/3eh88ac6"
        ),
         EventBase(
            name="In-person - The Politics of Steel: Nippon Steel’s Fight to Acquire US Steel",
            description="Discussion of Nippon Steel's acquisition attempt of US Steel",
            date="2025-02-20",
            location="San Francisco, CA",
            url="https://lu.ma/3eh88ac6"
        )

    ]

    evaluator = create_json_match_evaluator(
        # How to aggregate across a list of objects "average", or "all"
        list_aggregator="average", 
        # How to aggregate individual objects "average", "all", or None
        aggregator=None,
        # Which keys to ignore during evaluation
        exclude_keys=["url"],
        # The rubric for the LLM-as-judge to use for keys without exact match
        rubric={
            "location": "Are these the same location?",
            "description": "Does the output description contain all the information mentioned in the reference description?"
        },
        # How to match lists - "superset", "subset", "same_items", or "ordered"
        list_match_mode="superset",
        # The model to use
        model="openai:o3-mini"
    )

    # Pass everything as dictionaries
    result = evaluator(outputs = [e.model_dump() for e in events], reference_outputs = [e.model_dump() for e in reference_events])

    # Just for demo purposes
    print(result)

    # Each key will be returned
    for r in result:
        assert r['score'] == 1

