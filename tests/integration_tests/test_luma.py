import pytest
from src.tools import luma_scraper
from src.utils.schemas import EventBase
from openevals.evaluators.json import create_json_match_evaluator


@pytest.mark.langsmith
def test_luma():
    """Test that the luma tool works accurately."""
    ans = luma_scraper.invoke(input={"tool_call_id": "1"})
    events = ans.update['events']
    reference_events = [
        EventBase(
            name="Built on Bedrock Demo Nights",
            description="Exciting evening of Generative AI demos from local startups leveraging Amazon Bedrock",
            date="2025-02-18",
            location="San Francisco, CA",
            url="https://lu.ma/gqp6ar00"
        )
    ]
    evaluator = create_json_match_evaluator(
        list_aggregator="average", 
        aggregator=None,
        exclude_keys=["url"],
        rubric={
            "location": "Are these the same location?",
            "description": "Does the output description contain all the information mentioned in the reference description?"
        },
        list_match_mode="superset"
    )
    result = evaluator(outputs = [e.model_dump() for e in events], reference_outputs = [e.model_dump() for e in reference_events])
    for r in result:
        assert r['score'] == 1

