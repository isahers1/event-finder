from pydantic import BaseModel, Field

class EventBase(BaseModel):
    name: str = Field(..., description="The name of the event")
    description: str = Field(..., description="The description of the event")
    date: str = Field(..., description="The date of the event")
    location: str = Field(..., description="The location of the event")
    url: str = Field(..., description="The URL of the event")

class Events(BaseModel):
    events: list[EventBase] = Field(..., description="A list of events")