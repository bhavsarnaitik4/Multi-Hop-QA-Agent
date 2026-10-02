from pydantic import BaseModel, Field
from typing import List

class SearchQuery(BaseModel):
    query: str = Field(description="The exact search string to query the database. Keep it concise, 3-5 words.")

class FinalAnswer(BaseModel):
    answer: str = Field(description="The final synthesized answer to the user's question.")
    citations: List[str] = Field(description="The exact string sentences from the provided facts that you used to construct the answer.")

class RouteDecision(BaseModel):
    is_complete: bool = Field(description="True if the facts answer ALL parts of the original question, False if information is missing.")