from typing import TypedDict, Annotated, List
import operator

class GraphState(TypedDict):
    original_question: str
    hop_count: int
    current_query: str
    
    # Raw context from vector database
    retrieved_paragraphs: List[str]
    
    # Reducer: Appends new facts across multiple hops instead of overwriting
    selected_facts: Annotated[List[str], operator.add]
    
    final_answer: str