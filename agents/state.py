from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Represents the state/memory of the graph throughout the research process.
    """
    ticker: str
    raw_data: str
    analysis_report: str
    compliance_status: str
    iteration: int
