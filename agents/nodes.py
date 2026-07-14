from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import AgentState
from tools.financial_tools import get_stock_info, get_recent_news
from rag.qdrant_store import get_rag_context

import os

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), temperature=0.2)

def fetcher_node(state: AgentState) -> dict:
    """Agent responsible for gathering all external data."""
    ticker = state["ticker"]
    stock_info = get_stock_info(ticker)
    news = get_recent_news(ticker)
    rag_context = get_rag_context(f"{ticker} future plans and earnings")
    
    raw_data = f"--- STOCK INFO ---\n{stock_info}\n\n--- RECENT NEWS ---\n{news}\n\n--- RAG CONTEXT (10-K/Earnings) ---\n{rag_context}"
    return {"raw_data": raw_data, "iteration": state.get("iteration", 0) + 1}

def analyst_node(state: AgentState) -> dict:
    """Agent responsible for synthesizing data into a structured report."""
    raw_data = state["raw_data"]
    system_prompt = """
    You are an objective financial analyst. Synthesize the provided raw data into exactly these three sections using Markdown headers:
    ### 1. Historical Execution
    ### 2. Current Headwinds & Tailwinds
    ### 3. Forward Guidance & Future Plans
    
    Never provide investment advice.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Raw Data:\n{raw_data}")
    ]
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list) and len(content) > 0:
        content = content[0].get("text", "") if isinstance(content[0], dict) else content[0]
    if not isinstance(content, str):
        content = str(content)
    return {"analysis_report": content}

def compliance_node(state: AgentState) -> dict:
    """Agent responsible for reviewing the report to ensure no investment advice is given."""
    report = state["analysis_report"]
    system_prompt = """
    You are a strict compliance reviewer. Check the following financial report.
    If it contains any investment advice, stock ratings (Buy/Sell/Hold), or price predictions, respond with exactly 'FAIL'.
    If it is purely factual and objective, respond with exactly 'PASS'.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Report:\n{report}")
    ]
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list) and len(content) > 0:
        content = content[0].get("text", "") if isinstance(content[0], dict) else content[0]
    if not isinstance(content, str):
        content = str(content)
    status = content.strip().upper()
    
    # If the model hallucinates extra text, check for the keyword
    if "FAIL" in status:
        return {"compliance_status": "FAIL"}
    return {"compliance_status": "PASS"}
