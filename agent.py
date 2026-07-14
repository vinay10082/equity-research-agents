import streamlit as st
import re
from dotenv import load_dotenv
load_dotenv() # Load environment variables early so imported modules have access to them

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import fetcher_node, analyst_node, compliance_node

def build_graph():
    workflow = StateGraph(AgentState) # type: ignore
    
    workflow.add_node("Fetcher", fetcher_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Compliance", compliance_node)
    
    workflow.set_entry_point("Fetcher")
    workflow.add_edge("Fetcher", "Analyst")
    workflow.add_edge("Analyst", "Compliance")
    
    def router(state: AgentState):
        if state.get("compliance_status") == "FAIL" and state.get("iteration", 0) < 3:
            return "Analyst" # Retry if it failed compliance
        return END

    workflow.add_conditional_edges("Compliance", router, {"Analyst": "Analyst", END: END})
    return workflow.compile()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Compile the graph once at startup
graph = build_graph()

import os

def extract_ticker(user_input: str) -> str:
    """Uses the LLM to extract a stock ticker from a natural language query."""
    try:
        llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), temperature=0.1)
        system_prompt = "You extract stock ticker symbols from user messages. Respond ONLY with the 1-5 letter ticker symbol in uppercase. If no stock is mentioned, respond with 'UNKNOWN'."
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "") if isinstance(content[0], dict) else content[0]
        if not isinstance(content, str):
            content = str(content)
        return content.strip().upper()
    except Exception as e:
        print(f"Extraction error: {e}")
        return "UNKNOWN"

def main():
    st.set_page_config(page_title="Equity Research Agent", page_icon="🕵️", layout="centered")
    
    st.title("Equity Research Agent")
    st.caption("Powered by LangGraph, Qdrant, & Google Gemini 3.1 Pro")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # React to user input
    if prompt := st.chat_input("Ask for an objective research brief on any stock (e.g., 'How is Tesla doing?'):"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant Response block
        with st.chat_message("assistant"):
            
            final_report = None
            
            # Step 1: Extract ticker
            with st.status("Understanding request...", expanded=True) as status:
                st.write("Extracting ticker symbol...")
                ticker = extract_ticker(prompt)
                
                if ticker == "UNKNOWN" or not re.fullmatch(r"[A-Z0-9.\-]{1,10}", ticker):
                    status.update(label="No public ticker identified", state="complete", expanded=False)
                    final_report = "I couldn't identify a valid public stock ticker in your request. Please note that private companies do not have public stock tickers. Try asking about a publicly traded company!"
                else:
                    st.write(f"Identified Ticker: **{ticker}**")
                    status.update(label=f"Agents are analyzing {ticker}...", state="running")
                    
                    # Step 2: Run Multi-Agent Graph Trace
                    initial_state = {"ticker": ticker, "iteration": 0}
                    
                    try:
                        for output in graph.stream(initial_state):
                            for node_name, state_update in output.items():
                                if node_name == "Fetcher":
                                    st.write(f"🕵️ **Fetcher:** Gathered live market data, news, and RAG context.")
                                elif node_name == "Analyst":
                                    st.write(f"🧠 **Analyst:** Synthesized raw data into a structured research report.")
                                    final_report = state_update.get("analysis_report")
                                elif node_name == "Compliance":
                                    compliance_status = state_update.get("compliance_status")
                                    if compliance_status == "PASS":
                                        st.write(f"🛡️ **Compliance Reviewer:** Review Passed! No investment advice detected.")
                                    else:
                                        st.write(f"🛡️ **Compliance Reviewer:** Review Failed! Investment advice detected. Returning to Analyst.")
                        
                        if final_report:
                            status.update(label="Analysis Complete", state="complete", expanded=False)
                        else:
                            status.update(label="Analysis Failed", state="error", expanded=False)
                            final_report = "Agents failed to produce a compliant report."

                    except Exception as e:
                        status.update(label="Error occurred", state="error", expanded=False)
                        final_report = f"An error occurred during graph execution: {str(e)}"

            # Display the final report outside the status block
            if final_report:
                st.markdown(final_report)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_report})

if __name__ == "__main__":
    main()