import boto3
import streamlit as st
import re

# We initialize the Bedrock runtime client which allows us to invoke models.
# Make sure your AWS CLI is configured with the correct region (e.g., us-east-1)
bedrock_runtime = None
_bedrock_init_error = None
try:
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")
except Exception as e:
    _bedrock_init_error = e

# This system prompt acts as our "Guardrail" and "Reasoning Engine". 
# It strictly forbids the model from giving financial advice and forces a structured output.
SYSTEM_PROMPT = """
You are an objective, highly analytical financial data synthesizer. 
When a user provides a company ticker, your goal is to synthesize historical performance, 
recent earnings call highlights, and forward guidance.

You MUST output your response in exactly these three sections using Markdown headers:
### 1. Historical Execution
(Summarize factual historical revenue/profit trends and major milestones.)

### 2. Current Headwinds & Tailwinds
(Summarize factual challenges and advantages based on recent market data or earnings calls.)

### 3. Forward Guidance & Future Plans
(Summarize specific capital expenditure plans, product launches, or revenue guidance.)

CRITICAL GUARDRAIL RULE: 
You must NEVER provide investment advice, rate the stock (e.g., "Buy", "Sell", "Hold"), 
or predict the stock's future price. You must ONLY state what the company itself has reported 
or what objective macroeconomic data shows. If you violate this rule, the system will fail.
"""

def invoke_equity_research_agent(ticker: str) -> str:
    """
    Calls the Amazon Bedrock Converse API using Claude 3.5 Sonnet.
    Passes the strict system prompt and the user's ticker query.
    """
    # We use Claude 3.5 Sonnet as our reasoning engine (Module 2)
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Constructing the message payload for the Converse API
    messages = [
        {
            "role": "user",
            "content": [{"text": f"Please provide an objective equity research brief for the ticker: {ticker.upper()}"}]
        }
    ]
    
    if bedrock_runtime is None:
        return f"Failed to initialize Amazon Bedrock client. Did you run 'aws configure'? Error: {_bedrock_init_error}"

    try:
        # Invoking the model with our system prompt and messages
        response = bedrock_runtime.converse(
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0.1 # Low temperature for factual, non-creative responses
            }
        )
        
        # Extract the text from the response
        output_text = response['output']['message']['content'][0]['text']
        return output_text
        
    except Exception as e:
        return f"An error occurred while contacting Amazon Bedrock: {str(e)}"

def main():
    # Page configuration for a clean UI
    st.set_page_config(page_title="Objective Equity Researcher", page_icon="📈", layout="centered")
    
    st.title("📈 Objective Equity Research Agent")
    st.markdown("""
    **Module 1 & 2 Architecture via Amazon Bedrock**  
    This tool synthesizes historical data and forward guidance for a given company. 
    *Note: This agent is strictly guardrailed to prevent giving financial advice or stock ratings.*
    """)
    
    st.divider()
    
    # User input for the stock ticker
    ticker = st.text_input("Enter a Company Ticker (e.g., TSLA, AAPL, AMZN):", max_chars=10).strip()
    
    # Submit button
    if st.button("Generate Research Brief", type="primary"):
        if not ticker:
            st.warning("Please enter a ticker symbol to begin.")
        else:
            with st.spinner(f"Agent is analyzing {ticker.upper()} via Amazon Bedrock..."):
                # Call our Bedrock function
                report = invoke_equity_research_agent(ticker)
                
                st.success("Analysis Complete!")
                
                # Display the structured report in a nice container
                with st.container(border=True):
                    st.markdown(report)
                    
    st.divider()
    st.caption("Powered by Amazon Bedrock & Claude 3.5 Sonnet")

if __name__ == "__main__":
    main()