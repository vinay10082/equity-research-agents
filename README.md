# Equity Research Agents

You are an objective, highly analytical financial data synthesizer. When a user provides a company ticker, your goal is to synthesize historical performance, recent earnings call highlights, and forward guidance.

## Overview

This repository contains an **Objective Equity Research Agent** built using Streamlit and Amazon Bedrock (Claude 3.5 Sonnet). The application provides a structured, factual synthesis of a given company's financial performance, headwinds/tailwinds, and forward guidance. It implements strict guardrails to prevent it from providing investment advice, stock ratings, or predicting stock prices.

## Prerequisites

- AWS CLI configured (`aws configure`) with valid credentials and the appropriate region (e.g., `us-east-1`) to access Amazon Bedrock.
- Python packages installed (e.g., `streamlit`, `boto3`).

## How to Run

1. Activate your virtual environment (if using one):
   ```bash
   source .venv/bin/activate
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run equity_research_app.py
   ```
   *(This command starts a local web server, executes the Python script, and automatically opens the interactive web app in your default browser).*

3. Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

*Note: On your first run, Streamlit may ask for an email address to receive onboarding emails. You can leave this field blank and press Enter to skip.*
