import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun

def get_stock_info(ticker: str) -> str:
    """Fetch current stock price and basic info using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
        market_cap = info.get("marketCap", "N/A")
        sector = info.get("sector", "N/A")
        return f"Ticker: {ticker}\nCurrent Price: ${current_price}\nMarket Cap: {market_cap}\nSector: {sector}"
    except Exception as e:
        return f"Error fetching stock info: {str(e)}"

def get_recent_news(ticker: str) -> str:
    """Fetch recent news for a given ticker."""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(f"{ticker} stock news")
    except Exception as e:
        return f"Error fetching news: {str(e)}"
