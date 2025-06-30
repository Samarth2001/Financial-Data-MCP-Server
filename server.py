#!/usr/bin/env python3
"""
Financial Data MCP Server
Provides tools for accessing stock prices, options data, and calculating financial metrics
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
from functools import wraps
import sys

import yfinance as yf
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("financial-data-server")

# Simple in-memory cache
cache: Dict[str, tuple[Any, float]] = {}
CACHE_DURATION = 300  # 5 minutes

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 0.5  # 500ms between requests


def rate_limit(func):
    """Simple rate limiting decorator"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global last_request_time
        current_time = time.time()
        time_since_last = current_time - last_request_time

        if time_since_last < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - time_since_last)

        last_request_time = time.time()
        return await func(*args, **kwargs)

    return wrapper


def get_cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate cache key from function name and arguments"""
    return f"{func_name}:{str(args)}:{str(kwargs)}"


def cached(duration: int = CACHE_DURATION):
    """Cache decorator with configurable duration"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = get_cache_key(func.__name__, *args, **kwargs)

            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < duration:
                    return result

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            return result

        return wrapper

    return decorator


# Stock Data Tools


@mcp.tool()
@rate_limit
@cached(duration=60)  # Cache for 1 minute
async def get_stock_price(symbol: str) -> dict:
    """Get current stock price and basic info.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        Dictionary with price data and basic info
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        return {
            "symbol": symbol.upper(),
            "current_price": info.get(
                "currentPrice", info.get("regularMarketPrice", "N/A")
            ),
            "previous_close": info.get("previousClose", "N/A"),
            "day_high": info.get("dayHigh", "N/A"),
            "day_low": info.get("dayLow", "N/A"),
            "volume": info.get("volume", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "name": info.get("longName", symbol.upper()),
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}


@mcp.tool()
@rate_limit
@cached(duration=300)  # Cache for 5 minutes
async def get_historical_data(
    symbol: str, period: str = "1mo", interval: str = "1d"
) -> dict:
    """Get historical price data for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        Dictionary with historical price data
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period, interval=interval)

        if hist.empty:
            return {"error": f"No historical data available for {symbol}"}

        # Convert to simple dictionary format
        data = []
        for date, row in hist.iterrows():
            data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                }
            )

        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": data,
        }
    except Exception as e:
        return {"error": f"Failed to fetch historical data for {symbol}: {str(e)}"}


@mcp.tool()
@rate_limit
@cached(duration=300)
async def get_options_chain(symbol: str, expiration_date: Optional[str] = None) -> dict:
    """Get options chain data for a stock.

    Args:
        symbol: Stock ticker symbol
        expiration_date: Optional expiration date (YYYY-MM-DD format). If not provided, uses nearest expiration.

    Returns:
        Dictionary with calls and puts data
    """
    try:
        ticker = yf.Ticker(symbol.upper())

        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            return {"error": f"No options data available for {symbol}"}

        # Use provided date or first available
        if expiration_date:
            if expiration_date not in expirations:
                return {
                    "error": f"Expiration date {expiration_date} not available",
                    "available_dates": list(expirations),
                }
            exp_date = expiration_date
        else:
            exp_date = expirations[0]

        # Get options data
        opt = ticker.option_chain(exp_date)

        # Format calls data
        calls = []
        for _, row in opt.calls.head(20).iterrows():  # Limit to top 20
            calls.append(
                {
                    "strike": float(row["strike"]),
                    "last_price": (
                        float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0
                    ),
                    "bid": float(row["bid"]) if pd.notna(row["bid"]) else 0,
                    "ask": float(row["ask"]) if pd.notna(row["ask"]) else 0,
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    "open_interest": (
                        int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0
                    ),
                    "implied_volatility": (
                        float(row["impliedVolatility"])
                        if pd.notna(row["impliedVolatility"])
                        else 0
                    ),
                }
            )

        # Format puts data
        puts = []
        for _, row in opt.puts.head(20).iterrows():  # Limit to top 20
            puts.append(
                {
                    "strike": float(row["strike"]),
                    "last_price": (
                        float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0
                    ),
                    "bid": float(row["bid"]) if pd.notna(row["bid"]) else 0,
                    "ask": float(row["ask"]) if pd.notna(row["ask"]) else 0,
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    "open_interest": (
                        int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0
                    ),
                    "implied_volatility": (
                        float(row["impliedVolatility"])
                        if pd.notna(row["impliedVolatility"])
                        else 0
                    ),
                }
            )

        return {
            "symbol": symbol.upper(),
            "expiration_date": exp_date,
            "available_expirations": list(expirations)[:10],  # First 10 dates
            "calls": calls,
            "puts": puts,
        }
    except Exception as e:
        return {"error": f"Failed to fetch options data for {symbol}: {str(e)}"}


# Financial Calculation Tools


@mcp.tool()
async def calculate_moving_average(
    symbol: str, period: int = 20, ma_type: str = "SMA"
) -> dict:
    """Calculate moving average for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Number of periods for the moving average
        ma_type: Type of moving average (SMA or EMA)

    Returns:
        Dictionary with moving average data
    """
    try:
        # Get historical data
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="3mo")

        if hist.empty:
            return {"error": f"No data available for {symbol}"}

        close_prices = hist["Close"]

        if ma_type.upper() == "SMA":
            ma = close_prices.rolling(window=period).mean()
        elif ma_type.upper() == "EMA":
            ma = close_prices.ewm(span=period, adjust=False).mean()
        else:
            return {"error": "Invalid MA type. Use 'SMA' or 'EMA'"}

        # Get last 10 values
        recent_data = []
        for date, value in ma.tail(10).items():
            if pd.notna(value):
                recent_data.append(
                    {"date": date.strftime("%Y-%m-%d"), "value": round(value, 2)}
                )

        return {
            "symbol": symbol.upper(),
            "period": period,
            "type": ma_type.upper(),
            "current_value": round(ma.iloc[-1], 2) if pd.notna(ma.iloc[-1]) else None,
            "recent_values": recent_data,
        }
    except Exception as e:
        return {"error": f"Failed to calculate moving average: {str(e)}"}


@mcp.tool()
async def calculate_rsi(symbol: str, period: int = 14) -> dict:
    """Calculate Relative Strength Index (RSI) for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        Dictionary with RSI data
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="3mo")

        if hist.empty:
            return {"error": f"No data available for {symbol}"}

        close_prices = hist["Close"]

        # Calculate price changes
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Get last 10 values
        recent_data = []
        for date, value in rsi.tail(10).items():
            if pd.notna(value):
                recent_data.append(
                    {"date": date.strftime("%Y-%m-%d"), "value": round(value, 2)}
                )

        current_rsi = rsi.iloc[-1] if pd.notna(rsi.iloc[-1]) else None

        # Interpretation
        interpretation = "Neutral"
        if current_rsi and current_rsi > 70:
            interpretation = "Overbought"
        elif current_rsi and current_rsi < 30:
            interpretation = "Oversold"

        return {
            "symbol": symbol.upper(),
            "period": period,
            "current_rsi": round(current_rsi, 2) if current_rsi else None,
            "interpretation": interpretation,
            "recent_values": recent_data,
        }
    except Exception as e:
        return {"error": f"Failed to calculate RSI: {str(e)}"}


@mcp.tool()
async def calculate_sharpe_ratio(
    symbol: str, period: str = "1y", risk_free_rate: float = 0.05
) -> dict:
    """Calculate Sharpe Ratio for a stock.

    Args:
        symbol: Stock ticker symbol
        period: Time period for calculation (1mo, 3mo, 6mo, 1y, 2y, 5y)
        risk_free_rate: Annual risk-free rate (default: 0.05 or 5%)

    Returns:
        Dictionary with Sharpe Ratio and related metrics
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if hist.empty or len(hist) < 20:
            return {"error": f"Insufficient data for {symbol}"}

        # Calculate daily returns
        daily_returns = hist["Close"].pct_change().dropna()

        # Calculate annualized metrics
        trading_days = 252
        avg_daily_return = daily_returns.mean()
        daily_volatility = daily_returns.std()

        annualized_return = avg_daily_return * trading_days
        annualized_volatility = daily_volatility * np.sqrt(trading_days)

        # Calculate Sharpe Ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        return {
            "symbol": symbol.upper(),
            "period": period,
            "sharpe_ratio": round(sharpe_ratio, 3),
            "annualized_return": round(annualized_return * 100, 2),  # as percentage
            "annualized_volatility": round(
                annualized_volatility * 100, 2
            ),  # as percentage
            "risk_free_rate": round(risk_free_rate * 100, 2),  # as percentage
            "interpretation": (
                "Excellent"
                if sharpe_ratio > 2
                else (
                    "Good"
                    if sharpe_ratio > 1
                    else "Acceptable" if sharpe_ratio > 0.5 else "Poor"
                )
            ),
        }
    except Exception as e:
        return {"error": f"Failed to calculate Sharpe Ratio: {str(e)}"}


@mcp.tool()
async def compare_stocks(symbols: List[str], metric: str = "performance") -> dict:
    """Compare multiple stocks by various metrics.

    Args:
        symbols: List of stock ticker symbols
        metric: Comparison metric ('performance', 'volatility', 'volume', 'pe_ratio')

    Returns:
        Dictionary with comparison data
    """
    try:
        if len(symbols) > 5:
            return {"error": "Maximum 5 symbols allowed for comparison"}

        comparison_data = []

        for symbol in symbols:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            if metric == "performance":
                hist = ticker.history(period="1mo")
                if not hist.empty:
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    performance = ((end_price - start_price) / start_price) * 100
                else:
                    performance = None

                comparison_data.append(
                    {
                        "symbol": symbol.upper(),
                        "current_price": info.get("currentPrice", "N/A"),
                        "1m_performance": (
                            round(performance, 2) if performance else "N/A"
                        ),
                        "market_cap": info.get("marketCap", "N/A"),
                    }
                )

            elif metric == "volatility":
                hist = ticker.history(period="1mo")
                if not hist.empty:
                    daily_returns = hist["Close"].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
                else:
                    volatility = None

                comparison_data.append(
                    {
                        "symbol": symbol.upper(),
                        "volatility": round(volatility, 2) if volatility else "N/A",
                        "beta": info.get("beta", "N/A"),
                    }
                )

            elif metric == "pe_ratio":
                comparison_data.append(
                    {
                        "symbol": symbol.upper(),
                        "pe_ratio": info.get("trailingPE", "N/A"),
                        "forward_pe": info.get("forwardPE", "N/A"),
                        "peg_ratio": info.get("pegRatio", "N/A"),
                    }
                )

            elif metric == "volume":
                comparison_data.append(
                    {
                        "symbol": symbol.upper(),
                        "volume": info.get("volume", "N/A"),
                        "avg_volume": info.get("averageVolume", "N/A"),
                        "volume_ratio": (
                            round(
                                info.get("volume", 0) / info.get("averageVolume", 1), 2
                            )
                            if info.get("averageVolume", 0) > 0
                            else "N/A"
                        ),
                    }
                )

        return {"metric": metric, "comparison": comparison_data}
    except Exception as e:
        return {"error": f"Failed to compare stocks: {str(e)}"}


# Cache management tool
@mcp.tool()
async def clear_cache() -> str:
    """Clear the cache to force fresh data retrieval."""
    global cache
    cache.clear()
    return "Cache cleared successfully"


if __name__ == "__main__":
    # Run the MCP server
    print("Starting Financial Data MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")
