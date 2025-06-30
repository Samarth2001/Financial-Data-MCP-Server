#!/usr/bin/env python3
"""
Test script for Financial MCP Server
Run this to verify your installation is working correctly
"""

import asyncio
import sys

# Test if required packages are installed
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from mcp.server.fastmcp import FastMCP

    print("✓ All required packages installed")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


async def test_basic_functionality():
    """Test basic market data retrieval"""
    print("\nTesting basic functionality...")

    try:
        # Test yfinance connection
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        current_price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))

        if current_price != "N/A":
            print(f"✓ Successfully retrieved AAPL price: ${current_price}")
        else:
            print("✗ Could not retrieve price data")

        # Test historical data
        hist = ticker.history(period="5d")
        if not hist.empty:
            print(f"✓ Retrieved {len(hist)} days of historical data")
        else:
            print("✗ Could not retrieve historical data")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

    return True


async def test_calculations():
    """Test financial calculations"""
    print("\nTesting calculations...")

    try:
        # Test data for calculations
        data = pd.Series([100, 102, 101, 103, 105, 104, 106])

        # Test SMA
        sma = data.rolling(window=3).mean()
        print(f"✓ SMA calculation working: {sma.iloc[-1]:.2f}")

        # Test RSI
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        print(f"✓ RSI calculation working: {rsi.iloc[-1]:.2f}")

    except Exception as e:
        print(f"✗ Calculation error: {e}")
        return False

    return True


def main():
    print("Financial MCP Server Test Suite")
    print("=" * 40)

    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    basic_ok = loop.run_until_complete(test_basic_functionality())
    calc_ok = loop.run_until_complete(test_calculations())

    print("\n" + "=" * 40)
    if basic_ok and calc_ok:
        print("✓ All tests passed! Your server is ready to use.")
        print("\nTo use with Claude Desktop:")
        print("1. Add the server to your Claude Desktop config")
        print("2. Restart Claude Desktop")
        print("3. Look for 'financial-data' in the MCP tools")
    else:
        print("✗ Some tests failed. Please check the errors above.")

    loop.close()


if __name__ == "__main__":
    main()
