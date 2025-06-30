# Financial Data MCP Server

A Model Context Protocol (MCP) server that provides AI assistants with real-time financial market data through Yahoo Finance.

## Features

- **Stock Data**: Current prices, historical data, market metrics
- **Options**: Options chains with strikes, prices, and volume
- **Technical Analysis**: Moving averages, RSI, Sharpe ratio
- **Comparisons**: Compare multiple stocks across metrics

## Installation

```bash
pip install "mcp[cli]" yfinance pandas numpy
```

## Configuration

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "financial-data": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Available Tools

- `get_stock_price` - Current price and metrics
- `get_historical_data` - Historical price data
- `get_options_chain` - Options data for calls/puts
- `calculate_moving_average` - SMA/EMA calculations
- `calculate_rsi` - Relative Strength Index
- `calculate_sharpe_ratio` - Risk-adjusted returns
- `compare_stocks` - Multi-stock comparison
- `clear_cache` - Clear cached data

## Usage Examples

```
"What's the current price of AAPL?"
"Show me TSLA's 50-day moving average"
"Compare AAPL, MSFT, and GOOGL performance"
"Get SPY options expiring this Friday"
```

## Troubleshooting

- **Missing modules**: Run `pip install yfinance pandas numpy "mcp[cli]"`
- **No data**: Verify ticker symbol and internet connection
- **Server issues**: Check Python path in config, restart Claude Desktop

## Notes

- Data from Yahoo Finance (15-20 minute delay)
- Automatic caching and rate limiting
- For informational purposes only

## License

MIT License