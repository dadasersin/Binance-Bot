# Binance Trading Bot

This bot is designed to be deployed as a Background Worker on [Render](https://render.com).

## Features
- RSI and MACD indicators for trading signals.
- Automatic stop-loss based on daily volatility.
- Configurable trading pairs via environment variables.
- Robust error handling and logging.
- Precision handling for Binance lot sizes and price filters.

## Deployment to Render

1. Create a new **Background Worker** on Render.
2. Connect your GitHub repository.
3. Set the **Environment** to `Python`.
4. Set the **Start Command** to `python main.py` (or let Render use the `Procfile`).
5. Add the following **Environment Variables**:

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Your Binance API Key | Required |
| `BINANCE_API_SECRET` | Your Binance API Secret | Required |
| `TRADE_PAIR_BASE` | Base asset (e.g., BNB) | `BNB` |
| `TRADE_PAIR_QUOTE` | Quote asset (e.g., USDT) | `USDT` |
| `WIN_RATE` | Target win rate multiplier | `1.02` |

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export environment variables:
   ```bash
   export BINANCE_API_KEY='your_key'
   export BINANCE_API_SECRET='your_secret'
   ```
3. Run the bot:
   ```bash
   python main.py
   ```

## Disclaimer
Trading cryptocurrencies carries significant risk. This bot is provided for educational purposes only. Use it at your own risk.
