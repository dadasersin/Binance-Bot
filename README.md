# Trading Bots on Render

This repository contains multiple Python-based trading bots for Binance and KuCoin, ready to be deployed on [Render](https://render.com/).

## Deployment Instructions

1. **Fork this Repository:** Ensure these files are in your own GitHub repository.
2. **Go to Render Dashboard:**
   - Click **New** > **Blueprint**.
   - Connect your GitHub account and select this repository.
3. **Configure the Blueprint:**
   - Provide a name for the Blueprint (e.g., `trading-stack`).
   - Render will detect the `render.yaml` file and prepare a **Background Worker**.
4. **Set Environment Variables:**
   - During or after deployment, you must set the following variables in the Render Dashboard for the service to function:
     - `BOT_PATH`: Path to the bot you want to run (e.g., `BINANCE/bot_3.py`). Default is `BINANCE/bot_3.py`.
     - `BINANCE_API_KEY`: Your Binance API Key.
     - `BINANCE_API_SECRET`: Your Binance API Secret.
     - `KUCOIN_API_KEY`: Your KuCoin API Key (if using KuCoin bots).
     - `KUCOIN_API_SECRET`: Your KuCoin API Secret.
     - `KUCOIN_API_PASSPHRASE`: Your KuCoin API Passphrase.
5. **Deploy:** Click **Deploy**.

## Available Bots

- **Binance:**
  - `BINANCE/bot_1.py`: Simple price reference bot.
  - `BINANCE/bot_2.py`: RSI-based bot.
  - `BINANCE/bot_3.py`: RSI + MACD + StopLoss bot (Recommended).
- **KuCoin:**
  - `KUCOIN/bot_1301[kucoin].py`: Order rate and RSI based bot.

## Configuration Details

- **Runtime:** Docker (Python 3.9-slim).
- **Service Type:** Background Worker (Ideal for continuous loops).

---
*Prepared for deployment on Render.*
