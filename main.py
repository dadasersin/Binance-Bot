import time
import datetime as DT
import os
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def computeRSI(data, time_window):
    diff = np.diff(data)
    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]

    up_chg = pd.DataFrame(up_chg)
    down_chg = pd.DataFrame(down_chg)

    up_chg_avg = up_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1, min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    if rsi.empty or pd.isna(rsi[0].iloc[-1]):
        return 50 # Default to neutral if calculation fails
    return int(rsi[0].iloc[-1])

def get_macd_indicator(client, symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval=KLINE_INTERVAL_5MINUTE, limit=60)
        close_vals = [float(entry[4]) for entry in klines]
        close_df = pd.DataFrame(close_vals)
        ema12 = close_df.ewm(span=12).mean()
        ema26 = close_df.ewm(span=26).mean()
        macd = ema26 - ema12
        signal = macd.ewm(span=9).mean()

        macd_val = macd.values.flatten()
        signal_val = signal.values.flatten()

        if macd_val[-1] > signal_val[-1] and macd_val[-2] < signal_val[-2]:
            return 'BUY'
        elif macd_val[-1] < signal_val[-1] and macd_val[-2] > signal_val[-2]:
            return 'SELL'
        else:
            return 'HOLD'
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return 'HOLD'

def get_stop_loss_val(client, symbol):
    try:
        today = DT.date.today()
        week_ago = today - DT.timedelta(days=6)
        week_ago_str = week_ago.strftime('%d %b, %Y')
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, week_ago_str)
        if not klines:
            return 0
        high_vals = [float(entry[2]) for entry in klines]
        low_vals = [float(entry[3]) for entry in klines]
        close_vals = [float(entry[4]) for entry in klines]

        avg_down_drop = (sum(high_vals)/len(high_vals) - sum(low_vals)/len(low_vals)) / (sum(close_vals)/len(close_vals))
        stop_val = close_vals[-2] * (1 - avg_down_drop)
        return stop_val
    except Exception as e:
        logger.error(f"Error calculating Stop Loss: {e}")
        return 0

def round_step_size(quantity, step_size):
    precision = int(round(-np.log10(float(step_size))))
    return floor_to_precision(quantity, precision)

def floor_to_precision(value, precision):
    factor = 10 ** precision
    return np.floor(value * factor) / factor

def main():
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')

    if not api_key or not api_secret:
        logger.error("API Key or Secret not found in environment variables.")
        return

    trd_pair1 = os.getenv('TRADE_PAIR_BASE', 'BNB')
    trd_pair2 = os.getenv('TRADE_PAIR_QUOTE', 'USDT')
    win_rate = float(os.getenv('WIN_RATE', '1.02'))

    client = Client(api_key, api_secret)
    trade_pair = trd_pair1 + trd_pair2

    logger.info(f"Starting bot for {trade_pair}...")

    # Initialize state
    last_trade = None
    last_price = 0.0

    while True:
        try:
            # Get current price
            ticker = client.get_ticker(symbol=trade_pair)
            current_price = float(ticker['askPrice'])

            # Get balances
            base_balance = client.get_asset_balance(asset=trd_pair1)
            quote_balance = client.get_asset_balance(asset=trd_pair2)

            if not base_balance or not quote_balance:
                logger.error("Could not fetch balances.")
                time.sleep(60)
                continue

            base_free = float(base_balance['free'])
            quote_free = float(quote_balance['free'])

            base_value_in_quote = base_free * current_price

            # Determine last trade if not set
            if last_trade is None:
                if base_value_in_quote > quote_free:
                    last_trade = trd_pair1
                else:
                    last_trade = trd_pair2

                # Try to get last price from trade history
                try:
                    my_trades = client.get_my_trades(symbol=trade_pair, limit=1)
                    if my_trades:
                        last_price = float(my_trades[0]['price'])
                    else:
                        last_price = current_price
                except Exception as e:
                    logger.warning(f"Could not fetch trade history: {e}. Using current price as last price.")
                    last_price = current_price

            # Indicators
            klines = client.get_klines(symbol=trade_pair, interval=KLINE_INTERVAL_5MINUTE, limit=500)
            close_prices = [float(entry[4]) for entry in klines]
            rsi = computeRSI(close_prices[:-1], 14)
            macd_signal = get_macd_indicator(client, trade_pair)
            stop_loss_price = get_stop_loss_val(client, trade_pair)

            server_time = client.get_server_time()
            readable_time = time.strftime('%m/%d/%Y %H:%M:%S', time.gmtime(server_time['serverTime']/1000.))

            status = "HOLD"

            # SELL Logic
            if last_trade == trd_pair1:
                if ((current_price > last_price * win_rate) and (rsi > 70 or macd_signal == 'SELL')) or (current_price < stop_loss_price):
                    if current_price < stop_loss_price:
                        logger.info(f"STOP LOSS triggered: Price={current_price}, StopLoss={stop_loss_price}")
                        status = "STOPLOSS"
                    else:
                        logger.info(f"SELL signal: Price={current_price}, RSI={rsi}, MACD={macd_signal}")
                        status = "SELL"

                    # Fetch exchange info for precision
                    info = client.get_symbol_info(trade_pair)
                    step_size = [f['stepSize'] for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0]
                    tick_size = [f['tickSize'] for f in info['filters'] if f['filterType'] == 'PRICE_FILTER'][0]

                    quantity = round_step_size(base_free, step_size)
                    price_str = format(floor_to_precision(current_price, int(round(-np.log10(float(tick_size))))), 'f')

                    try:
                        logger.info(f"Executing {status} order: {quantity} {trd_pair1} @ {price_str}")
                        order = client.create_order(
                            symbol=trade_pair,
                            side=SIDE_SELL,
                            type=ORDER_TYPE_LIMIT,
                            timeInForce=TIME_IN_FORCE_GTC,
                            quantity=quantity,
                            price=price_str
                        )
                        last_trade = trd_pair2
                        last_price = current_price
                    except Exception as e:
                        logger.error(f"Order failed: {e}")

                else:
                    status = f"HOLD {trd_pair1} (Target: {last_price * win_rate:.4f})"

            # BUY Logic
            elif last_trade == trd_pair2:
                if (current_price * win_rate < last_price) and (macd_signal == 'BUY' or rsi < 30):
                    logger.info(f"BUY signal: Price={current_price}, RSI={rsi}, MACD={macd_signal}")
                    status = "BUY"

                    # Fetch exchange info for precision
                    info = client.get_symbol_info(trade_pair)
                    step_size = [f['stepSize'] for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0]
                    tick_size = [f['tickSize'] for f in info['filters'] if f['filterType'] == 'PRICE_FILTER'][0]

                    # Calculate quantity based on quote balance
                    quantity_raw = quote_free / current_price
                    quantity = round_step_size(quantity_raw * 0.99, step_size) # 99% to account for fees/fluctuations
                    price_str = format(floor_to_precision(current_price, int(round(-np.log10(float(tick_size))))), 'f')

                    try:
                        logger.info(f"Executing BUY order: {quantity} {trd_pair1} @ {price_str}")
                        order = client.create_order(
                            symbol=trade_pair,
                            side=SIDE_BUY,
                            type=ORDER_TYPE_LIMIT,
                            timeInForce=TIME_IN_FORCE_GTC,
                            quantity=quantity,
                            price=price_str
                        )
                        last_trade = trd_pair1
                        last_price = current_price
                    except Exception as e:
                        logger.error(f"Order failed: {e}")
                else:
                    status = f"HOLD {trd_pair2} (Target: {last_price / win_rate:.4f})"

            logger.info(f"{readable_time} | {trd_pair1}: {base_free:.4f} | RSI: {rsi} | MACD: {macd_signal} | Price: {current_price:.4f} | Status: {status}")

        except BinanceAPIException as e:
            logger.error(f"Binance API Exception: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")

        time.sleep(60)

if __name__ == "__main__":
    main()
