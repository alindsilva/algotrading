import threading
import time
import empyrical as ep

import sqlite3

from wrapper import IBWrapper
from client import IBClient
from contract import stock
from order import market, limit, BUY, SELL


class IBApp(IBWrapper, IBClient):
    def __init__(self, ip, port, client_id, account, interval=5):
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)
        self.account = account
        self.create_table()

        self.connect(ip, port, client_id)

        threading.Thread(target=self.run, daemon=True).start()
        time.sleep(5)
        threading.Thread(
            target=self.get_streaming_returns,
            args=(99, interval, "unrealized_pnl"),
            daemon=True,
        ).start()

    @property
    def connection(self):
        return sqlite3.connect("tick_data.sqlite", isolation_level=None)

    def create_table(self):
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS bid_ask_data (timestamp datetime, symbol string, bid_price real, ask_price real, bid_size integer, ask_size integer)"
        )

    def stream_to_sqlite(self, request_id, contract, run_for_in_seconds=23400):
        cursor = self.connection.cursor()
        end_time = time.time() + run_for_in_seconds + 10
        for tick in app.get_streaming_data(request_id, contract):
            query = "INSERT INTO bid_ask_data (timestamp, symbol, bid_price, ask_price, bid_size, ask_size) VALUES (?, ?, ?, ?, ?, ?)"
            values = (
                tick.timestamp_.strftime("%Y-%m-%d %H:%M:%S"),
                contract.symbol,
                tick.bid_price,
                tick.ask_price,
                tick.bid_size,
                tick.ask_size,
            )
            cursor.execute(query, values)
            if time.time() >= end_time:
                break

        self.stop_streaming_data(request_id)

    @property
    def cumulative_returns(self):
        return ep.cum_returns(self.portfolio_returns, 1)

    @property
    def max_drawdown(self):
        return ep.max_drawdown(self.portfolio_returns)

    @property
    def volatility(self):
        return self.portfolio_returns.std(ddof=1)

    @property
    def omega_ratio(self):
        return ep.omega_ratio(self.portfolio_returns, annualization=1)

    @property
    def sharpe_ratio(self):
        return self.portfolio_returns.mean() / self.portfolio_returns.std(ddof=1)

    @property
    def cvar(self):
        net_liquidation = self.get_account_values("NetLiquidation")[0]
        cvar_ = ep.conditional_value_at_risk(self.portfolio_returns)
        return (cvar_, cvar_ * net_liquidation)


if __name__ == "__main__":
    app = IBApp("127.0.0.1", 7497, client_id=11, account="DU8665110")

    aapl = stock("AAPL", "SMART", "USD")
    mid = stock("MID","ARCA", "USD")
    xlk = stock("XLK", "ARCA", "USD")
    xlv = stock("XLV", "ARCA", "USD")
    xly = stock("XLY", "ARCA", "USD")
    xlu = stock("XLU", "ARCA", "USD")
    xlf = stock("XLF", "ARCA", "USD")
    xlb = stock("XLB", "ARCA", "USD")
    xli = stock("XLI", "ARCA", "USD")
    xlp = stock("XLP", "ARCA", "USD")
    xlc = stock("XLC", "ARCA", "USD")
    xle = stock("XLE", "ARCA", "USD")
    xlre = stock("XLRE", "ARCA", "USD")
    snow = stock("SNOW", "SMART", "USD")
    ddog = stock("DDOG", "SMART", "USD")
    crm = stock("CRM", "SMART", "USD")
    eem = stock("EEM", "ARCA", "USD")
    gld = stock("GLD", "ARCA", "USD")

    # market order value
    # app.order_value(aapl, market, 1000, action=BUY)

    # market order target quantity
    # app.order_target_quantity(aapl, market, -5)

    # market order percent
    # app.order_percent(aapl, market, 0.1, action=BUY)
    # app.order_percent(aapl, limit, 0.1, action=BUY, limit_price=185.0)

    # market order target value
    # app.order_target_value(aapl, market, 3000)
    # app.order_target_value(aapl, stop, 3000, stop_price=180.0)

    # market order target percent
    app.order_target_percent(xlk, market, 0.275)
    app.order_target_percent(eem, market, 0.025)
    app.order_target_percent(ddog, market, 0.015)
    app.order_target_percent(xle, market, 0.05)
    app.order_target_percent(gld, market, 0.256)
    app.order_target_percent(mid, market, 0.1)
    app.order_target_percent(xlb, market, 0.019)
    
    
    app.order_target_percent(xlre, market, 0.023)
    app.order_target_percent(xlv, market, 0.101)
    app.order_target_percent(xlf, market, 0.136)

    time.sleep(15)
    app.disconnect()
