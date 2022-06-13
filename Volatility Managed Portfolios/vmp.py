#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
The following code represents an implementation of a volatility managed portfolio,
with periodic rebalances.
License_info: Licensed to Portfolio Management Club Nova SBE;
"""

__author__ = "Povilas Navickas, Diogo Barreiros"
__copyright__ = "Copyright Spring 2022, PMC Quantitative Team"
__credits__ = ["Daniel Gonçalves"]
__license__ = "PMC"
__maintainer__ = "Quantitative Team"
__email__ = "portfoliomanagementclub@novasbe.pt"
__status__ = "Production"


# --------------------------------------------------------------------------- #
#                            Imports Checker                                  #
# --------------------------------------------------------------------------- #
hard_dependencies = (
    "yfinance",
    "pandas",
    "pandas_datareader",
    "datetime",
    "arch",
    "matplotlib.pyplot",
    "quantstats"
    "tabulate",
)
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" +
        "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

# --------------------------------------------------------------------------- #
#                                  Imports                                    #
# --------------------------------------------------------------------------- #

import plotly.graph_objects as go
import quantstats as qs
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas_datareader.data as reader
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------- #
#                               Class Assembly                                #
# --------------------------------------------------------------------------- #


class VolatilityManagedPortfolio(object):
    """
    This class encapsules a complete volatility managed portfolio implementation
    based on a GARCH estimation of future volatilities.


    Parameters
    ----------
    Pass at least one of the following two:

        portfolio: pd.DataFrame or iterable of tickers
            DataFrame object with Price Data, Index: DateTimeIndex, Columns: Tickers
            Iterable with tickers, prices are downloaded from yfinance

        tradinglog: pd.DataFrame
            DataFrame Object with the following columns:
            ["Date", "Ticker", "Quantity Inflow", "Price Inflow", "Amount Inflow", 
            "Quantity Outflow","Price Outflow", "Amount Inflow"].
            All positive values.

    benchmark: pd.Series of prices or normalized prices.
        If None, benchmark is the equal-weighted portfolio of the tickers passed.

    Atributes
    ---------
        benchmark: pd.Series with Benchmark prices/ price information for the tickers
        
        portfolio: pd.DataFrame with price information for the tickers

        trading_log: equal to tradinglog (if passed)

        predictions_df: pd.DataFrame with the prediction of the tickers's volatility

        garch_params: provides information on the GARCH Model Fit for every ticker
        
        strategy_returns: strategy cumulative returns rebased to 0 (cum returns - 1)
        
        three_factor_regression_summary: result summary of FamaFrench Regression with 3 factors

        fivee_factor_regression_summary: result summary of FamaFrench Regression with 5 factors

        strategy_weights: pd.DataFrame with ticker weights under the volatility managed portfolio 

        daily_positions: pd.DataFrame with the #Shares held in each day based on tradinglog

        benchmark_cumulative_returns: benchmark cumulative returns rebased to 0 (cum returns - 1)

    Methods
    -------
    volatility_portfolio:

    portfolio_analysis:

    summary_plot:

    plot_weights:

    report:

    Auxiliar Methods:
    -----------------
    garch:

    three_factor_regression:

    five_factor_regression:


    """

    def __init__(
        self,
        portfolio: pd.DataFrame = None,
        benchmark: pd.DataFrame = None,
        trading_log: pd.DataFrame = None,
    ):

        self.benchmark = benchmark
        self.portfolio = portfolio
        self.trading_log = trading_log
        self.predictions_df: pd.DataFrame = pd.DataFrame()
        self.garch_params: dict = dict()
        self.strategy_returns = None
        self.three_factor_regression_summary = None
        self.five_factor_regression_summary = None
        self.strategy_weights: pd.DataFrame = None
        self.daily_positions = None
        self.benchmark_cumulative_returns = None

        if isinstance(portfolio, (list, set, tuple)):
            self.portfolio = yf.download(portfolio)['Adj Close']
            self.portfolio_returns = self.portfolio.pct_change().dropna()

        elif isinstance(portfolio, (pd.DataFrame, pd.Series)):
            self.portfolio_returns = portfolio.pct_change().dropna()

        if portfolio is None and trading_log is None:
            raise KeyError("Missing Portfolio or Trading_Log Parameter")

        if trading_log is not None:
            trading_log["Date"] = pd.to_datetime(trading_log["Date"])
            main_df = pd.DataFrame()
            for ticker in set(trading_log["Ticker"]):
                tic_date = []
                for date in pd.bdate_range(min(trading_log["Date"]), dt.date.today()):
                    warehouse_filtered = trading_log.loc[
                        trading_log["Ticker"] == str(ticker)
                    ]
                    shares_df = warehouse_filtered.loc[
                        trading_log["Date"]
                        <= dt.datetime(date.year, date.month, date.day)
                    ]
                    shares = (
                        shares_df["Quantity Inflow"].sum()
                        - shares_df["Quantity Outflow"].sum()
                    )
                    tic_date.append((pd.to_datetime(date), shares))
                ser_tic = pd.DataFrame(tic_date, columns=["Date", str(ticker)])
                ser_tic.set_index("Date", inplace=True)
                main_df = pd.concat([main_df, ser_tic], axis=1)

            self.daily_positions = main_df
            self.portfolio = yf.download(
                set(trading_log["Ticker"]))["Adj Close"]

        if benchmark is None or len(benchmark.columns) >= 1:
            self.benchmark = self.portfolio
            # Equal Weighted Portfolio
            benchmark_returns = self.benchmark.pct_change().dropna().sum(axis=1) / len(
                self.benchmark.columns
            )
            self.benchmark_returns = benchmark_returns
        else:
            self.benchmark_returns = benchmark.pct_change().dropna()

    def volatility_portfolio(
        self, start=None, sample_size=0.8, cutoff_date=None, rebalancing="daily", weight_limit=0,
    ):
        start = start if start is not None else min(self.portfolio.index)
        # Fill the predictions DataFrame with predicted volatilities for each asset:
        for ticker in self.portfolio_returns:
            garch_result = self.garch(ticker, sample_size, cutoff_date)
            self.garch_params[str(ticker)] = garch_result

        # Creating a portfolio of the tickers /based on their volatility (weight inverse to volatility basically)

        inverse_volatilities = 1 / self.predictions_df
        weights = inverse_volatilities.div(
            inverse_volatilities.sum(axis=1), axis=0)

        if self.trading_log is not None:
            for i in self.daily_positions.index.intersection(weights.index):
                for ticker in inverse_volatilities:
                    if self.daily_positions.loc[i, ticker] == 0:
                        weights.loc[i, ticker] = 0
            weights = weights.div(weights.sum(axis=1), axis=0)

        if rebalancing.lower() == "weekly":
            weights_weekly = pd.DataFrame()
            for i in weights.index:
                for ticker in weights:
                    if i.weekday() == 0:
                        weights_weekly.loc[i, ticker] = weights.loc[i, ticker]
                    else:
                        weights_weekly.loc[i, ticker] = np.nan
            weights_weekly = weights_weekly.ffill().bfill()
            weights = weights_weekly.div(weights_weekly.sum(axis=1), axis=0)

        if weight_limit:
            for i in weights.index:
                for ticker in weights:
                    if weights.loc[i, ticker] > weight_limit:
                        weights.loc[i, ticker] = weight_limit
                    weights = weights.div(weights.sum(axis=1), axis=0)

        weighted_returns = weights * \
            self.portfolio_returns.loc[min(weights.index):]
        weighted_returns_sum = weighted_returns.sum(axis=1)
        weighted_returns_sum = weighted_returns_sum.to_frame()

        cumulative_returns = (weighted_returns_sum + 1).cumprod() - 1
        cumulative_returns.columns = ["VM Portfolio"]
        benchmark_cumulative = (
            self.benchmark_returns.loc[min(cumulative_returns.index):] + 1
        ).cumprod() - 1
        benchmark_cumulative.name = "Equal Weighted Portfolio"

        self.strategy_weights = weights
        self.strategy_returns = cumulative_returns
        self.benchmark_cumulative_returns = benchmark_cumulative

        self.three_factor_regression(weighted_returns_sum)
        self.five_factor_regression(weighted_returns_sum)
        return (
            (
                weights,
                pd.concat([cumulative_returns, benchmark_cumulative],
                          axis="columns"),
                self.summary_plot()
            ),
            (self.three_factor_regression_summary,
             self.five_factor_regression_summary),
        )

    def garch(self, ticker, sample_size=0.8, cutoff_date=None):
        cutoff = (
            self.portfolio_returns.index.get_loc(cutoff_date, "bfill")
            if cutoff_date is not None
            else int(len(self.portfolio_returns) * sample_size)
        )
        train_df = self.portfolio_returns.iloc[:cutoff]  # default to 80/20
        test_df = self.portfolio_returns.iloc[cutoff:]
        n = train_df.shape[0]

        garch_df = pd.DataFrame(
            self.portfolio_returns[ticker].shift(
                1).loc[self.portfolio_returns.index]
        )
        garch_df.loc[train_df.index, ticker] = train_df[ticker]

        model = arch_model(garch_df[ticker], p=1,
                           q=1, vol="GARCH", rescale=False)
        model_results = model.fit(
            last_obs=train_df.index[n - 1], update_freq=5)
        summary = model_results.summary()

        # Building Predictions Data
        self.predictions_df[ticker] = (
            model_results.forecast(
                reindex=False).residual_variance.loc[test_df.index] * 100
        )  # Gives predictions on volatility that are based on previous residuals and volatilities
        return summary

    def three_factor_regression(self, returns):
        returns_monthly = returns.resample(
            "M").agg(lambda x: (x + 1).prod() - 1)
        returns_monthly.index = returns_monthly.index.to_period("M")

        factors = reader.DataReader(
            "F-F_Research_Data_Factors",
            "famafrench",
            dt.date(2015, 1, 1),
            dt.date(2022, 1, 1),
        )[0]
        merged = pd.merge(returns_monthly, factors, on="Date")
        merged.dropna(inplace=True)
        merged[["Mkt-RF", "SMB", "HML", "RF"]] = (
            merged[["Mkt-RF", "SMB", "HML", "RF"]] / 100
        )

        y = merged.iloc[:, 0]
        # y = merged["Portfolio"]
        x = merged[["Mkt-RF", "SMB", "HML"]]
        x_sm = sm.add_constant(x)
        model = sm.OLS(y, x_sm)
        results = model.fit()
        self.three_factor_regression_summary = results.summary()
        return results.summary()

    def five_factor_regression(self, returns):
        df_factors = reader.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily", "famafrench"
        )[0]/100
        my_index = returns.index.intersection(df_factors.index)
        returns = returns.loc[my_index]
        merged = pd.merge(returns, df_factors, on="Date")
        merged[["Mkt-RF", "SMB", "HML", "RF", "CMA", "RMW"]] = (
            merged[["Mkt-RF", "SMB", "HML", "RF", "CMA", "RMW"]] / 100
        )
        y = merged.iloc[:, 0]
        # y = merged["Portfolio"]
        x = merged[["Mkt-RF", "SMB", "HML", "CMA", "RF"]]
        x_sm = sm.add_constant(x)
        model = sm.OLS(y, x_sm)
        results = model.fit()
        self.five_factor_regression_summary = results.summary()
        return results.summary()

    def portfolio_analysis(self):
        # Returns
        def portfolio_returns_func(returns):
            return float(round((returns.mean() * 260), 4))

        # Volatility of portfolio returns
        def portfolio_volatility(returns):
            return float(round(returns.std() * (260 ** 0.5), 4))

        # Sharpe Ratio
        def sharpe_ratio(portfolio_volatility, portfolio_returns):
            sharpe_ratio = float(
                round((portfolio_returns / portfolio_volatility), 4))
            return sharpe_ratio

        port_returns = portfolio_returns_func(self.strategy_returns)
        port_volatility = portfolio_volatility(self.strategy_returns)
        port_sharpe = sharpe_ratio(port_volatility, port_returns)

        benchmark_returns = self.benchmark_returns.loc["2015-01-01":].sum(
            axis=1)

        bench_returns = portfolio_returns_func(benchmark_returns)
        bench_volatility = portfolio_volatility(benchmark_returns)
        bench_sharpe = sharpe_ratio(bench_volatility, bench_returns)

        table = [
            ["Portfolio", "Annual Returns", "Annual Volatility", "Sharpe Ratio"],
            ["SPY", bench_returns, bench_volatility, bench_sharpe],
            ["Volatility-Managed", port_returns, port_volatility, port_sharpe],
        ]
        table = pd.DataFrame(table)

        return table

    def summary_plot(self):
        both_ret = pd.DataFrame()
        both_ret["Base Scenario"] = self.strategy_returns
        both_ret["Volatility Managed Strategy"] = self.benchmark_cumulative_returns
        both_ret.plot()
        plt.show()

    def plot_weights(self, title="Weights VMP"):
        fig = go.Figure()
        columns = self.strategy_weights.columns
        x = self.strategy_weights.index
        for col in columns:
            y = self.strategy_weights[col]
            trace = go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=0.5),
                groupnorm="percent",
                name=col,
                stackgroup="one",  # define stack group
            )
            fig.add_trace(trace)
        fig.update_layout(template="plotly_white")
        fig.update_layout(title=title)
        fig.show()

    def report(self,
               report_name=f"VMP_{dt.datetime.now()}", download_filename=f"VMP_{dt.datetime.now()}"
               ):
        return qs.reports.html(
            (self.strategy_returns+1).squeeze(),
            (self.benchmark_cumulative_returns+1).squeeze(),
            output=report_name+".html",
            download_filename=download_filename+".html",
        )
