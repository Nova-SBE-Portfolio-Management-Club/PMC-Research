#!/usr/bin/env python
# coding: utf-8

"""
Risk Metrics using Monte Carlo Simulations

This script: 
- runs a large number of Monte Carlo simulations over a selected period of time;
- prints the distribution of the different risk-related statistics across the various simulations.

This can be used to determine whether a specific portfolio composition offers the right amount of risk.
"""
__author__ = ['Nicholas Gorham', 'Tim Gajewski']
__copyright__ = 'Copyright 2022, PMC Quantitative Team'
__credits__ = ['Daniel Goncalves']
__license__ = 'PMC'
__maintainer__ = 'Quantitative Team PMC'
__email__ = 'portfoliomanagementclub@novasbe.pt'
__status__ = 'Production'


# --------------------------------------------------------------------------- #
#                                  Imports                                    #
# --------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import yfinance as yf

# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #


class MonteCarloRiskMetrics(object):
    """
    Brief Description

    Parameters
    ----------

    portfolio: iterable of tickers

    portfolio_value: float or int

    timeframe: int
        Nr of days to be forecasted. Defaults to 21.

    #TODO

    Attributes
    ----------


    Methods
    -------
    monte_carlo_simulation:
    """

    def __init__(
        self,
        portfolio,
        portfolio_value=1000000,
        nr_sims=20000,
        timeframe=21,
        start=None,
        end=None,
        distribution="normal",
        weights=None,
        confidence_level=0.95,
    ) -> None:

        self.nr_sims = nr_sims
        self.timeframe = timeframe
        self.portfolio = portfolio
        self.end = end if end is not None else dt.datetime.today()
        self.start = (
            start if start is not None else self.end -
            dt.timedelta(days=365 * 5)
        )
        self.distribution = distribution
        # Static, the distribution might be selectable in the future
        # https://numpy.org/doc/1.16/reference/routines.random.html

        if weights is None:
            weights = np.random.random(len(portfolio))
        self.weights = pd.Series((weights / np.sum(weights)), index=portfolio)
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level

    def get_data(self):
        """ Retrieves price data for the tickers input """
        stockData = yf.download(self.portfolio, start=self.start, end=self.end)[
            "Adj Close"
        ]
        returns = np.log1p(stockData.pct_change())
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        self.meanReturns = meanReturns
        self.covMatrix = covMatrix
        return meanReturns, covMatrix

    def monte_carlo_simulation(self):
        self.get_data()
        meanM = np.full(
            shape=(self.timeframe, len(self.weights)), fill_value=self.meanReturns
        )
        meanM = meanM.T
        portfolio_sims_raw = np.full(
            shape=(self.timeframe, self.nr_sims), fill_value=0.0
        )

        for m in range(self.nr_sims):
            Z = np.random.normal(size=(self.timeframe, len(self.weights)))
            L = np.linalg.cholesky(
                self.covMatrix
            )  # Account for correlations using the Cholesky Decomposition
            dailyReturns = meanM + np.inner(L, Z)
            portfolio_sims_raw[:, m] = (
                np.cumprod(np.inner(self.weights, dailyReturns.T) + 1)
                * self.portfolio_value
            )
        portfolio_sims_abs = pd.DataFrame(portfolio_sims_raw)
        portfolio_sims_abs.index += 1
        portfolio_sims_abs.iloc[0] = self.portfolio_value
        self.portfolio_sims_abs = portfolio_sims_abs
        return portfolio_sims_abs

    def plot_simulations(self):
        plt.figure(figsize=(15, 10))
        plt.style.use("seaborn-notebook")
        plt.plot(self.portfolio_sims_abs)
        plt.ylabel("Portfolio Value($)")
        plt.xlabel("Days")
        plt.title("MC Simulation of a portfolio")
        plt.show()

    def generate_stats(self):
        # Daily Returns
        portfolio_daily_returns = np.log1p(
            self.portfolio_sims_abs.pct_change())

        # Average Expected Return
        portfolio_expected_returns = portfolio_daily_returns.mean() * self.timeframe
        portfolio_expected_returns.name = 'Expected Return'

        # Standard Deviation
        portfolio_std = portfolio_daily_returns.std() * np.sqrt(self.timeframe)
        portfolio_std.name = 'Standard Deviation'

        # Max Drawdown
        dd = (self.portfolio_sims_abs /
              self.portfolio_sims_abs.expanding(min_periods=0).max() - 1).min()
        dd.name = 'Máx DrawDown'

        # Sharpe Ratio
        # rf = pdr.DataReader("DTB4WK", "fred", self.start, self.end).iloc[-1].values[0]/100
        rf = 0
        sharpe_ratio = (portfolio_expected_returns-rf)/portfolio_std
        sharpe_ratio.name = "Sharpe Ratio"

        # Extract the values for the selected percentile, i.e., each individual VaR
        VaR_df = portfolio_daily_returns.quantile(
            q=1 - self.confidence_level,  # Percentile
            axis=0,
            numeric_only=True,
            interpolation="linear",
        )
        VaR_df.name = "VaR"

        # CVaR
        cVaR_df = pd.Series(
            index=portfolio_daily_returns.columns, dtype='float64', name="cVaR")
        for ticker in portfolio_daily_returns.columns:
            cvar = portfolio_daily_returns[ticker].loc[portfolio_daily_returns[ticker] < VaR_df[ticker]].mean(
            )
            cVaR_df[ticker] = cvar

        stats = pd.concat([sharpe_ratio, portfolio_expected_returns,
                          portfolio_std, dd, VaR_df, cVaR_df], axis=1)
        self.stats = stats
        return stats

    def display_stats(self):
        # Iterate over self.stats, generate dinamically charts and descriptive tables
        # Pretty Plot Everything
        stats_summary = pd.DataFrame()
        for metric in self.stats.columns:
            metric_min = self.stats[metric].min()
            metric_mean = self.stats[metric].mean()
            metric_max = self.stats[metric].max()
            metric_std = self.stats[metric].std()

            stats_dict = {
                "Min": [metric_min],
                "Mean": [metric_mean],
                "Max": [metric_max],
                "Std dev": [metric_std],
            }
            stats_df = pd.DataFrame(stats_dict, index=[metric, ])
            stats_summary = pd.concat([stats_df, stats_summary])

            fig = sns.histplot(self.stats[metric], kde=True, legend=True)
            sns.despine()
            plt.title = "Distribution across the Monte Carlo Simulations"
            text_position_x = plt.gca().get_xbound(
            )[0]+0.05*abs(plt.gca().get_xbound()[0])
            if metric != "Sharpe Ratio":
                fig.text(text_position_x, 0.95*plt.gca().get_ybound()
                         [1], f"Min: {round(metric_min*100,2)}%")
                fig.text(text_position_x, 0.9*plt.gca().get_ybound()
                         [1], f"\u03bc: {round(metric_mean*100,2)}% ")
                fig.text(text_position_x, 0.85*plt.gca().get_ybound()
                         [1], f"Máx: {round(metric_max*100,2)}%")
                fig.text(text_position_x, 0.8*plt.gca().get_ybound()
                         [1], f"\u03C3: {round(metric_std*100,2)}%")
            else:
                fig.text(text_position_x, 0.95*plt.gca().get_ybound()
                         [1], f"Min: {round(metric_min,2)}")
                fig.text(text_position_x, 0.9*plt.gca().get_ybound()
                         [1], f"\u03bc: {round(metric_mean,2)} ")
                fig.text(text_position_x, 0.85*plt.gca().get_ybound()
                         [1], f"Máx: {round(metric_max,2)}")
                fig.text(text_position_x, 0.8*plt.gca().get_ybound()
                         [1], f"\u03C3: {round(metric_std,2)}")
            plt.show()
        self.stats_summary = stats_summary
        return stats_summary


# --------------------------------------------------------------------------- #
#                                Test Case                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    teste = MonteCarloRiskMetrics(
        ["SPY", "TLT", "TSLA", "XOP", "FXI", "XRT"], timeframe=21)
    teste = teste.monte_carlo_simulation()
    teste.plot_simulations()
    teste.generate_stats()
    teste.display_stats()
