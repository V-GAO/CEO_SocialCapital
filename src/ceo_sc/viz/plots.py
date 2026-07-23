"""Research-quality visualizations for the CEO_SC pipeline.

Every function returns the created ``matplotlib`` Axes/Figure so callers
can further customize or save the plot; none of these call ``plt.show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


def plot_network(G: nx.Graph, ax: plt.Axes | None = None, node_size: int = 50,
                  seed: int = 42) -> plt.Axes:
    ax = ax or plt.gca()
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=node_size, with_labels=False,
                      edge_color="grey", alpha=0.7)
    ax.set_title("CEO Social Network")
    ax.axis("off")
    return ax


def plot_factor_distribution(series: pd.Series, ax: plt.Axes | None = None, bins: int = 50) -> plt.Axes:
    ax = ax or plt.gca()
    sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(f"Distribution of {series.name or 'factor'}")
    return ax


def plot_ic_timeseries(ic_series: pd.Series, rolling_window: int = 12,
                        ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ic_series.plot(ax=ax, alpha=0.4, label="IC")
    ic_series.rolling(rolling_window).mean().plot(ax=ax, label=f"{rolling_window}-period MA")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.set_title("Information Coefficient Over Time")
    return ax


def plot_cumulative_returns(returns: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    (1 + returns.fillna(0)).cumprod().plot(ax=ax)
    ax.set_title("Cumulative Return")
    ax.set_ylabel("Growth of $1")
    return ax


def plot_rolling_sharpe(rolling_sharpe: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    rolling_sharpe.plot(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling Sharpe Ratio")
    return ax


def plot_turnover(turnover: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    turnover.plot(ax=ax, kind="bar")
    ax.set_title("Portfolio Turnover")
    ax.set_ylabel("Turnover")
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    return ax


def plot_risk_attribution(contributions: pd.DataFrame, top_n: int = 15,
                           ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    mean_contrib = contributions.drop(columns=["total_return"], errors="ignore").mean().sort_values()
    mean_contrib.tail(top_n).plot(ax=ax, kind="barh")
    ax.set_title("Average Return Contribution by Entity (Top {})".format(top_n))
    return ax
