"""Systematic portfolio construction: rebalancing, weighting, turnover, attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def monthly_rebalance_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """Return the last available trading date of each calendar month."""
    ts = pd.to_datetime(pd.Series(dates).unique())
    df = pd.DataFrame({"date": ts})
    df["ym"] = df["date"].dt.to_period("M")
    return pd.DatetimeIndex(df.groupby("ym")["date"].max().sort_values())


def equal_weight_portfolio(df: pd.DataFrame, entity_col: str = "entity_id",
                            date_col: str = "date") -> pd.Series:
    """Assign equal weight 1/N to every entity held on each date."""
    counts = df.groupby(date_col)[entity_col].transform("count")
    return (1.0 / counts).rename("weight")


def value_weight_portfolio(df: pd.DataFrame, mktcap_col: str = "market_cap",
                            date_col: str = "date") -> pd.Series:
    """Assign weight proportional to market cap within each date."""
    total = df.groupby(date_col)[mktcap_col].transform("sum")
    return (df[mktcap_col] / total).rename("weight")


def build_weight_matrix(df: pd.DataFrame, weight_col: str, entity_col: str = "entity_id",
                         date_col: str = "date") -> pd.DataFrame:
    """Pivot long-format (date, entity, weight) into a wide date x entity weight matrix."""
    return df.pivot_table(index=date_col, columns=entity_col, values=weight_col, fill_value=0.0)


def turnover_analysis(weight_matrix: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance date: 0.5 * sum(|w_t - w_{t-1}|)."""
    return (weight_matrix.diff().abs().sum(axis=1) * 0.5).rename("turnover")


def performance_attribution(weight_matrix: pd.DataFrame, return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Decompose total portfolio return into per-entity contributions on each date.

    Both inputs must share the same date index and entity columns.
    """
    common_cols = weight_matrix.columns.intersection(return_matrix.columns)
    contrib = weight_matrix[common_cols].shift(1) * return_matrix[common_cols]
    contrib["total_return"] = contrib.sum(axis=1)
    return contrib


def risk_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Basic risk metrics: volatility, downside deviation, VaR, CVaR."""
    clean = returns.dropna()
    if clean.empty:
        return {}
    downside = clean[clean < 0]
    var_95 = clean.quantile(0.05)
    cvar_95 = clean[clean <= var_95].mean() if (clean <= var_95).any() else np.nan
    return {
        "volatility_annualized": clean.std(ddof=0) * np.sqrt(periods_per_year),
        "downside_deviation_annualized": downside.std(ddof=0) * np.sqrt(periods_per_year) if not downside.empty else 0.0,
        "value_at_risk_95": var_95,
        "conditional_var_95": cvar_95,
    }
