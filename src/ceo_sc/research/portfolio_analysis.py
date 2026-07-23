"""Quantile portfolio analysis, long-short construction, and performance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def assign_quantiles(df: pd.DataFrame, factor_col: str, n_quantiles: int = 5,
                      date_col: str = "date") -> pd.Series:
    """Assign each entity to a cross-sectional quantile bucket (1..n_quantiles) of the factor."""
    def _bucket(group: pd.DataFrame) -> pd.Series:
        return pd.qcut(group[factor_col], n_quantiles, labels=False, duplicates="drop") + 1

    return df.groupby(date_col, group_keys=False).apply(_bucket, include_groups=False)


def quantile_portfolio_returns(df: pd.DataFrame, factor_col: str, return_col: str,
                                n_quantiles: int = 5, date_col: str = "date",
                                weight_col: str | None = None) -> pd.DataFrame:
    """Equal- or value-weighted mean forward return per quantile bucket, per date."""
    work = df.copy()
    work["_quantile"] = assign_quantiles(work, factor_col, n_quantiles, date_col)

    if weight_col:
        def _wavg(g: pd.DataFrame) -> float:
            w = g[weight_col].to_numpy(dtype=float)
            r = g[return_col].to_numpy(dtype=float)
            mask = ~(np.isnan(w) | np.isnan(r))
            if mask.sum() == 0 or w[mask].sum() == 0:
                return np.nan
            return float(np.average(r[mask], weights=w[mask]))

        result = work.groupby([date_col, "_quantile"]).apply(_wavg, include_groups=False).rename(return_col)
    else:
        result = work.groupby([date_col, "_quantile"])[return_col].mean()

    return result.unstack("_quantile")


def long_short_returns(quantile_returns: pd.DataFrame, long_quantile: int | None = None,
                        short_quantile: int = 1) -> pd.Series:
    """Long top quantile, short bottom quantile (default: highest vs. lowest)."""
    if long_quantile is None:
        long_quantile = quantile_returns.columns.max()
    return (quantile_returns[long_quantile] - quantile_returns[short_quantile]).rename("long_short")


def performance_stats(returns: pd.Series, periods_per_year: int = 12,
                       risk_free: float = 0.0) -> dict:
    """Risk-adjusted performance metrics for a return series."""
    clean = returns.dropna()
    if clean.empty:
        return {}
    excess = clean - risk_free / periods_per_year
    mean_ret = excess.mean()
    std_ret = excess.std(ddof=0)
    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    return {
        "annualized_return": mean_ret * periods_per_year,
        "annualized_vol": std_ret * np.sqrt(periods_per_year),
        "sharpe_ratio": (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret else np.nan,
        "max_drawdown": drawdown.min(),
        "cumulative_return": cumulative.iloc[-1] - 1,
        "n_periods": len(clean),
    }


def rolling_sharpe(returns: pd.Series, window: int = 12, periods_per_year: int = 12) -> pd.Series:
    mean_roll = returns.rolling(window).mean()
    std_roll = returns.rolling(window).std(ddof=0)
    return (mean_roll / std_roll * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def transaction_cost_analysis(weights: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Estimate turnover and transaction cost drag from a wide (date x entity) weight matrix.

    ``cost_bps`` is the one-way transaction cost in basis points.
    """
    turnover = weights.diff().abs().sum(axis=1)
    cost_drag = turnover * (cost_bps / 1e4)
    return pd.DataFrame({"turnover": turnover, "cost_drag": cost_drag})
