"""Information Coefficient (IC), Rank IC, and factor decay analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def information_coefficient(df: pd.DataFrame, factor_col: str, forward_return_col: str,
                             date_col: str = "date") -> pd.Series:
    """Per-date Pearson correlation between factor and forward return."""
    def _ic(group: pd.DataFrame) -> float:
        valid = group[[factor_col, forward_return_col]].dropna()
        if len(valid) < 3:
            return np.nan
        return pearsonr(valid[factor_col], valid[forward_return_col])[0]

    return df.groupby(date_col).apply(_ic, include_groups=False).rename("ic")


def rank_ic(df: pd.DataFrame, factor_col: str, forward_return_col: str,
            date_col: str = "date") -> pd.Series:
    """Per-date Spearman rank correlation between factor and forward return."""
    def _rank_ic(group: pd.DataFrame) -> float:
        valid = group[[factor_col, forward_return_col]].dropna()
        if len(valid) < 3:
            return np.nan
        return spearmanr(valid[factor_col], valid[forward_return_col])[0]

    return df.groupby(date_col).apply(_rank_ic, include_groups=False).rename("rank_ic")


def ic_summary(ic_series: pd.Series) -> dict:
    """Summary statistics for an IC (or Rank IC) time series."""
    clean = ic_series.dropna()
    mean_ic = clean.mean()
    std_ic = clean.std(ddof=0)
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": mean_ic / std_ic if std_ic else np.nan,
        "pct_positive": (clean > 0).mean(),
        "n_periods": len(clean),
    }


def factor_decay(df: pd.DataFrame, factor_col: str, return_col: str, entity_col: str = "entity_id",
                  date_col: str = "date", max_lag: int = 12) -> pd.DataFrame:
    """Compute Rank IC of the factor against returns at horizons 1..max_lag periods ahead.

    Assumes ``return_col`` is the *single-period-ahead* forward return already
    aligned per entity; horizon-h forward returns are the compounded product
    of the next ``h`` single-period returns for that entity.
    """
    sorted_df = df.sort_values([entity_col, date_col]).copy()
    results = []
    for lag in range(1, max_lag + 1):
        fwd_col = f"_fwd_ret_{lag}"
        compounded = (
            sorted_df.groupby(entity_col)[return_col]
            .apply(lambda s: (1 + s).rolling(lag).apply(lambda w: w.prod() - 1, raw=True).shift(-(lag - 1)))
        )
        sorted_df[fwd_col] = compounded.reset_index(level=0, drop=True)
        ric = rank_ic(sorted_df, factor_col, fwd_col, date_col=date_col)
        summary = ic_summary(ric)
        summary["horizon"] = lag
        results.append(summary)
        sorted_df.drop(columns=[fwd_col], inplace=True)

    return pd.DataFrame(results).set_index("horizon")
