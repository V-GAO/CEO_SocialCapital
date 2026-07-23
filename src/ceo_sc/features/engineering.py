"""Feature engineering utilities for alpha factor construction.

All functions operate on long-format panel DataFrames with at least
``date`` and ``entity_id`` columns, and are cross-sectional (applied
independently within each ``date`` group) unless noted otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series to its [lower, upper] quantiles."""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lower=lo, upper=hi)


def winsorize_cross_sectional(df: pd.DataFrame, col: str, date_col: str = "date",
                               lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    return df.groupby(date_col)[col].transform(lambda s: winsorize(s, lower, upper))


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a series to zero mean, unit variance."""
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def zscore_cross_sectional(df: pd.DataFrame, col: str, date_col: str = "date") -> pd.Series:
    return df.groupby(date_col)[col].transform(zscore)


def industry_neutralize(df: pd.DataFrame, col: str, industry_col: str = "industry",
                         date_col: str = "date") -> pd.Series:
    """Demean a feature within each (date, industry) group."""
    return df.groupby([date_col, industry_col])[col].transform(lambda s: s - s.mean())


def market_cap_neutralize(df: pd.DataFrame, col: str, mktcap_col: str = "market_cap",
                           date_col: str = "date") -> pd.Series:
    """Neutralize a feature against log market cap via cross-sectional OLS residuals."""
    def _residualize(group: pd.DataFrame) -> pd.Series:
        y = group[col].to_numpy(dtype=float)
        x = np.log(group[mktcap_col].to_numpy(dtype=float).clip(min=1e-9))
        mask = ~(np.isnan(y) | np.isnan(x))
        resid = pd.Series(np.nan, index=group.index)
        if mask.sum() < 2:
            return resid
        x_design = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, *_ = np.linalg.lstsq(x_design, y[mask], rcond=None)
        fitted = x_design @ beta
        resid.iloc[np.where(mask)[0]] = y[mask] - fitted
        return resid

    return df.groupby(date_col, group_keys=False).apply(_residualize, include_groups=False)


def lag_feature(df: pd.DataFrame, col: str, periods: int = 1, entity_col: str = "entity_id",
                 date_col: str = "date") -> pd.Series:
    """Lag a feature by ``periods`` within each entity, sorted by date, to prevent look-ahead bias."""
    sorted_df = df.sort_values([entity_col, date_col])
    lagged = sorted_df.groupby(entity_col)[col].shift(periods)
    return lagged.reindex(df.index)


def cross_sectional_rank(df: pd.DataFrame, col: str, date_col: str = "date",
                          pct: bool = True) -> pd.Series:
    """Rank a feature cross-sectionally within each date (percentile rank by default)."""
    return df.groupby(date_col)[col].rank(pct=pct)


def validate_feature(series: pd.Series, name: str = "feature", max_nan_frac: float = 0.5) -> dict:
    """Run basic sanity checks on a constructed feature and return a report dict."""
    n = len(series)
    n_nan = series.isna().sum()
    nan_frac = n_nan / n if n else float("nan")
    report = {
        "name": name,
        "n_obs": n,
        "n_nan": int(n_nan),
        "nan_frac": nan_frac,
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "n_unique": series.nunique(dropna=True),
        "passed": bool(nan_frac <= max_nan_frac and series.nunique(dropna=True) > 1),
    }
    if not report["passed"]:
        logger.warning("Feature validation FAILED for '%s': %s", name, report)
    else:
        logger.info("Feature validation passed for '%s' (nan_frac=%.3f)", name, nan_frac)
    return report
