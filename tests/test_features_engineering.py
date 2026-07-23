import numpy as np
import pandas as pd

from ceo_sc.features.engineering import (
    cross_sectional_rank,
    industry_neutralize,
    lag_feature,
    market_cap_neutralize,
    validate_feature,
    winsorize,
    winsorize_cross_sectional,
    zscore,
    zscore_cross_sectional,
)


def test_winsorize_clips_extremes():
    s = pd.Series([1, 2, 3, 4, 5, 1000])
    w = winsorize(s, lower=0.1, upper=0.9)
    assert w.max() < 1000
    assert w.min() >= s.quantile(0.1)


def test_zscore_mean_zero_std_one():
    s = pd.Series(np.arange(100, dtype=float))
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1) < 1e-9


def test_zscore_constant_series_returns_nan():
    s = pd.Series([5.0] * 10)
    z = zscore(s)
    assert z.isna().all()


def test_winsorize_cross_sectional_per_date(synthetic_panel):
    out = winsorize_cross_sectional(synthetic_panel, "raw_factor")
    assert len(out) == len(synthetic_panel)
    assert out.notna().any()


def test_zscore_cross_sectional_per_date_mean_near_zero(synthetic_panel):
    out = zscore_cross_sectional(synthetic_panel, "raw_factor")
    per_date_mean = out.groupby(synthetic_panel["date"]).mean()
    assert np.allclose(per_date_mean, 0, atol=1e-8)


def test_industry_neutralize_demeans_within_group(synthetic_panel):
    out = industry_neutralize(synthetic_panel, "raw_factor")
    grouped_mean = out.groupby([synthetic_panel["date"], synthetic_panel["industry"]]).mean()
    assert np.allclose(grouped_mean, 0, atol=1e-8)


def test_market_cap_neutralize_runs_and_returns_same_length(synthetic_panel):
    out = market_cap_neutralize(synthetic_panel, "raw_factor")
    assert len(out) == len(synthetic_panel)


def test_lag_feature_shifts_within_entity(synthetic_panel):
    lagged = lag_feature(synthetic_panel, "raw_factor", periods=1)
    merged = synthetic_panel.assign(lagged=lagged).sort_values(["entity_id", "date"])
    first_per_entity = merged.groupby("entity_id").head(1)
    assert first_per_entity["lagged"].isna().all()


def test_cross_sectional_rank_is_percentile(synthetic_panel):
    ranks = cross_sectional_rank(synthetic_panel, "raw_factor", pct=True)
    assert ranks.min() > 0
    assert ranks.max() <= 1.0


def test_validate_feature_flags_high_nan_fraction():
    s = pd.Series([np.nan] * 90 + [1.0] * 10)
    report = validate_feature(s, max_nan_frac=0.5)
    assert report["passed"] is False
    assert report["nan_frac"] == 0.9
