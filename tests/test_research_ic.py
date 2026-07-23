import numpy as np

from ceo_sc.research.ic import ic_summary, information_coefficient, rank_ic
from ceo_sc.research.portfolio_analysis import (
    assign_quantiles,
    long_short_returns,
    performance_stats,
    quantile_portfolio_returns,
)


def test_information_coefficient_positive_for_known_relationship(synthetic_panel):
    ic = information_coefficient(synthetic_panel, "raw_factor", "forward_return")
    assert ic.mean() > 0


def test_rank_ic_positive_for_known_relationship(synthetic_panel):
    ric = rank_ic(synthetic_panel, "raw_factor", "forward_return")
    assert ric.mean() > 0


def test_ic_summary_keys(synthetic_panel):
    ic = information_coefficient(synthetic_panel, "raw_factor", "forward_return")
    summary = ic_summary(ic)
    assert {"mean_ic", "std_ic", "icir", "pct_positive", "n_periods"}.issubset(summary.keys())


def test_assign_quantiles_within_range(synthetic_panel):
    buckets = assign_quantiles(synthetic_panel, "raw_factor", n_quantiles=5)
    assert buckets.dropna().between(1, 5).all()


def test_quantile_portfolio_returns_shape(synthetic_panel):
    q_returns = quantile_portfolio_returns(synthetic_panel, "raw_factor", "forward_return", n_quantiles=5)
    assert q_returns.shape[1] <= 5


def test_long_short_returns_positive_mean(synthetic_panel):
    q_returns = quantile_portfolio_returns(synthetic_panel, "raw_factor", "forward_return", n_quantiles=5)
    ls = long_short_returns(q_returns)
    assert ls.mean() > 0


def test_performance_stats_contains_sharpe(synthetic_panel):
    q_returns = quantile_portfolio_returns(synthetic_panel, "raw_factor", "forward_return", n_quantiles=5)
    ls = long_short_returns(q_returns)
    stats = performance_stats(ls)
    assert "sharpe_ratio" in stats
    assert not np.isnan(stats["sharpe_ratio"])
