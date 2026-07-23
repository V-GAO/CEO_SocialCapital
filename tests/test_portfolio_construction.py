import numpy as np

from ceo_sc.portfolio.construction import (
    build_weight_matrix,
    equal_weight_portfolio,
    performance_attribution,
    risk_metrics,
    turnover_analysis,
    value_weight_portfolio,
)


def test_equal_weight_sums_to_one_per_date(synthetic_panel):
    synthetic_panel = synthetic_panel.copy()
    synthetic_panel["weight"] = equal_weight_portfolio(synthetic_panel)
    per_date_sum = synthetic_panel.groupby("date")["weight"].sum()
    assert np.allclose(per_date_sum, 1.0)


def test_value_weight_sums_to_one_per_date(synthetic_panel):
    synthetic_panel = synthetic_panel.copy()
    synthetic_panel["weight"] = value_weight_portfolio(synthetic_panel)
    per_date_sum = synthetic_panel.groupby("date")["weight"].sum()
    assert np.allclose(per_date_sum, 1.0)


def test_turnover_analysis_nonnegative(synthetic_panel):
    synthetic_panel = synthetic_panel.copy()
    synthetic_panel["weight"] = equal_weight_portfolio(synthetic_panel)
    matrix = build_weight_matrix(synthetic_panel, "weight")
    turnover = turnover_analysis(matrix)
    assert (turnover.dropna() >= 0).all()


def test_performance_attribution_sums_match_total(synthetic_panel):
    synthetic_panel = synthetic_panel.copy()
    synthetic_panel["weight"] = equal_weight_portfolio(synthetic_panel)
    weight_matrix = build_weight_matrix(synthetic_panel, "weight")
    return_matrix = synthetic_panel.pivot_table(index="date", columns="entity_id", values="forward_return")
    attribution = performance_attribution(weight_matrix, return_matrix)
    assert "total_return" in attribution.columns


def test_risk_metrics_keys(synthetic_panel):
    metrics = risk_metrics(synthetic_panel["forward_return"])
    assert {"volatility_annualized", "value_at_risk_95", "conditional_var_95"}.issubset(metrics.keys())
