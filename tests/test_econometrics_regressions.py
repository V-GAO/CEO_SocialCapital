from ceo_sc.econometrics.regressions import (
    cross_sectional_regression,
    fama_macbeth,
    out_of_sample_evaluation,
    significance_test,
)


def test_cross_sectional_regression_has_expected_columns(synthetic_panel):
    result = cross_sectional_regression(synthetic_panel, "forward_return", ["raw_factor"])
    assert "coef_raw_factor" in result.columns
    assert "tstat_raw_factor" in result.columns
    assert len(result) > 0


def test_fama_macbeth_detects_known_positive_relationship(synthetic_panel):
    fm = fama_macbeth(synthetic_panel, "forward_return", ["raw_factor"])
    assert "raw_factor" in fm.index
    assert fm.loc["raw_factor", "mean"] > 0


def test_out_of_sample_evaluation_returns_r2(synthetic_panel):
    result = out_of_sample_evaluation(synthetic_panel, "forward_return", ["raw_factor"])
    assert "out_of_sample_r2" in result
    assert "in_sample_r2" in result


def test_significance_test_flags_significant_result():
    result = significance_test(coef_mean=0.05, std_error=0.01, n_obs=100)
    assert result["significant"] is True

    result_insignificant = significance_test(coef_mean=0.001, std_error=0.05, n_obs=100)
    assert result_insignificant["significant"] is False
