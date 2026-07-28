"""Entry point: statistical validation (Fama-MacBeth, robustness) and
systematic portfolio construction (rebalancing, weighting, turnover, risk).

Runs Fama-MacBeth / robustness / out-of-sample evaluation for *every*
social-capital metric engineered by 09_feature_engineer_connections.py
(all "*_rank" columns in the panel), plus a multivariate specification
regressing all factors together (to see which remain significant once
controlling for the others -- these metrics are likely correlated, e.g.
degree_centrality vs. n_unique_connected).

The portfolio construction section (weighting/turnover/attribution/risk)
is factor-agnostic -- it builds a whole-universe equal- or value-weighted
benchmark, independent of which social-capital metric is used as a signal
(factor-based long/short portfolios are handled separately in
10_run_research.py's quantile analysis).

Additionally, for *every* factor, builds and evaluates a long-only
portfolio holding just the top quantile of that factor each year (see
``_run_long_only_portfolio``) -- turnover, return, Sharpe, and VaR/CVaR are
saved per-factor and summarized in ``long_only_comparison.csv``, directly
comparable to the whole-universe benchmark above since both use the same
systematic portfolio-construction machinery.
"""

import pandas as pd

from ceo_sc.econometrics.regressions import fama_macbeth, out_of_sample_evaluation, robustness_test
from ceo_sc.portfolio.construction import (
    build_weight_matrix,
    equal_weight_portfolio,
    performance_attribution,
    risk_metrics,
    turnover_analysis,
    value_weight_portfolio,
)
from ceo_sc.research.portfolio_analysis import performance_stats
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Primary factor used for the backward-compatible, unprefixed output
# filenames (fama_macbeth.csv, robustness.csv). All *_rank factors are
# still evaluated -- see main().
PRIMARY_FACTOR_COL = "n_unique_connected_rank"
FORWARD_RETURN_COL = "forward_return"
CONTROL_COLS = ["mkvalt", "at", "sale"]  # available controls in merged panel
DATE_COL = "year"
ENTITY_COL = "dirid_starting"


def _run_factor(df: pd.DataFrame, factor_col: str, available_controls: list[str],
                 cfg: dict, out_dir, prefix: str) -> dict:
    """Run Fama-MacBeth, robustness, and out-of-sample evaluation for a
    single factor, save its outputs, and return a summary row."""
    fm = fama_macbeth(df, FORWARD_RETURN_COL, [factor_col],
                       date_col=DATE_COL,
                       newey_west_lags=cfg["fama_macbeth"]["newey_west_lags"])
    logger.info("[%s] Fama-MacBeth results:\n%s", factor_col, fm)

    robustness = robustness_test(df, FORWARD_RETURN_COL, [factor_col],
                                  control_specs=[[], available_controls],
                                  date_col=DATE_COL)
    logger.info("[%s] Robustness across specifications:\n%s", factor_col, robustness)

    oos = out_of_sample_evaluation(df, FORWARD_RETURN_COL, [factor_col],
                                    date_col=DATE_COL,
                                    train_frac=cfg["out_of_sample"]["train_frac"])
    logger.info("[%s] Out-of-sample evaluation: %s", factor_col, oos)

    fm.to_csv(out_dir / f"{prefix}fama_macbeth.csv")
    robustness.to_csv(out_dir / f"{prefix}robustness.csv", index=False)

    row = {"factor": factor_col, "out_of_sample_r2": oos.get("out_of_sample_r2")}
    if factor_col in fm.index:
        row.update({
            "mean_coef": fm.loc[factor_col, "mean"],
            "t_stat": fm.loc[factor_col, "t_stat"],
            "p_value": fm.loc[factor_col, "p_value"],
        })
    return row


def _run_long_only_portfolio(df: pd.DataFrame, factor_col: str, cfg: dict, out_dir, prefix: str) -> dict:
    """Build a long-only portfolio holding the top quantile of ``factor_col``
    each year (weighted per ``cfg["weighting"]["scheme"]``), and evaluate its
    turnover/return/risk -- using the same systematic portfolio-construction
    machinery (build_weight_matrix/turnover_analysis/performance_attribution/
    risk_metrics) as the whole-universe benchmark below, so results are
    directly comparable to it.

    ``factor_col`` is a cross-sectional percentile rank in (0, 1], so the top
    quantile threshold is ``(long_quantile - 1) / n_quantiles`` (e.g. 0.8 for
    the top quintile of 5).
    """
    n_quantiles = cfg["quantiles"]["n_quantiles"]
    long_quantile = cfg["quantiles"]["long_quantile"]
    threshold = (long_quantile - 1) / n_quantiles

    top = df[df[factor_col] > threshold].copy()
    if top.empty:
        logger.warning("[%s] No rows above top-quantile threshold %.2f; skipping long-only portfolio.",
                        factor_col, threshold)
        return {"factor": factor_col}

    if cfg["weighting"]["scheme"] == "value" and "mkvalt" in top.columns:
        top["weight"] = value_weight_portfolio(top, mktcap_col="mkvalt", date_col=DATE_COL)
    else:
        top["weight"] = equal_weight_portfolio(top, date_col=DATE_COL, entity_col=ENTITY_COL)

    weight_matrix = build_weight_matrix(top, "weight", entity_col=ENTITY_COL, date_col=DATE_COL)
    turnover = turnover_analysis(weight_matrix)

    return_matrix = top.pivot_table(index=DATE_COL, columns=ENTITY_COL, values=FORWARD_RETURN_COL)
    attribution = performance_attribution(weight_matrix, return_matrix)
    stats = performance_stats(attribution["total_return"],
                               periods_per_year=cfg["performance"]["periods_per_year"],
                               risk_free=cfg["performance"]["risk_free_rate"])
    risk = risk_metrics(attribution["total_return"], periods_per_year=cfg["performance"]["periods_per_year"])
    logger.info("[%s] Long-only (top quantile) performance: %s | risk: %s", factor_col, stats, risk)

    attribution.to_csv(out_dir / f"{prefix}long_only_attribution.csv")
    turnover.to_csv(out_dir / f"{prefix}long_only_turnover.csv")

    return {
        "factor": factor_col,
        "avg_turnover": turnover.mean(),
        "n_holdings_avg": (weight_matrix > 0).sum(axis=1).mean(),
        **stats,
        **risk,
    }


def main() -> None:
    features_path = project_root() / "data" / "processed" / "panel_with_returns.parquet"
    if not features_path.exists():
        logger.error("Missing panel at %s. Run 08_merge_returns.py first.", features_path)
        return

    df = pd.read_parquet(features_path)
    cfg = load_config(configs_dir() / "backtest.yaml")

    if FORWARD_RETURN_COL not in df.columns:
        logger.error("Column '%s' not found in panel. Merge stock return data "
                     "before running econometrics. Skipping.", FORWARD_RETURN_COL)
        return

    factor_cols = sorted(c for c in df.columns if c.endswith("_rank"))
    if not factor_cols:
        logger.error("No '*_rank' factor columns found in panel. Run "
                     "09_feature_engineer_connections.py first.")
        return
    logger.info("Running econometrics for %d factor(s): %s", len(factor_cols), factor_cols)

    out_dir = project_root() / "data" / "processed"
    available_controls = [c for c in CONTROL_COLS if c in df.columns]

    summary_rows = []
    long_only_rows = []
    for factor_col in factor_cols:
        prefix = "" if factor_col == PRIMARY_FACTOR_COL else f"{factor_col}_"
        summary_rows.append(_run_factor(df, factor_col, available_controls, cfg, out_dir, prefix))
        long_only_rows.append(_run_long_only_portfolio(df, factor_col, cfg, out_dir, prefix))

    pd.DataFrame(summary_rows).set_index("factor").to_csv(out_dir / "factor_regression_comparison.csv")
    pd.DataFrame(long_only_rows).set_index("factor").to_csv(out_dir / "long_only_comparison.csv")

    # Multivariate spec: all factors together, to see which remain
    # significant once controlling for the others (these social-capital
    # metrics are likely correlated with each other).
    if len(factor_cols) > 1:
        multivariate = fama_macbeth(df, FORWARD_RETURN_COL, factor_cols,
                                     date_col=DATE_COL,
                                     newey_west_lags=cfg["fama_macbeth"]["newey_west_lags"])
        logger.info("Multivariate Fama-MacBeth (all factors):\n%s", multivariate)
        multivariate.to_csv(out_dir / "fama_macbeth_multivariate.csv")

    if cfg["weighting"]["scheme"] == "value" and "mkvalt" in df.columns:
        df["weight"] = value_weight_portfolio(df, mktcap_col="mkvalt", date_col=DATE_COL)
    else:
        df["weight"] = equal_weight_portfolio(df, date_col=DATE_COL, entity_col=ENTITY_COL)

    weight_matrix = build_weight_matrix(df, "weight", entity_col=ENTITY_COL, date_col=DATE_COL)
    turnover = turnover_analysis(weight_matrix)
    logger.info("Average turnover: %.4f", turnover.mean())

    return_matrix = df.pivot_table(index=DATE_COL, columns=ENTITY_COL, values=FORWARD_RETURN_COL)
    attribution = performance_attribution(weight_matrix, return_matrix)
    risk = risk_metrics(attribution["total_return"],
                         periods_per_year=cfg["performance"]["periods_per_year"])
    logger.info("Risk metrics: %s", risk)

    turnover.to_csv(out_dir / "turnover.csv")
    attribution.to_csv(out_dir / "performance_attribution.csv")
    logger.info("Econometrics/portfolio outputs for %d factor(s) saved to %s", len(factor_cols), out_dir)


if __name__ == "__main__":
    main()
