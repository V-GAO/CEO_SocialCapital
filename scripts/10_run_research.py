"""Entry point: run the end-to-end alpha research workflow on engineered features.

IC / Rank IC -> factor decay -> quantile portfolios -> long-short -> performance.

Runs the full workflow for *every* social-capital metric engineered by
``09_feature_engineer_connections.py`` (all ``*_rank`` columns in the panel --
network size, prominence, and structural-position/brokerage factors), not
just a single hardcoded factor. The primary factor (``PRIMARY_FACTOR_COL``)
additionally gets the original unprefixed output filenames for backward
compatibility with ``viz/research_summary.ipynb``; every factor (including
the primary one) also gets its own ``{factor}_...csv`` files, and a
cross-factor ``factor_comparison.csv`` summarizes IC/Sharpe side by side.
"""

import pandas as pd

from ceo_sc.research.ic import factor_decay, ic_summary, information_coefficient, rank_ic
from ceo_sc.research.portfolio_analysis import (
    long_short_returns,
    performance_stats,
    quantile_portfolio_returns,
    rolling_sharpe,
)
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Factor used for the backward-compatible, unprefixed output filenames
# (ic_timeseries.csv, quantile_returns.csv, ...) consumed by
# viz/research_summary.ipynb. All *_rank factors are still researched and
# saved with a factor-specific prefix -- see main().
PRIMARY_FACTOR_COL = "n_unique_connected_rank"
FORWARD_RETURN_COL = "forward_return"
DATE_COL = "year"
ENTITY_COL = "dirid_starting"


def _run_factor(df: pd.DataFrame, factor_col: str, cfg: dict, out_dir, prefix: str) -> dict:
    """Run the IC -> decay -> quantile -> long-short -> performance workflow
    for a single factor, save its outputs, and return a summary row."""
    ic = information_coefficient(df, factor_col, FORWARD_RETURN_COL, date_col=DATE_COL)
    ric = rank_ic(df, factor_col, FORWARD_RETURN_COL, date_col=DATE_COL)
    ic_stats = ic_summary(ic)
    ric_stats = ic_summary(ric)
    logger.info("[%s] IC summary: %s", factor_col, ic_stats)
    logger.info("[%s] Rank IC summary: %s", factor_col, ric_stats)

    decay = factor_decay(df, factor_col, FORWARD_RETURN_COL,
                         entity_col=ENTITY_COL, date_col=DATE_COL, max_lag=4)

    quantiles = quantile_portfolio_returns(
        df, factor_col, FORWARD_RETURN_COL,
        n_quantiles=cfg["quantiles"]["n_quantiles"], date_col=DATE_COL
    )
    long_short = long_short_returns(
        quantiles, long_quantile=cfg["quantiles"]["long_quantile"],
        short_quantile=cfg["quantiles"]["short_quantile"],
    )
    stats = performance_stats(long_short, periods_per_year=cfg["performance"]["periods_per_year"],
                               risk_free=cfg["performance"]["risk_free_rate"])
    logger.info("[%s] Long-short performance: %s", factor_col, stats)

    ic.to_frame().join(ric).to_csv(out_dir / f"{prefix}ic_timeseries.csv")
    decay.to_csv(out_dir / f"{prefix}factor_decay.csv")
    quantiles.to_csv(out_dir / f"{prefix}quantile_returns.csv")
    long_short.to_csv(out_dir / f"{prefix}long_short_returns.csv")
    rolling_sharpe(long_short, window=cfg["performance"]["rolling_sharpe_window"],
                    periods_per_year=cfg["performance"]["periods_per_year"]).to_csv(
        out_dir / f"{prefix}rolling_sharpe.csv"
    )

    return {
        "factor": factor_col,
        "mean_ic": ic_stats["mean_ic"],
        "mean_rank_ic": ric_stats["mean_ic"],
        "rank_icir": ric_stats["icir"],
        "long_short_sharpe": stats.get("sharpe_ratio"),
        "long_short_ann_return": stats.get("annualized_return"),
        "long_short_max_drawdown": stats.get("max_drawdown"),
    }


def main() -> None:
    features_path = project_root() / "data" / "processed" / "panel_with_returns.parquet"
    if not features_path.exists():
        logger.error("Missing panel at %s. Run 08_merge_returns.py first.", features_path)
        return

    df = pd.read_parquet(features_path)
    cfg = load_config(configs_dir() / "backtest.yaml")

    if FORWARD_RETURN_COL not in df.columns:
        logger.error("Column '%s' not found. Run 08_merge_returns.py first.", FORWARD_RETURN_COL)
        return

    factor_cols = sorted(c for c in df.columns if c.endswith("_rank"))
    if not factor_cols:
        logger.error("No '*_rank' factor columns found in panel. Run "
                     "09_feature_engineer_connections.py first.")
        return
    logger.info("Running research for %d factor(s): %s", len(factor_cols), factor_cols)

    out_dir = project_root() / "data" / "processed"
    summary_rows = []
    for factor_col in factor_cols:
        prefix = "" if factor_col == PRIMARY_FACTOR_COL else f"{factor_col}_"
        summary_rows.append(_run_factor(df, factor_col, cfg, out_dir, prefix))

    pd.DataFrame(summary_rows).set_index("factor").to_csv(out_dir / "factor_comparison.csv")
    logger.info("Research outputs for %d factor(s) saved to %s", len(factor_cols), out_dir)


if __name__ == "__main__":
    main()
