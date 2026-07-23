"""Entry point: run the end-to-end alpha research workflow on engineered features.

IC / Rank IC -> factor decay -> quantile portfolios -> long-short -> performance.
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

FACTOR_COL = "n_unique_connected_rank"
FORWARD_RETURN_COL = "forward_return"
DATE_COL = "year"
ENTITY_COL = "dirid_starting"


def main() -> None:
    features_path = project_root() / "data" / "processed" / "panel_with_returns.parquet"
    if not features_path.exists():
        logger.error("Missing panel at %s. Run 10_merge_returns.py first.", features_path)
        return

    df = pd.read_parquet(features_path)
    cfg = load_config(configs_dir() / "backtest.yaml")

    if FORWARD_RETURN_COL not in df.columns:
        logger.error("Column '%s' not found. Run 10_merge_returns.py first.", FORWARD_RETURN_COL)
        return

    ic = information_coefficient(df, FACTOR_COL, FORWARD_RETURN_COL, date_col=DATE_COL)
    ric = rank_ic(df, FACTOR_COL, FORWARD_RETURN_COL, date_col=DATE_COL)
    logger.info("IC summary: %s", ic_summary(ic))
    logger.info("Rank IC summary: %s", ic_summary(ric))

    decay = factor_decay(df, FACTOR_COL, FORWARD_RETURN_COL,
                         entity_col=ENTITY_COL, date_col=DATE_COL, max_lag=4)
    logger.info("Factor decay:\n%s", decay)

    quantiles = quantile_portfolio_returns(
        df, FACTOR_COL, FORWARD_RETURN_COL,
        n_quantiles=cfg["quantiles"]["n_quantiles"], date_col=DATE_COL
    )
    long_short = long_short_returns(
        quantiles, long_quantile=cfg["quantiles"]["long_quantile"],
        short_quantile=cfg["quantiles"]["short_quantile"],
    )
    stats = performance_stats(long_short, periods_per_year=cfg["performance"]["periods_per_year"],
                               risk_free=cfg["performance"]["risk_free_rate"])
    logger.info("Long-short performance: %s", stats)

    out_dir = project_root() / "data" / "processed"
    ic.to_frame().join(ric).to_csv(out_dir / "ic_timeseries.csv")
    quantiles.to_csv(out_dir / "quantile_returns.csv")
    long_short.to_csv(out_dir / "long_short_returns.csv")
    rolling_sharpe(long_short, window=cfg["performance"]["rolling_sharpe_window"],
                    periods_per_year=cfg["performance"]["periods_per_year"]).to_csv(
        out_dir / "rolling_sharpe.csv"
    )
    logger.info("Research outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
