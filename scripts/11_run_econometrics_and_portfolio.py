"""Entry point: statistical validation (Fama-MacBeth, robustness) and
systematic portfolio construction (rebalancing, weighting, turnover, risk).
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
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

FACTOR_COL = "n_unique_connected_rank" #brokerage_measure_rank
FORWARD_RETURN_COL = "forward_return"
CONTROL_COLS = ["mkvalt", "at", "sale"]  # available controls in merged panel
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
        logger.error("Column '%s' not found in panel. Merge stock return data "
                     "before running econometrics. Skipping.", FORWARD_RETURN_COL)
        return

    fm = fama_macbeth(df, FORWARD_RETURN_COL, [FACTOR_COL],
                       date_col=DATE_COL,
                       newey_west_lags=cfg["fama_macbeth"]["newey_west_lags"])
    logger.info("Fama-MacBeth results:\n%s", fm)

    available_controls = [c for c in CONTROL_COLS if c in df.columns]
    robustness = robustness_test(df, FORWARD_RETURN_COL, [FACTOR_COL],
                                  control_specs=[[], available_controls],
                                  date_col=DATE_COL)
    logger.info("Robustness across specifications:\n%s", robustness)

    oos = out_of_sample_evaluation(df, FORWARD_RETURN_COL, [FACTOR_COL],
                                    date_col=DATE_COL,
                                    train_frac=cfg["out_of_sample"]["train_frac"])
    logger.info("Out-of-sample evaluation: %s", oos)

    if cfg["weighting"]["scheme"] == "value" and "mkvalt" in df.columns:
        df["weight"] = value_weight_portfolio(df, mktcap_col="mkvalt",
                                                date_col=DATE_COL, entity_col=ENTITY_COL)
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

    out_dir = project_root() / "data" / "processed"
    fm.to_csv(out_dir / "fama_macbeth.csv")
    robustness.to_csv(out_dir / "robustness.csv", index=False)
    turnover.to_csv(out_dir / "turnover.csv")
    attribution.to_csv(out_dir / "performance_attribution.csv")
    logger.info("Econometrics/portfolio outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
