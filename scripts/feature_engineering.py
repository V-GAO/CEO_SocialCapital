"""Entry point: transform raw network metrics into investable alpha features.

Applies winsorization -> standardization -> industry/market-cap
neutralization -> lag -> cross-sectional rank, per `configs/features.yaml`.
"""

import pandas as pd

from ceo_sc.features.engineering import (
    cross_sectional_rank,
    industry_neutralize,
    lag_feature,
    market_cap_neutralize,
    validate_feature,
    winsorize_cross_sectional,
    zscore_cross_sectional,
)
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

RAW_METRIC_COLS = [
    "degree_centrality",
    "eigenvector_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "pagerank",
    "structural_holes_effective_size",
    "network_constraint",
    "brokerage_measure",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "features.yaml")
    out = panel.copy()

    for col in RAW_METRIC_COLS:
        if col not in out.columns:
            continue
        w = winsorize_cross_sectional(out, col, lower=cfg["winsorization"]["lower"],
                                       upper=cfg["winsorization"]["upper"])
        out[f"{col}_w"] = w
        out[f"{col}_z"] = zscore_cross_sectional(out.assign(**{col: w}), col)
        out[f"{col}_ind_neutral"] = industry_neutralize(
            out.assign(**{col: out[f"{col}_z"]}), col, industry_col=cfg["neutralization"]["industry_col"]
        )
        out[f"{col}_cap_neutral"] = market_cap_neutralize(
            out.assign(**{col: out[f"{col}_ind_neutral"]}), col,
            mktcap_col=cfg["neutralization"]["market_cap_col"]
        )
        out[f"{col}_lagged"] = lag_feature(
            out.assign(**{col: out[f"{col}_cap_neutral"]}), col, periods=cfg["lagging"]["periods"]
        )
        out[f"{col}_rank"] = cross_sectional_rank(
            out.assign(**{col: out[f"{col}_lagged"]}), col, pct=cfg["ranking"]["pct"]
        )
        validate_feature(out[f"{col}_rank"], name=f"{col}_rank", max_nan_frac=cfg["validation"]["max_nan_frac"])

    return out


def main() -> None:
    in_path = project_root() / "data" / "processed" / "panel_with_network_metrics.parquet"
    if not in_path.exists():
        logger.error("Missing input panel at %s. Build it by merging fundamentals + "
                     "network_metrics + linking_table first.", in_path)
        return

    panel = pd.read_parquet(in_path)
    features = build_features(panel)

    out_path = project_root() / "data" / "processed" / "features.parquet"
    features.to_parquet(out_path)
    logger.info("Saved engineered features to %s", out_path)


if __name__ == "__main__":
    main()
