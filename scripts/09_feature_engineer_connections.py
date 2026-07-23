"""Entry point: feature-engineer the merged connection panel.

Reads ``merged_panel.parquet`` (from 09_merge_panel.py) which contains
n_unique_connected plus fundamentals (mkvalt) via the linking table.
Applies the same pipeline as ``06_feature_engineering.py``:

winsorization -> z-score -> market-cap neutralization -> lag -> rank.

Industry neutralization is skipped (SIC codes unavailable from WRDS).

Output: data/processed/connection_features.parquet
"""

import pandas as pd

from ceo_sc.features.engineering import (
    cross_sectional_rank,
    lag_feature,
    market_cap_neutralize,
    validate_feature,
    winsorize_cross_sectional,
    zscore_cross_sectional,
)
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

RAW_COL = "n_unique_connected"


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "features.yaml")
    out = panel.copy()

    col = RAW_COL
    if col not in out.columns:
        logger.error("Column '%s' not found in panel; aborting.", col)
        return out

    w = winsorize_cross_sectional(out, col, date_col="year",
                                  lower=cfg["winsorization"]["lower"],
                                  upper=cfg["winsorization"]["upper"])
    out[f"{col}_w"] = w

    out[f"{col}_z"] = zscore_cross_sectional(out.assign(**{col: w}), col, date_col="year")

    # Market-cap neutralization (only for rows with mkvalt).
    # Rows without mkvalt fall back to the z-scored value so they
    # don't become NaN and drop out of the rank.
    if "mkvalt" in out.columns:
        cap_neutral = market_cap_neutralize(
            out.assign(**{col: out[f"{col}_z"]}), col,
            mktcap_col="mkvalt", date_col="year",
        )
        out[f"{col}_cap_neutral"] = cap_neutral.fillna(out[f"{col}_z"])
        neutral_col = f"{col}_cap_neutral"
    else:
        logger.warning("mkvalt not in panel; skipping market-cap neutralization")
        neutral_col = f"{col}_z"

    out[f"{col}_lagged"] = lag_feature(
        out.assign(**{col: out[neutral_col]}), col,
        entity_col="dirid_starting", date_col="year",
        periods=cfg["lagging"]["periods"],
    )

    out[f"{col}_rank"] = cross_sectional_rank(
        out.assign(**{col: out[f"{col}_lagged"]}), col,
        date_col="year", pct=cfg["ranking"]["pct"],
    )
    validate_feature(out[f"{col}_rank"], name=f"{col}_rank",
                     max_nan_frac=cfg["validation"]["max_nan_frac"])

    return out


def main() -> None:
    in_path = project_root() / "data" / "processed" / "merged_panel.parquet"
    if not in_path.exists():
        logger.error("Missing input panel at %s. Run 09_merge_panel.py first.", in_path)
        return

    panel = pd.read_parquet(in_path)
    logger.info("Loaded connection panel: %d rows, %d unique CEOs",
                len(panel), panel["dirid_starting"].nunique())

    features = build_features(panel)

    out_path = project_root() / "data" / "processed" / "connection_features.parquet"
    features.to_parquet(out_path)
    logger.info("Saved engineered connection features to %s (%d rows)", out_path, len(features))


if __name__ == "__main__":
    main()
