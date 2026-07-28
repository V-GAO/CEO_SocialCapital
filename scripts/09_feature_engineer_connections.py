"""Entry point: feature-engineer the merged connection panel.

Reads ``merged_panel.parquet`` (from 07_merge_panel.py) which contains
n_unique_connected plus fundamentals (mkvalt) via the linking table, and
merges in the richer point-in-time network metrics panel (from
06_build_network_panel.py) -- degree/eigenvector centrality, PageRank,
Burt's structural holes / constraint, and a brokerage proxy -- so that
CEO social capital is captured along multiple dimensions (network size,
prominence, and structural position), not just raw connection count.

Both the "cumulative" convention (a connection persists once formed, even
after the underlying role ends -- all social capital accrued up to that
year) and the "active" convention (only connections still live in that
year) are merged in as parallel, separately-ranked factor sets, since
06_build_network_panel.py already computes both.

For each raw metric, applies the same pipeline as ``06_feature_engineering.py``:

log-transform (for heavy-tailed count/ratio metrics) -> winsorization ->
z-score -> industry neutralization -> market-cap neutralization -> lag -> rank.

Industry is defined as the first 2 digits of the SIC code (``siccd``,
merged in from ``data/raw/backtest_returns.parquet`` via ``permco``).

Output: data/processed/connection_features.parquet
"""

import numpy as np
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

# Point-in-time network metrics computed by 06_build_network_panel.py
# (cumulative: connection persists once formed; active: only currently
# live connections). n_unique_connected has no "active" counterpart --
# it's only ever built under the cumulative convention.
NETWORK_METRIC_COLS = [
    "degree_centrality",
    "eigenvector_centrality",
    "pagerank",
    "structural_holes_effective_size",
    "network_constraint",
    "brokerage_measure",
]

# CEO social capital, along multiple dimensions rather than just raw
# connection count: network size (n_unique_connected, degree_centrality),
# prominence (eigenvector_centrality, pagerank), and structural position /
# brokerage opportunity (structural_holes_effective_size,
# network_constraint, brokerage_measure) -- each under both the
# cumulative (all-time) and active (currently-live) conventions.
RAW_COLS = (
    ["n_unique_connected"]
    + NETWORK_METRIC_COLS
    + [f"{c}_active" for c in NETWORK_METRIC_COLS]
)

# Heavy-tailed / power-law-like metrics (raw counts, PageRank probability
# mass, and the brokerage reciprocal 1/constraint) get a log1p transform
# before winsorize/z-score, matching the log(mktcap) treatment already
# used in market_cap_neutralize.
LOG_TRANSFORM_COLS = {
    "n_unique_connected",
    "pagerank", "pagerank_active",
    "structural_holes_effective_size", "structural_holes_effective_size_active",
    "brokerage_measure", "brokerage_measure_active",
}

NETWORK_METRICS_CUMULATIVE_PATH = "data/processed/network_metrics_cumulative.parquet"
NETWORK_METRICS_ACTIVE_PATH = "data/processed/network_metrics_active.parquet"


def _engineer_one(out: pd.DataFrame, col: str, cfg: dict) -> pd.DataFrame:
    """Apply [log-transform ->] winsorize -> z-score -> industry-neutralize -> cap-neutralize -> lag -> rank
    to a single raw column."""
    if col in LOG_TRANSFORM_COLS:
        input_col = f"{col}_log"
        out[input_col] = np.log1p(out[col].clip(lower=0))
    else:
        input_col = col

    w = winsorize_cross_sectional(out, input_col, date_col="year",
                                  lower=cfg["winsorization"]["lower"],
                                  upper=cfg["winsorization"]["upper"])
    out[f"{col}_w"] = w

    out[f"{col}_z"] = zscore_cross_sectional(out.assign(**{col: w}), col, date_col="year")

    # Industry neutralization (demean within year x industry, using the
    # first 2 digits of the SIC code as industry_id). Rows without an
    # industry_id fall back to the z-scored value so they don't become
    # NaN and drop out of the rank.
    if "industry_id" in out.columns and out["industry_id"].notna().any():
        industry_neutral = industry_neutralize(
            out.assign(**{col: out[f"{col}_z"]}), col,
            industry_col="industry_id", date_col="year",
        )
        out[f"{col}_ind_neutral"] = industry_neutral.fillna(out[f"{col}_z"])
        base_col = f"{col}_ind_neutral"
    else:
        logger.warning("industry_id not in panel; skipping industry neutralization for '%s'", col)
        base_col = f"{col}_z"

    # Market-cap neutralization (only for rows with mkvalt).
    # Rows without mkvalt fall back to the industry-neutralized value so
    # they don't become NaN and drop out of the rank.
    if "mkvalt" in out.columns:
        cap_neutral = market_cap_neutralize(
            out.assign(**{col: out[base_col]}), col,
            mktcap_col="mkvalt", date_col="year",
        )
        out[f"{col}_cap_neutral"] = cap_neutral.fillna(out[base_col])
        neutral_col = f"{col}_cap_neutral"
    else:
        logger.warning("mkvalt not in panel; skipping market-cap neutralization for '%s'", col)
        neutral_col = base_col

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


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "features.yaml")
    out = panel.copy()

    for col in RAW_COLS:
        if col not in out.columns:
            logger.warning("Column '%s' not found in panel; skipping.", col)
            continue
        out = _engineer_one(out, col, cfg)

    return out


def main() -> None:
    in_path = project_root() / "data" / "processed" / "merged_panel.parquet"
    if not in_path.exists():
        logger.error("Missing input panel at %s. Run 07_merge_panel.py first.", in_path)
        return

    panel = pd.read_parquet(in_path)
    logger.info("Loaded connection panel: %d rows, %d unique CEOs",
                len(panel), panel["dirid_starting"].nunique())

    # Merge in richer point-in-time network metrics (degree/eigenvector
    # centrality, pagerank, structural holes, constraint, brokerage) so
    # CEO social capital isn't reduced to just connection count. Cumulative
    # (all-time) and active (currently-live) conventions are merged in as
    # parallel factor sets -- see NETWORK_METRIC_COLS/RAW_COLS above.
    cumulative_path = project_root() / NETWORK_METRICS_CUMULATIVE_PATH
    if cumulative_path.exists():
        cumulative = pd.read_parquet(cumulative_path).rename(columns={"node_id": "dirid_starting"})
        panel = panel.merge(cumulative, on=["dirid_starting", "year"], how="left")
        logger.info("Merged cumulative network metrics from %s: %d/%d rows matched",
                    cumulative_path,
                    panel["degree_centrality"].notna().sum() if "degree_centrality" in panel.columns else 0,
                    len(panel))
    else:
        logger.warning("Cumulative network metrics panel not found at %s; only n_unique_connected will be used. "
                        "Run 06_build_network_panel.py to compute richer social capital metrics.", cumulative_path)

    active_path = project_root() / NETWORK_METRICS_ACTIVE_PATH
    if active_path.exists():
        active = pd.read_parquet(active_path).rename(columns={"node_id": "dirid_starting"})
        active = active.rename(columns={c: f"{c}_active" for c in NETWORK_METRIC_COLS if c in active.columns})
        panel = panel.merge(active, on=["dirid_starting", "year"], how="left")
        logger.info("Merged active network metrics from %s: %d/%d rows matched",
                    active_path,
                    panel["degree_centrality_active"].notna().sum() if "degree_centrality_active" in panel.columns else 0,
                    len(panel))
    else:
        logger.warning("Active network metrics panel not found at %s; skipping active social-capital factors. "
                        "Run 06_build_network_panel.py to compute them.", active_path)

    # Merge in industry_id (first 2 digits of SIC code) for industry
    # neutralization, via permco from data/raw/backtest_returns.parquet.
    returns_path = project_root() / "data" / "raw" / "backtest_returns.parquet"
    if returns_path.exists() and "permco" in panel.columns:
        ind = pd.read_parquet(returns_path, columns=["permco", "siccd"]).dropna().drop_duplicates("permco")
        ind["industry_id"] = (pd.to_numeric(ind["siccd"], errors="coerce") // 100).astype("Int64").astype(str)
        panel = panel.merge(ind[["permco", "industry_id"]], on="permco", how="left")
        logger.info("Merged industry_id (SIC 2-digit) for %d/%d rows",
                    panel["industry_id"].notna().sum(), len(panel))
    else:
        logger.warning("Could not merge industry_id (missing %s or 'permco' column); "
                        "industry neutralization will be skipped.", returns_path)

    features = build_features(panel)

    out_path = project_root() / "data" / "processed" / "connection_features.parquet"
    features.to_parquet(out_path)
    logger.info("Saved engineered connection features to %s (%d rows)", out_path, len(features))


if __name__ == "__main__":
    main()
