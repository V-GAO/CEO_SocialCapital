"""Entry point: merge stock returns into the connection panel and compute forward_return.

Reads ``data/raw/backtest_returns.parquet`` and ``data/processed/connection_features.parquet``,
links them via the linking table, aggregates monthly returns to annual, and computes
forward_return (next year's annual return) for Fama-MacBeth regressions.

Link path:
    backtest_returns (permno/permco/gvkey, date, ret)
        --> linking_table (permco <-> gvkey <-> companyid)
        --> connection_features (dirid_starting, year, ...)

Output: data/processed/panel_with_returns.parquet
"""

import polars as pl

from ceo_sc.utils.config import project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    raw = project_root() / "data" / "raw"
    proc = project_root() / "data" / "processed"

    # --- 1. Load returns ---
    ret = pl.read_parquet(raw / "backtest_returns.parquet")
    ret_cols = ret.columns
    logger.info("Returns data: %d rows, columns: %s", ret.height, ret_cols)

    # Determine which ID column links to the linking table
    linking = pl.read_parquet(raw / "linking_table.parquet").filter(pl.col("preferred") == 1.0)

    if "gvkey" in ret_cols:
        id_col = "gvkey"
        logger.info("Using gvkey to link returns to linking table")
    elif "permco" in ret_cols:
        id_col = "permco"
        logger.info("Using permco to link returns to linking table")
    else:
        logger.error("Returns data has neither 'gvkey' nor 'permco'. "
                     "Re-collect with permco in configs/data.yaml backtest_data.columns.")
        return

    # --- 2. Parse date -> year, aggregate monthly returns to annual ---
    ret = (
        ret
        .with_columns(pl.col("date").str.slice(0, 4).cast(pl.Int32).alias("year"))
        .filter(pl.col("ret").is_not_null())
    )

    # Annual return = product of (1 + monthly returns) - 1, compounded within year
    has_siccd = "siccd" in ret_cols
    annual_ret = (
        ret
        .group_by([id_col, "year"])
        .agg(
            (pl.col("ret").add(1).product() - 1).alias("annual_return"),
            pl.col("ret").count().alias("n_months"),
            *([pl.col("siccd").first().alias("siccd")] if has_siccd else []),
        )
        .filter(pl.col("n_months") >= 6)  # require at least 6 months of data
        .sort([id_col, "year"])
    )
    logger.info("Annual returns: %d rows (%d entities)",
                annual_ret.height, annual_ret[id_col].n_unique())

    # --- 3. Compute forward_return (shift annual_return by -1 within entity) ---
    annual_ret = annual_ret.with_columns(
        pl.col("annual_return").shift(-1).over(id_col).alias("forward_return")
    )

    # --- 4. Link returns to gvkey (if using permco) ---
    if id_col == "permco":
        annual_ret = annual_ret.join(
            linking.select(["permco", "gvkey"]).unique(),
            on="permco",
            how="inner",
        )
        id_col = "gvkey"

    # --- 5. Load connection features and merge ---
    features = pl.read_parquet(proc / "connection_features.parquet")
    logger.info("Connection features: %d rows, %d CEOs",
                features.height, features["dirid_starting"].n_unique())

    # Merge via gvkey + year (features panel has gvkey from merged_panel)
    if "gvkey" not in features.columns:
        logger.error("connection_features.parquet has no 'gvkey' column. "
                     "Run 07_merge_panel.py then 08_feature_engineer_connections.py first.")
        return

    merge_cols = [id_col, "year", "annual_return", "forward_return"]
    if has_siccd:
        merge_cols.append("siccd")

    panel = features.join(
        annual_ret.select(merge_cols),
        on=["gvkey", "year"],
        how="left",
    )

    n_with_ret = panel.filter(pl.col("forward_return").is_not_null()).height
    n_total = panel.height
    logger.info("Merged panel: %d rows, %d with forward_return (%.1f%%)",
                n_total, n_with_ret, 100 * n_with_ret / n_total if n_total else 0)

    # --- 6. Save ---
    out_path = proc / "panel_with_returns.parquet"
    panel.write_parquet(out_path)
    logger.info("Saved panel with returns to %s", out_path)


if __name__ == "__main__":
    main()
