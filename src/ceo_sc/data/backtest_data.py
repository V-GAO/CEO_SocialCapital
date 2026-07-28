"""Backtesting data collection (e.g. CRSP monthly returns) from WRDS."""

from __future__ import annotations

import pandas as pd

from ceo_sc.data.wrds_client import WRDSClient
from ceo_sc.utils.config import load_config, configs_dir
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

# The code was using inner join in SQL to match the two tables but it was slow
# So we are fetching the data separately and merging them locally

def get_backtest_returns(client: WRDSClient, cfg: dict) -> pd.DataFrame:
    bcfg = cfg["backtest_data"]
    icfg = cfg["industry_data"]

    # --- 1. Fetch monthly returns (large table, filtered by date only, no join). ---
    b_cols = ", ".join(bcfg["columns"])
    b_sql = f"""
        SELECT {b_cols}
        FROM {bcfg['wrds_library']}.{bcfg['wrds_table']}
        WHERE date >= %(start_date)s
        {"AND date <= %(end_date)s" if bcfg.get("end_date") else ""}
    """
    b_params = {"start_date": bcfg["start_date"]}
    if bcfg.get("end_date"):
        b_params["end_date"] = bcfg["end_date"]

    logger.info("Fetching backtest returns from %s.%s...", bcfg["wrds_library"], bcfg["wrds_table"])
    returns = client.query(b_sql, params=b_params)

    # --- 2. Fetch industry/name-history table (small, no join, fetched whole). ---
    i_cols = ", ".join(icfg["columns"])
    i_sql = f"SELECT {i_cols} FROM {icfg['wrds_library']}.{icfg['wrds_table']}"

    logger.info("Fetching industry data from %s.%s...", icfg["wrds_library"], icfg["wrds_table"])
    industry = client.query(i_sql)

    # --- 3. Merge locally on permco only. stocknames has multiple rows per
    # permco (one per historical name/exchange period), so we first collapse
    # it to a single row per permco (most recent classification) to avoid
    # duplicating return rows in the merge.
    logger.info("Merging %d return rows with %d industry rows on permco...",
                len(returns), len(industry))

    industry["namedt"] = pd.to_datetime(industry["namedt"])
    industry_dedup = (
        industry.sort_values("namedt")
        .groupby("permco", as_index=False)
        .last()
        .drop(columns=["namedt", "nameenddt"])
    )

    merged = returns.merge(industry_dedup, on="permco", how="left")

    logger.info("Merged result: %d rows", len(merged))
    return merged


def collect_and_save(output_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "data.yaml")
    with WRDSClient() as client:
        df = get_backtest_returns(client, cfg)
    if output_path:
        df.to_parquet(output_path, index=False)
        logger.info("Saved backtest returns to %s (%d rows)", output_path, len(df))
    return df


if __name__ == "__main__":
    collect_and_save("data/raw/backtest_returns.parquet")
