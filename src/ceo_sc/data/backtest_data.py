"""Backtesting data collection (e.g. CRSP monthly returns) from WRDS."""

from __future__ import annotations

import pandas as pd

from ceo_sc.data.wrds_client import WRDSClient
from ceo_sc.utils.config import load_config, configs_dir
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_backtest_returns(client: WRDSClient, cfg: dict) -> pd.DataFrame:
    bcfg = cfg["backtest_data"]
    cols = ", ".join(bcfg["columns"])
    sql = f"""
        SELECT {cols}
        FROM {bcfg['wrds_library']}.{bcfg['wrds_table']}
        WHERE date >= %(start_date)s
        {"AND date <= %(end_date)s" if bcfg.get("end_date") else ""}
    """
    params = {"start_date": bcfg["start_date"]}
    if bcfg.get("end_date"):
        params["end_date"] = bcfg["end_date"]

    logger.info("Fetching backtest returns from %s.%s...", bcfg["wrds_library"], bcfg["wrds_table"])
    return client.query(sql, params=params)


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
