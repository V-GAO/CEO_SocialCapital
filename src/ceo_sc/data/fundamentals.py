"""Company fundamental data collection (S&P 1500 universe) from WRDS."""

from __future__ import annotations

import pandas as pd

from ceo_sc.data.wrds_client import WRDSClient
from ceo_sc.utils.config import load_config, configs_dir
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_sp1500_universe(client: WRDSClient, cfg: dict) -> pd.DataFrame:
    """Return the historical S&P 1500 constituent list (gvkey, from, thru)."""
    uni_cfg = cfg["universe"]
    sql = f"""
        SELECT gvkey, tic, fromdate AS from_date, companyname AS company_name
        FROM {uni_cfg['index_library']}.{uni_cfg['index_table']}
        WHERE indexid = %(index_id)s
    """
    logger.info("Fetching S&P 1500 constituent history...")
    return client.query(sql, params={"index_id": uni_cfg["index_id"]})


def get_fundamentals(client: WRDSClient, cfg: dict) -> pd.DataFrame:
    """Fetch company fundamentals (Compustat funda) restricted to the S&P 1500 universe."""
    fcfg = cfg["fundamentals"]
    cols = ", ".join(fcfg["columns"])
    sql = f"""
        SELECT {cols}
        FROM {fcfg['wrds_library']}.{fcfg['wrds_table']}
        WHERE datadate >= %(start_date)s
        {"AND datadate <= %(end_date)s" if fcfg.get("end_date") else ""}
        AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
    """
    params = {"start_date": fcfg["start_date"]}
    if fcfg.get("end_date"):
        params["end_date"] = fcfg["end_date"]

    logger.info("Fetching fundamentals from %s.%s...", fcfg["wrds_library"], fcfg["wrds_table"])
    fundamentals = client.query(sql, params=params)

    universe = get_sp1500_universe(client, cfg)
    fundamentals = fundamentals.merge(universe[["gvkey"]].drop_duplicates(), on="gvkey", how="inner")
    return fundamentals


def collect_and_save(output_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "data.yaml")
    with WRDSClient() as client:
        df = get_fundamentals(client, cfg)
    if output_path:
        df.to_parquet(output_path, index=False)
        logger.info("Saved fundamentals to %s (%d rows)", output_path, len(df))
    return df


if __name__ == "__main__":
    collect_and_save("data/raw/fundamentals.parquet")
