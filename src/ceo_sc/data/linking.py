"""Linking table construction: company IDs (gvkey/permno) <-> CEO/director IDs.

The exact BoardEx company-linking table must be confirmed against the
WRDS BoardEx schema browser; see ``configs/data.yaml:linking_table``.
"""

from __future__ import annotations

import pandas as pd

from ceo_sc.data.wrds_client import WRDSClient
from ceo_sc.utils.config import load_config, configs_dir
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_linking_table(client: WRDSClient, cfg: dict) -> pd.DataFrame:
    """Build a gvkey <-> BoardEx director_id linking table.

    Returns
    -------
    pd.DataFrame with columns: gvkey, director_id, role, start_date, end_date
    """
    lcfg = cfg["linking_table"]
    cols = ", ".join(lcfg["columns"])
    sql = f"""
        SELECT {cols}
        FROM {lcfg['wrds_library']}.{lcfg['wrds_table']}
        WHERE preferred = 1
    """
    logger.info("Fetching linking table from %s.%s...", lcfg["wrds_library"], lcfg["wrds_table"])
    return client.query(sql)


def collect_and_save(output_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "data.yaml")
    with WRDSClient() as client:
        df = build_linking_table(client, cfg)
    if output_path:
        df.to_parquet(output_path, index=False)
        logger.info("Saved linking table to %s (%d rows)", output_path, len(df))
    return df


if __name__ == "__main__":
    collect_and_save("data/raw/linking_table.parquet")
