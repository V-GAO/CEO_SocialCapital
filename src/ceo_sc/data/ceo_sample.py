"""CEO sample construction via BoardEx Organization Composition.

Filters BoardEx's "Organization - Composition of Officers, Directors and
Senior Managers" table to CEO-titled roles, restricted to the set of
BoardEx ``companyid`` values already resolved in the linking table
(``data/raw/linking_table.parquet``, produced by ``linking.py``).
"""

from __future__ import annotations

import pandas as pd

from ceo_sc.data.wrds_client import WRDSClient
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_company_ids(linking_table_path: str) -> list[str]:
    path = project_root() / linking_table_path
    if not path.exists():
        raise FileNotFoundError(
            f"Linking table not found at {path}. Run linking.py "
            "(scripts/03_build_linking_table.py) first."
        )
    linking = pd.read_parquet(path)
    return (
        linking["companyid"]
        .dropna()
        .astype("Int64")   # nullable int, avoids float64 upcast artifacts (e.g. "133.0")
        .astype(str)
        .unique()
        .tolist()
    )


def get_ceo_sample(client: WRDSClient, cfg: dict, company_ids: list[str]) -> pd.DataFrame:
    """Query BoardEx Organization Composition for CEO roles at the given company IDs.

    ``company_id_column`` is cast to TEXT on both sides of the comparison
    since BoardEx numeric-looking identifiers are frequently stored as
    varchar (see the same issue encountered with year columns).
    """
    scfg = cfg["ceo_sample"]
    role_conditions = " OR ".join(
        f"{scfg['role_column']} ILIKE %(kw{i})s" for i in range(len(scfg["ceo_role_keywords"]))
    )
    exclude_keywords = scfg.get("exclude_role_keywords", [])
    exclude_clause = ""
    if exclude_keywords:
        exclude_conditions = " AND ".join(
            f"{scfg['role_column']} NOT ILIKE %(ex{i})s" for i in range(len(exclude_keywords))
        )
        exclude_clause = f"AND ({exclude_conditions})"

    sql = f"""
        SELECT companyid, directorid AS directorID_starting, directorname AS dirName_starting,
               rolename, datestartrole, dateendrole, seniority
        FROM {scfg['wrds_library']}.{scfg['wrds_table']}
        WHERE CAST({scfg['company_id_column']} AS TEXT) = ANY(%(company_ids)s)
          AND ({role_conditions})
          {exclude_clause}
    """
    params = {"company_ids": company_ids}
    params.update({f"kw{i}": kw for i, kw in enumerate(scfg["ceo_role_keywords"])})
    params.update({f"ex{i}": kw for i, kw in enumerate(exclude_keywords)})

    logger.info("Fetching CEO sample from %s.%s for %d companies...",
                scfg["wrds_library"], scfg["wrds_table"], len(company_ids))
    return client.query(sql, params=params)


def collect_and_save(output_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(configs_dir() / "data.yaml")
    scfg = cfg["ceo_sample"]

    company_ids = _load_company_ids(scfg["linking_table_path"])
    with WRDSClient() as client:
        df = get_ceo_sample(client, cfg, company_ids)

    out_path = output_path or scfg["output_path"]
    if out_path:
        df.to_parquet(out_path, index=False)
        logger.info("Saved CEO sample to %s (%d rows)", out_path, len(df))
    return df


if __name__ == "__main__":
    collect_and_save()
