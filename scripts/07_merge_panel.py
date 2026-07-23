"""Entry point: merge connection panel with fundamentals via linking table.

Join path:
    connection_count_panel  (dirid_starting, year, n_unique_connected)
        -- dirid_starting matches ceo_sample.directorid_starting -->
    ceo_sample              (directorid_starting, companyid, role dates)
        -- companyid matches linking_table.companyid -->
    linking_table           (companyid, gvkey, permco)
        -- gvkey + fyear matches fundamentals.gvkey + fyear -->
    fundamentals            (gvkey, fyear, at, sale, ni, ceq, mkvalt)

A CEO may hold multiple roles over time; we explode each role into
yearly rows (start_year..end_year) so that each (CEO, year) maps to
the correct company and its fundamentals.

Output: data/processed/merged_panel.parquet
"""

import polars as pl

from ceo_sc.utils.config import project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    raw = project_root() / "data" / "raw"
    proc = project_root() / "data" / "processed"

    # --- 1. Load connection panel ---
    conn = pl.read_parquet(proc / "connection_count_panel.parquet")
    logger.info("Connection panel: %d rows, %d CEOs",
                conn.height, conn["dirid_starting"].n_unique())

    # --- 2. Load CEO sample, explode roles into yearly rows ---
    ceo = pl.read_parquet(raw / "ceo_sample.parquet")

    # Parse YYYY-MM-DD dates -> year integers. Treat null/empty end date
    # as 2026 (still active).
    ceo = ceo.with_columns([
        pl.col("datestartrole").str.slice(0, 4).cast(pl.Int32).alias("start_year"),
        pl.when(pl.col("dateendrole").is_null() | (pl.col("dateendrole") == ""))
        .then(2026)
        .otherwise(pl.col("dateendrole").str.slice(0, 4).cast(pl.Int32))
        .alias("end_year"),
        pl.col("directorid_starting").cast(pl.Int64, strict=False).cast(pl.Utf8)
        .alias("dirid_starting"),
    ]).drop_nulls(["dirid_starting", "start_year", "end_year"])

    # Explode each role into yearly rows
    ceo_yearly = (
        ceo.with_columns(
            pl.int_ranges(pl.col("start_year"), pl.col("end_year") + 1).alias("year")
        )
        .explode("year")
        .select([
            "dirid_starting", "companyid", "year",
            "dirname_starting", "rolename", "seniority",
        ])
    )
    logger.info("CEO sample exploded to yearly: %d rows", ceo_yearly.height)

    # --- 3. Join connection panel with CEO roles ---
    panel = conn.join(ceo_yearly, on=["dirid_starting", "year"], how="left")
    logger.info("After joining CEO roles: %d rows (left join, some CEOs may not be in ceo_sample)",
                panel.height)

    # --- 4. Join with linking table on companyid ---
    linking = pl.read_parquet(raw / "linking_table.parquet")
    # All rows have preferred=1, but filter just in case future data differs
    linking = linking.filter(pl.col("preferred") == 1.0)

    panel = panel.join(
        linking.select(["companyid", "gvkey", "permco"]),
        on="companyid",
        how="left",
    )
    logger.info("After joining linking table: %d rows with gvkey, %d without",
                panel.filter(pl.col("gvkey").is_not_null()).height,
                panel.filter(pl.col("gvkey").is_null()).height)

    # --- 5. Join with fundamentals on (gvkey, fyear=year) ---
    fund = pl.read_parquet(raw / "fundamentals.parquet")
    fund = fund.rename({"fyear": "year"}).with_columns(pl.col("year").cast(pl.Int32))

    panel = panel.join(
        fund.select(["gvkey", "year", "at", "sale", "ni", "ceq", "mkvalt"]),
        on=["gvkey", "year"],
        how="left",
    )
    logger.info("After joining fundamentals: %d rows with mkvalt, %d without",
                panel.filter(pl.col("mkvalt").is_not_null()).height,
                panel.filter(pl.col("mkvalt").is_null()).height)

    # --- 6. Save ---
    panel = panel.sort(["dirid_starting", "year"])
    out_path = proc / "merged_panel.parquet"
    panel.write_parquet(out_path)
    logger.info("Saved merged panel to %s (%d rows, %d columns)",
                out_path, panel.height, panel.width)


if __name__ == "__main__":
    main()
