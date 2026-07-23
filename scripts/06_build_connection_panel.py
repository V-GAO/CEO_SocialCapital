"""Entry point: build a yearly panel of unique connections per CEO.

For each ``dirid_starting`` (CEO), counts the number of unique
``dirid_connected`` individuals in their network for each year from
2000 to 2026. Uses the "cumulative" convention: a connection enters
the CEO's network in the year its relationship started and persists
through all subsequent years (no forward-looking bias).

Uses Polars for efficient lazy processing of the large raw BoardEx
parquet files -- reads only 3 columns, deduplicates to unique pairs,
then explodes into yearly rows in a single pass.

Output: data/processed/connection_count_panel.parquet
    Columns: dirid_starting (str), year (int), n_unique_connected (int)
"""

import polars as pl

from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config(configs_dir() / "data.yaml")
    ccfg = cfg["ceo_social_capital"]
    ncfg = cfg["network_panel"]

    raw_dir = project_root() / ccfg["raw_output_dir"]
    raw_files = sorted(raw_dir.glob("*.parquet"))
    if not raw_files:
        logger.error("No CEO_SC raw files found in %s. Run 02_collect_ceo_sc.py first.", raw_dir)
        return

    year_start = ncfg["year_start"]
    year_end = ncfg["year_end"]

    logger.info("Building connection panel from %d raw files, years %d-%d",
                len(raw_files), year_start, year_end)

    # Single lazy pass: read only the 3 columns we need, cast IDs from
    # float64 -> Int64 -> str (fixes float artifacts like "123.0"),
    # deduplicate to unique (source, target) pairs keeping the earliest
    # start_year as the year the connection first enters the network.
    lazy_frames = []
    for f in raw_files:
        lf = (
            pl.scan_parquet(f)
            .select(
                pl.col("dirid_starting").cast(pl.Int64, strict=False).cast(pl.Utf8),
                pl.col("dirid_connected").cast(pl.Int64, strict=False).cast(pl.Utf8),
                pl.col("start_year").cast(pl.Int32, strict=False),
            )
            .drop_nulls(["dirid_starting", "dirid_connected", "start_year"])
        )
        lazy_frames.append(lf)

    df = (
        pl.concat(lazy_frames)
        .unique(subset=["dirid_starting", "dirid_connected", "start_year"])
        .sort(["dirid_starting", "dirid_connected", "start_year"])
        .group_by(["dirid_starting", "dirid_connected"])
        .agg(pl.col("start_year").min().alias("first_year"))
        .filter(pl.col("first_year").is_not_null())
    ).collect()

    logger.info("Deduplicated to %d unique (CEO, connection) pairs", len(df))

    # Instead of exploding each pair into yearly rows (which would
    # materialize ~600M rows and OOM), compute new connections per
    # (CEO, first_year), then cumulative sum across years. This gives
    # the same result -- n_unique_connected per CEO per year under the
    # cumulative convention -- with only ~30k CEOs x 27 years of output.
    new_per_year = (
        df.group_by(["dirid_starting", "first_year"])
        .agg(pl.col("dirid_connected").n_unique().alias("new_connections"))
        .sort(["dirid_starting", "first_year"])
    )

    # Build a complete (CEO x year) grid and left-join the new-connections,
    # filling missing years with 0, then cumulative sum per CEO.
    all_years = pl.DataFrame({"year": list(range(year_start, year_end + 1))})
    all_ceos = df.select("dirid_starting").unique()
    grid = all_ceos.join(all_years, how="cross")

    panel = (
        grid.join(new_per_year, left_on=["dirid_starting", "year"],
                  right_on=["dirid_starting", "first_year"], how="left")
        .with_columns(pl.col("new_connections").fill_null(0))
        .sort(["dirid_starting", "year"])
        .with_columns(
            pl.col("new_connections")
            .cum_sum()
            .over("dirid_starting")
            .alias("n_unique_connected")
        )
        .select(["dirid_starting", "year", "n_unique_connected"])
    )

    out_path = project_root() / "data" / "processed" / "connection_count_panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)
    logger.info("Saved connection panel to %s (%d rows, %d unique CEOs)",
                out_path, len(panel), panel["dirid_starting"].n_unique())


if __name__ == "__main__":
    main()
