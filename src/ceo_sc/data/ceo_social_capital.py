"""CEO social capital (BoardEx) data collection from WRDS.

BoardEx enforces a maximum date-range window per query (default 3600
days, configurable via ``configs/data.yaml:ceo_social_capital.max_query_window_days``).
This module chunks the requested date range accordingly and persists
each chunk to disk so partially-completed collections can be resumed.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ceo_sc.data.wrds_client import WRDSClient
from ceo_sc.utils.config import load_config, configs_dir, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _date_chunks(start: str, end: str, max_window_days: int) -> list[tuple[date, date]]:
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date() if end else date.today()
    chunks = []
    cursor = start_d
    step = timedelta(days=max_window_days)
    while cursor <= end_d:
        chunk_end = min(cursor + step - timedelta(days=1), end_d)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _load_ceo_director_ids(ceo_sample_path: str, id_column: str) -> list[str]:
    path = project_root() / ceo_sample_path
    if not path.exists():
        raise FileNotFoundError(
            f"CEO sample not found at {path}. Run ceo_sample.py "
            "(scripts/09_build_ceo_sample.py) first."
        )
    sample = pd.read_parquet(path)
    return (
        sample[id_column]
        .dropna()
        .astype("Int64")   # nullable int, avoids float64 upcast artifacts (e.g. "340769.0")
        .astype(str)
        .unique()
        .tolist()
    )


def _batched(seq: list[str], batch_size: int) -> list[list[str]]:
    return [seq[i:i + batch_size] for i in range(0, len(seq), batch_size)]


def _build_ceo_sc_query(cfg: dict, start: date, end: date,
                        ceo_director_ids: list[str]) -> tuple[str, dict]:
    """Build the SQL + params for a BoardEx role/employment query overlapping
    [start, end], restricted to the given CEO director IDs.

    BoardEx role tables store ``start_year_column``/``end_year_column`` as
    4-digit integer years (not dates), and ``end_year_column`` is typically
    NULL (or 'Curr') for still-active roles. Comparing against integer years
    (not date strings) and using an interval-overlap predicate correctly
    captures any role active at any point during the chunk window.
    """
    ccfg = cfg["ceo_social_capital"]
    sql = f"""
        SELECT
            CASE WHEN overlapyearstart ~ '^[0-9]+$'
                 THEN CAST(overlapyearstart AS INTEGER) ELSE NULL END AS start_year,
            CASE WHEN overlapyearend ~ '^[0-9]+$'
                 THEN CAST(overlapyearend AS INTEGER) ELSE NULL END AS end_year,
            associationtype,
            companyname AS company_connected,
            directorname AS dirName_connected,
            directorid AS dirID_connected,
            role,
            associatedrole,
            orgtype,
            roletitle,
            companyid AS companyid_connected,
            dirbrdname AS dirName_starting,
            dirbrdid AS dirID_starting
        FROM
            {ccfg['wrds_library']}.{ccfg['wrds_table']}
        WHERE
            CAST(dirbrdid AS TEXT) = ANY(%(ceo_director_ids)s)
            AND (
                overlapyearstart ~ '^[0-9]+$'
                AND CAST(overlapyearstart AS INTEGER) <= %(end_year)s
            )
            AND (
                overlapyearend IS NULL
                OR overlapyearend ILIKE 'curr'
                OR (
                    overlapyearend ~ '^[0-9]+$'
                    AND CAST(overlapyearend AS INTEGER) >= %(start_year)s
                )
            );
    """
    params = {
        "start_year": start.year,
        "end_year": end.year,
        "ceo_director_ids": ceo_director_ids,
    }
    return sql, params


def _stream_query_to_parquet(client: WRDSClient, sql: str, params: dict,
                             out_path: Path, chunksize: int) -> int:
    """Execute ``sql`` and stream results straight to a Parquet file in
    row batches of ``chunksize``, instead of buffering the full result set
    in memory (avoids OOM on large BoardEx result sets).

    Returns the total number of rows written. Writes an empty (0-row)
    Parquet file if the query returns no rows, so the chunk is still
    marked as "completed" for caching/resume purposes.
    """
    batches = client.query(sql, params=params, chunksize=chunksize)
    writer = None
    total_rows = 0
    try:
        for batch_df in batches:
            if batch_df.empty:
                continue
            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            total_rows += len(batch_df)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame().to_parquet(out_path, index=False)
    return total_rows


def collect_ceo_sc(force: bool = False) -> None:
    """Collect BoardEx network/relationship data in (date x CEO-id-batch)
    chunks and stream each straight to its own cached Parquet file.

    Chunking along both dimensions (date window and CEO-id batch) bounds
    the size of each individual query's result set; streaming with
    ``fetch_chunksize`` bounds the peak memory used while writing it to
    disk. This avoids the OOM errors that a single large in-memory
    DataFrame (one per date chunk, or worse, the full collection) can hit.

    Set ``force=True`` to re-download chunks that already exist on disk.
    Does not return a combined DataFrame -- read the per-chunk Parquet
    files in ``raw_output_dir`` directly (see 05_build_network_metrics.py)
    to avoid re-buffering the entire collection in memory.
    """
    cfg = load_config(configs_dir() / "data.yaml")
    ccfg = cfg["ceo_social_capital"]

    ceo_director_ids = _load_ceo_director_ids(ccfg["ceo_sample_path"], ccfg["ceo_sample_id_column"])
    logger.info("Restricting CEO_SC collection to %d CEOs from ceo_sample.parquet", len(ceo_director_ids))

    id_batches = _batched(ceo_director_ids, ccfg.get("ceo_id_batch_size", len(ceo_director_ids)))
    fetch_chunksize = ccfg.get("fetch_chunksize", 50_000)

    out_dir = project_root() / ccfg["raw_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    date_chunks = _date_chunks(ccfg["start_date"], ccfg["end_date"], ccfg["max_query_window_days"])
    logger.info("Collecting CEO_SC data in %d date chunk(s) x %d CEO-id batch(es)",
                len(date_chunks), len(id_batches))

    with WRDSClient() as client:
        for start, end in date_chunks:
            for batch_idx, id_batch in enumerate(id_batches):
                chunk_path = out_dir / f"boardex_{start}_{end}_batch{batch_idx}.parquet"
                if chunk_path.exists() and not force:
                    logger.info("Chunk %s..%s batch %d/%d already cached, skipping",
                                start, end, batch_idx + 1, len(id_batches))
                    continue
                logger.info("Fetching chunk %s..%s batch %d/%d (%d CEOs)",
                            start, end, batch_idx + 1, len(id_batches), len(id_batch))
                sql, params = _build_ceo_sc_query(cfg, start, end, id_batch)
                n_rows = _stream_query_to_parquet(client, sql, params, chunk_path, fetch_chunksize)
                logger.info("Saved %d rows to %s", n_rows, chunk_path)


if __name__ == "__main__":
    collect_ceo_sc()
