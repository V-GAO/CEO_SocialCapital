"""Point-in-time ("as-of") edge list construction for CEO social capital.

BoardEx relationship rows carry a ``start_year``/``end_year`` for each
connection. To avoid look-ahead bias when using network metrics as
features in a trading strategy, any snapshot "as of year Y" must only
include edges with ``start_year <= Y`` -- a connection that begins in the
future cannot influence a CEO's network today.

This module:

1. Reduces the raw, wide BoardEx chunk files (many columns, produced by
   ``ceo_social_capital.collect_ceo_sc``) down to a single narrow "edges
   core" Parquet file containing only the columns needed for point-in-time
   graph construction (``source``, ``target``, ``start_year``, ``end_year``).
   This is streamed batch-by-batch so peak memory is bounded by one batch,
   not the full raw dataset.
2. Given that edges-core file, extracts the edge list valid "as of" a
   given year, under one of two conventions:
   - ``cumulative``: any edge with ``start_year <= as_of_year`` (a
     connection persists as a social-capital resource even after the
     underlying role ends).
   - ``active``: only edges "live" at ``as_of_year``, i.e.
     ``start_year <= as_of_year <= end_year`` (or ``end_year`` unknown/
     still-ongoing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_edges_core(
    raw_files: list[Path],
    out_path: Path,
    source_col: str,
    target_col: str,
    start_col: str,
    end_col: str,
    batch_size: int = 500_000,
) -> int:
    """Stream ``raw_files`` (wide BoardEx chunk Parquet files) down to a
    single narrow "edges core" Parquet file with just the columns needed
    for point-in-time graph construction.

    Rows missing ``source_col``, ``target_col``, or ``start_col`` are
    dropped (an edge with no known start year can never be safely
    included in a point-in-time snapshot). Returns the number of rows
    written.
    """
    columns = [source_col, target_col, start_col, end_col]
    writer = None
    total_rows = 0
    try:
        for f in raw_files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                df = batch.to_pandas()
                df = df.dropna(subset=[source_col, target_col, start_col])
                if df.empty:
                    continue
                # Normalize IDs to plain integer strings -- BoardEx IDs can
                # round-trip through Postgres/pandas as float64 (e.g. NaNs
                # elsewhere in the column upcast the whole column), which
                # would otherwise produce mismatched "123.0" vs "123" ids.
                df[source_col] = pd.to_numeric(df[source_col], errors="coerce").astype("Int64").astype(str)
                df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64").astype(str)
                df[start_col] = pd.to_numeric(df[start_col], errors="coerce").astype("Int64")
                df[end_col] = pd.to_numeric(df[end_col], errors="coerce").astype("Int64")
                df = df.dropna(subset=[source_col, target_col, start_col])

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                total_rows += len(df)
            logger.info("Reduced %s into edges core (%d rows so far)", f.name, total_rows)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame(columns=columns).to_parquet(out_path, index=False)
    return total_rows


def load_asof_edges(
    edges_core_path: Path,
    as_of_year: int,
    mode: str,
    source_col: str,
    target_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """Load the unique edge list valid "as of" ``as_of_year`` from the
    edges-core Parquet file, using either the ``cumulative`` or ``active``
    convention (see module docstring). Only ``start_col <= as_of_year`` is
    ever used as a predicate, so a future connection can never leak into
    an earlier snapshot.

    Reads and decompresses the edges-core file from disk on every call --
    fine for a single as-of snapshot, but when building a full multi-year
    panel (as in 06_build_network_panel.py), prefer loading the file once
    with :func:`load_edges_core_frame` and slicing per year with
    :func:`asof_edges_from_frame` instead, to avoid re-reading/decompressing
    the same file from disk once per (year, mode) pair.
    """
    if mode not in ("cumulative", "active"):
        raise ValueError(f"mode must be 'cumulative' or 'active', got {mode!r}")

    filters = [(start_col, "<=", as_of_year)]
    df = pd.read_parquet(edges_core_path, filters=filters)

    if mode == "active":
        is_ongoing = df[end_col].isna()
        is_still_active = df[end_col] >= as_of_year
        df = df[is_ongoing | is_still_active]

    return df[[source_col, target_col]].drop_duplicates()


def load_edges_core_frame(
    edges_core_path: Path,
    source_col: str,
    target_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """Load the edges-core Parquet file into memory once, sorted by
    ``start_col``.

    Intended for building a full multi-year as-of panel: reading and
    decompressing a large Parquet file from disk is the dominant cost of
    ``load_asof_edges``, and doing it once per (year, mode) pair is
    wasteful when the same underlying rows are re-read (and increasingly
    so, since the "cumulative" filter only grows with year) on every
    iteration. Sorting by ``start_col`` here lets :func:`asof_edges_from_frame`
    slice the "as of" rows with a fast in-memory binary search instead of
    re-scanning/re-filtering the whole frame from scratch each time.
    """
    df = pd.read_parquet(edges_core_path, columns=[source_col, target_col, start_col, end_col])
    return df.sort_values(start_col, kind="mergesort", ignore_index=True)


def asof_edges_from_frame(
    edges_core_df: pd.DataFrame,
    as_of_year: int,
    mode: str,
    source_col: str,
    target_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """In-memory equivalent of :func:`load_asof_edges`, operating on a frame
    already loaded via :func:`load_edges_core_frame` (sorted by ``start_col``).

    Uses a binary search on the pre-sorted ``start_col`` to slice the
    "as of" rows in O(log n), rather than re-scanning the whole frame.
    """
    if mode not in ("cumulative", "active"):
        raise ValueError(f"mode must be 'cumulative' or 'active', got {mode!r}")

    end_idx = int(np.searchsorted(edges_core_df[start_col].to_numpy(), as_of_year, side="right"))
    df = edges_core_df.iloc[:end_idx]

    if mode == "active":
        is_ongoing = df[end_col].isna()
        is_still_active = df[end_col] >= as_of_year
        df = df[is_ongoing | is_still_active]

    return df[[source_col, target_col]].drop_duplicates()
