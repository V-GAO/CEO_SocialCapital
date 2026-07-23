"""Entry point: build point-in-time (as-of) CEO social network metric panels.

Produces two (node_id x year) panels -- one under the "cumulative" edge
convention, one under "active" -- to avoid look-ahead bias when these
metrics are later used as trading-strategy features (see
configs/features.yaml:lagging for the additional reporting-lag applied
downstream). See src/ceo_sc/network/edges.py for the point-in-time
construction logic.

Usage: python scripts/06_build_network_panel.py [--force-edges-core]
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from ceo_sc.network.edges import build_edges_core, load_asof_edges
from ceo_sc.network.metrics import compute_metrics_hybrid
from ceo_sc.utils.config import configs_dir, load_config, project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main(force_edges_core: bool = False) -> None:
    cfg = load_config(configs_dir() / "data.yaml")
    ncfg = cfg["network_panel"]
    ccfg = cfg["ceo_social_capital"]

    source_col = ncfg["source_column"]
    target_col = ncfg["target_column"]
    start_col = ncfg["start_year_column"]
    end_col = ncfg["end_year_column"]

    edges_core_path = project_root() / ncfg["edges_core_path"]
    edges_core_path.parent.mkdir(parents=True, exist_ok=True)

    if edges_core_path.exists() and not force_edges_core:
        logger.info("Edges core already exists at %s, skipping rebuild", edges_core_path)
    else:
        raw_dir = project_root() / ccfg["raw_output_dir"]
        raw_files = list(raw_dir.glob("*.parquet"))
        if not raw_files:
            logger.error("No CEO_SC raw files found in %s. Run 02_collect_ceo_sc.py first.", raw_dir)
            return
        n_rows = build_edges_core(raw_files, edges_core_path, source_col, target_col, start_col, end_col)
        logger.info("Built edges core with %d rows at %s", n_rows, edges_core_path)

    years = range(ncfg["year_start"], ncfg["year_end"] + 1)

    for mode, out_key in (("cumulative", "cumulative_output_path"), ("active", "active_output_path")):
        out_path = project_root() / ncfg[out_key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for year in years:
                edges = load_asof_edges(edges_core_path, year, mode, source_col, target_col, start_col, end_col)
                metrics = compute_metrics_hybrid(edges, source=source_col, target=target_col)
                metrics = metrics.reset_index()  # index.name == "node_id"
                metrics["year"] = year
                logger.info("[%s] year=%d: %d nodes", mode, year, len(metrics))
                if metrics.empty:
                    continue

                table = pa.Table.from_pandas(metrics, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        logger.info("Saved %s network metrics panel to %s", mode, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-edges-core", action="store_true",
                        help="Rebuild data/processed/ceo_sc_edges_core.parquet even if it already exists.")
    args = parser.parse_args()
    main(force_edges_core=args.force_edges_core)
