"""Entry point: build the CEO social network and compute network metrics.

Expects a BoardEx relationship edge list at ``data/raw/ceo_sc`` (produced by
``02_collect_ceo_sc.py``) with at least two director-id columns to form edges.
Adjust `source`/`target` column names below to match the actual BoardEx
relationship table schema once confirmed.
"""

from pathlib import Path

#from ceo_sc.network.metrics import build_graph_from_parquet_files, compute_network_metrics
from ceo_sc.network.metrics import build_graph_from_parquet_files, compute_metrics_hybrid
from ceo_sc.utils.config import project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    raw_dir = project_root() / "data" / "raw" / "ceo_sc"
    edge_files = list(raw_dir.glob("*.parquet"))
    if not edge_files:
        logger.error("No CEO_SC raw files found in %s. Run 02_collect_ceo_sc.py first.", raw_dir)
        return

    # dirid_starting is the ceo of interest and dirid_connected is the ID of connected individual.
    G = build_graph_from_parquet_files(edge_files, source="dirid_starting", target="dirid_connected")
    metrics = compute_network_metrics(G)

    out_path = project_root() / "data" / "processed" / "network_metrics.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_path)
    logger.info("Saved network metrics to %s (%d nodes)", out_path, len(metrics))


if __name__ == "__main__":
    main()
