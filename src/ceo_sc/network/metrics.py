"""Social network metrics for CEO social capital construction.

Computes, per node (director/CEO), on a NetworkX graph built from
BoardEx co-employment/board-membership relationships:

- Degree Centrality
- Eigenvector Centrality
- Betweenness Centrality
- Closeness Centrality
- PageRank
- Structural Holes (Burt's effective size)
- Network Constraint (Burt's constraint)
- Brokerage Measures (proxy: inverse of constraint, i.e. brokerage
  opportunity from occupying non-redundant structural positions)

All metrics are computed per connected component to avoid undefined
values (e.g. closeness centrality) on disconnected graphs.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import networkx as nx
import pandas as pd

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_graph(edges: pd.DataFrame, source: str = "source", target: str = "target",
                 weight: str | None = None) -> nx.Graph:
    """Build an undirected graph from an edge list DataFrame."""
    G = nx.Graph()
    if weight and weight in edges.columns:
        G.add_weighted_edges_from(edges[[source, target, weight]].itertuples(index=False, name=None))
    else:
        G.add_edges_from(edges[[source, target]].itertuples(index=False, name=None))
    return G


def build_graph_from_parquet_files(files: list[Path], source: str = "source",
                                    target: str = "target",
                                    weight: str | None = None) -> nx.Graph:
    """Build an undirected graph from a list of edge-list Parquet files,
    without ever materializing the full combined edge list in memory.

    Only ``source``/``target``(/``weight``) columns are read from each file,
    and each file's edges are added to the graph incrementally -- this
    keeps peak memory bounded by the largest single file rather than the
    sum of all files.
    """
    columns = [source, target] + ([weight] if weight else [])
    G = nx.Graph()
    for f in files:
        df = pd.read_parquet(f, columns=columns).dropna(subset=[source, target])
        df = df.drop_duplicates()
        if weight and weight in df.columns:
            G.add_weighted_edges_from(df[[source, target, weight]].itertuples(index=False, name=None))
        else:
            G.add_edges_from(df[[source, target]].itertuples(index=False, name=None))
        logger.info("Added edges from %s (%d unique rows)", f.name, len(df))
    return G


def compute_metrics_hybrid(edges: pd.DataFrame, source: str = "source",
                            target: str = "target") -> pd.DataFrame:
    """Compute network metrics entirely with ``igraph`` (C backend).

    Burt's constraint/effective size are local (ego-network) measures, so
    they can be computed in O(sum of degree^2) via ``igraph``'s native
    ``constraint()`` and local clustering coefficient -- unlike
    ``networkx.effective_size()``, which materializes a dense N x N
    matrix internally and OOMs on graphs with hundreds of thousands of
    nodes, even though the underlying graph is sparse.

    Betweenness and closeness centrality are intentionally omitted: both
    require all-pairs shortest paths (O(V*(V+E))) even with igraph's C
    implementation, which is infeasible at this graph's scale (~400k
    nodes / ~3.4M edges) repeated across many yearly snapshots.

    Returns a DataFrame indexed by node id (without ``betweenness_centrality``
    or ``closeness_centrality``, unlike :func:`compute_network_metrics`).
    """
    edges = edges[[source, target]].dropna().drop_duplicates()
    if edges.empty:
        empty = pd.DataFrame(columns=[
            "degree_centrality", "eigenvector_centrality",
            "pagerank", "structural_holes_effective_size",
            "network_constraint", "brokerage_measure",
        ])
        empty.index.name = "node_id"
        return empty

    nodes = pd.unique(edges[[source, target]].to_numpy().ravel())
    node_index = {node: i for i, node in enumerate(nodes)}
    igraph_edges = [
        (node_index[s], node_index[t])
        for s, t in edges.itertuples(index=False, name=None)
    ]

    g = ig.Graph(n=len(nodes), edges=igraph_edges, directed=False)
    g.simplify()

    logger.info("Computing igraph metrics for graph with %d nodes, %d edges", g.vcount(), g.ecount())

    degree = g.degree()
    n = g.vcount()
    degree_centrality = [d / (n - 1) if n > 1 else 0.0 for d in degree]
    pagerank = g.pagerank()
    try:
        eigenvector = g.eigenvector_centrality()
    except ig.InternalError:
        logger.warning("igraph eigenvector centrality failed to converge; falling back to zeros")
        eigenvector = [0.0] * n

    # Burt's constraint: native igraph implementation, local to each node's
    # ego-network (no dense N x N matrix).
    constraint = g.constraint()

    # Burt's effective size, binary/unweighted case: effective_size(i) =
    # degree(i) - clustering_coefficient(i) * (degree(i) - 1). Derived from
    # Burt's z_i (ties among i's neighbors) via the local clustering
    # coefficient identity z_i = C(i) * n_i * (n_i - 1) / 2. ``mode="zero"``
    # returns 0 (rather than nan) for degree < 2 nodes, which is the
    # correct effective size for a node with no redundancy possible.
    clustering = g.transitivity_local_undirected(mode="zero")
    effective_size = [d - c * (d - 1) for d, c in zip(degree, clustering)]

    brokerage = [
        (1.0 / c) if c not in (None, 0) else float("nan")
        for c in constraint
    ]

    df = pd.DataFrame({
        "degree_centrality": degree_centrality,
        "eigenvector_centrality": eigenvector,
        "pagerank": pagerank,
        "structural_holes_effective_size": effective_size,
        "network_constraint": constraint,
        "brokerage_measure": brokerage,
    }, index=nodes)
    df.index.name = "node_id"
    return df


def compute_network_metrics(G: nx.Graph, weight: str | None = "weight") -> pd.DataFrame:
    """Compute all network metrics for every node in ``G``.

    Returns a DataFrame indexed by node id with one column per metric.
    """
    logger.info("Computing network metrics for graph with %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight=weight)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, weight=weight)
    constraint = nx.constraint(G)
    effective_size = nx.effective_size(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, weight=weight, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality failed to converge; falling back to numpy solver")
        eigenvector = nx.eigenvector_centrality_numpy(G, weight=weight)

    brokerage = {
        node: (1.0 / constraint[node]) if constraint.get(node) not in (None, 0) else float("nan")
        for node in G.nodes()
    }

    df = pd.DataFrame({
        "degree_centrality": degree,
        "eigenvector_centrality": eigenvector,
        "betweenness_centrality": betweenness,
        "closeness_centrality": closeness,
        "pagerank": pagerank,
        "structural_holes_effective_size": effective_size,
        "network_constraint": constraint,
        "brokerage_measure": brokerage,
    })
    df.index.name = "node_id"
    return df
