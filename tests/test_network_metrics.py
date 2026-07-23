from ceo_sc.network.metrics import build_graph, compute_network_metrics


def test_build_graph_has_expected_nodes_and_edges(synthetic_edges):
    G = build_graph(synthetic_edges)
    assert G.number_of_edges() == len(synthetic_edges)
    expected_nodes = set(synthetic_edges["source"]) | set(synthetic_edges["target"])
    assert set(G.nodes()) == expected_nodes


def test_compute_network_metrics_returns_all_columns(synthetic_edges):
    G = build_graph(synthetic_edges)
    metrics = compute_network_metrics(G, weight=None)
    expected_cols = {
        "degree_centrality",
        "eigenvector_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
        "structural_holes_effective_size",
        "network_constraint",
        "brokerage_measure",
    }
    assert expected_cols.issubset(metrics.columns)
    assert len(metrics) == G.number_of_nodes()


def test_degree_centrality_within_unit_interval(synthetic_edges):
    G = build_graph(synthetic_edges)
    metrics = compute_network_metrics(G, weight=None)
    assert metrics["degree_centrality"].between(0, 1).all()
