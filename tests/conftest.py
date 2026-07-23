"""Shared pytest fixtures: synthetic panel data for testing pure-Python logic
(feature engineering, network metrics, research, econometrics, portfolio
construction) without requiring a live WRDS connection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-31", periods=24, freq="ME")
    entities = [f"E{i:03d}" for i in range(30)]
    industries = ["Tech", "Finance", "Healthcare", "Energy"]

    rows = []
    for date in dates:
        for entity in entities:
            rows.append({
                "date": date,
                "entity_id": entity,
                "industry": industries[hash(entity) % len(industries)],
                "market_cap": rng.lognormal(mean=8, sigma=1.5),
                "raw_factor": rng.normal(0, 1),
                "forward_return": rng.normal(0.01, 0.05),
            })
    df = pd.DataFrame(rows)
    # inject a mild, known relationship between factor and forward return
    df["forward_return"] = df["forward_return"] + 0.02 * df["raw_factor"]
    return df


@pytest.fixture
def synthetic_edges() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    nodes = [f"D{i:03d}" for i in range(20)]
    edges = set()
    while len(edges) < 40:
        a, b = rng.choice(nodes, size=2, replace=False)
        edges.add(tuple(sorted((a, b))))
    return pd.DataFrame(list(edges), columns=["source", "target"])
