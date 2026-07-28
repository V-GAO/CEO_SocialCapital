"""PCA-based composite factor construction from correlated alpha factors.

The social-capital metrics engineered by ``09_feature_engineer_connections.py``
(degree/eigenvector centrality, PageRank, structural holes, constraint,
brokerage -- each under the "cumulative" and "active" edge conventions --
plus raw connection count) are highly correlated with each other (see
``fama_macbeth_multivariate.csv`` from ``11_run_econometrics_and_portfolio.py``,
where no single factor remains significant once the others are controlled
for). PCA collapses these correlated factors into a small number of
orthogonal composite factors that capture most of the shared cross-sectional
variance, which can then be used as new, less-redundant "tradable" factors.

Note: the PCA loadings here are fit once on the whole sample (all years),
which is standard practice for exploratory factor-model construction but
technically uses information from later years to define the composite
weights applied to earlier years. For research/exploration this is an
acceptable simplification; for point-in-time trading-signal construction,
consider re-fitting PCA on an expanding or rolling window of years instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fit_pca_composites(
    df: pd.DataFrame,
    factor_cols: list[str],
    n_components: int = 3,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fit PCA on cross-sectionally (per-``date_col``) z-scored ``factor_cols``
    and return per-row composite scores.

    Standardizing within each date first (rather than PCA's own global
    centering/scaling) keeps composite scores comparable across years and
    prevents the component(s) from being dominated by whichever factor
    happens to have the largest raw cross-sectional variance in any single
    period.

    Rows with a NaN in any ``factor_cols`` are excluded from both the PCA
    fit and the returned scores (left as NaN) -- a missing social-capital
    metric for a CEO/year is informative on its own and shouldn't be
    silently imputed.

    Returns:
        scores: DataFrame indexed like ``df``, columns ``pc1..pcN``.
        loadings: DataFrame (factor_cols x components) of PCA loadings, with
            an extra ``explained_variance_ratio`` row.
        explained_variance_ratio: array of length ``n_components``.
    """
    standardized = pd.DataFrame(index=df.index)
    for col in factor_cols:
        standardized[col] = df.groupby(date_col)[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else np.nan
        )

    mask = standardized.notna().all(axis=1)
    n_valid = int(mask.sum())
    if n_valid < n_components:
        raise ValueError(f"Only {n_valid} complete rows across {len(factor_cols)} factors; "
                         f"cannot fit a {n_components}-component PCA.")

    pc_names = [f"pc{i + 1}" for i in range(n_components)]

    pca = PCA(n_components=n_components)
    fitted_scores = pca.fit_transform(standardized.loc[mask, factor_cols].to_numpy())

    scores = pd.DataFrame(np.nan, index=df.index, columns=pc_names)
    scores.loc[mask, :] = fitted_scores

    loadings = pd.DataFrame(pca.components_.T, index=factor_cols, columns=pc_names)
    loadings.loc["explained_variance_ratio"] = pca.explained_variance_ratio_

    logger.info("PCA fit on %d/%d rows, %d factors -> %d components, explained variance ratio: %s",
                n_valid, len(df), len(factor_cols), n_components,
                np.round(pca.explained_variance_ratio_, 4))

    return scores, loadings, pca.explained_variance_ratio_
