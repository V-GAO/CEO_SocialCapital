"""Entry point: PCA-based composite social-capital factors.

The social-capital metrics engineered by 09_feature_engineer_connections.py
are highly correlated with each other (see fama_macbeth_multivariate.csv
from 11_run_econometrics_and_portfolio.py, where no single factor remains
significant once the others are controlled for). This script fits PCA on
the point-in-time-safe, cross-sectionally standardized "*_lagged" factor
values and extracts a small number of orthogonal composite factors, then
cross-sectionally ranks each component into new tradable "pcN_rank"
factors that plug directly into 10_run_research.py and
11_run_econometrics_and_portfolio.py (both auto-discover every "*_rank"
column in the panel).

Must be re-run after 08_merge_returns.py (and before 10/11), since it reads
and re-saves panel_with_returns.parquet with the added composite columns --
re-running 08 will remove them again until this script is re-run.

See src/ceo_sc/features/pca.py for the PCA fitting logic and its
point-in-time caveats.

Output: overwrites data/processed/panel_with_returns.parquet with added
pc1..pcN and pc1_rank..pcN_rank columns, plus data/processed/pca_loadings.csv.
"""

import pandas as pd

from ceo_sc.features.engineering import cross_sectional_rank, validate_feature
from ceo_sc.features.pca import fit_pca_composites
from ceo_sc.utils.config import project_root
from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

DATE_COL = "year"
N_COMPONENTS = 3


def main() -> None:
    panel_path = project_root() / "data" / "processed" / "panel_with_returns.parquet"
    if not panel_path.exists():
        logger.error("Missing panel at %s. Run 08_merge_returns.py first.", panel_path)
        return

    df = pd.read_parquet(panel_path)

    factor_cols = sorted(c for c in df.columns if c.endswith("_lagged"))
    if not factor_cols:
        logger.error("No '*_lagged' factor columns found in panel. Run "
                     "09_feature_engineer_connections.py first.")
        return
    logger.info("Fitting %d-component PCA from %d lagged factor columns: %s",
                N_COMPONENTS, len(factor_cols), factor_cols)

    scores, loadings, explained = fit_pca_composites(df, factor_cols, n_components=N_COMPONENTS, date_col=DATE_COL)

    out_dir = project_root() / "data" / "processed"
    loadings.to_csv(out_dir / "pca_loadings.csv")
    logger.info("Saved PCA loadings/explained-variance to %s", out_dir / "pca_loadings.csv")

    for col in scores.columns:
        df[col] = scores[col]
        df[f"{col}_rank"] = cross_sectional_rank(df, col, date_col=DATE_COL, pct=True)
        validate_feature(df[f"{col}_rank"], name=f"{col}_rank", max_nan_frac=0.9)

    df.to_parquet(panel_path)
    logger.info("Saved panel with %d PCA composite factors (%s, and their *_rank) to %s",
                N_COMPONENTS, list(scores.columns), panel_path)


if __name__ == "__main__":
    main()
