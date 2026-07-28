# CEO_SC

Production-style research pipeline for the **CEO Social Capital (CEO_SC)** alpha
factor: does a CEO's position in the executive/director social network
(BoardEx) predict company stock returns? Ispired by the research paper \link{https://drive.google.com/file/d/1t5ypOE0pggnoA5jMa695AGZmm9vA-CU1/view}

## Project structure

```
configs/                  YAML configs (data sources, feature engineering, backtest)
src/ceo_sc/
  data/                    WRDS data collection (fundamentals, BoardEx CEO_SC, linking, CRSP)
  network/                 Social network metrics (NetworkX: centrality, constraint, brokerage)
  features/                Feature engineering (winsorize, zscore, neutralize, lag, rank)
  research/                IC / Rank IC, factor decay, quantile & long-short portfolios
  econometrics/            Fama-MacBeth, robustness, subsample, OOS, significance testing
  portfolio/               Rebalancing, weighting, turnover, attribution, risk metrics
  viz/                     Research-quality plots
  utils/                   Config loading, logging
scripts/                   Numbered pipeline entry points (01_... through 08_...)
tests/                     pytest unit tests (synthetic data, no WRDS required)
data/                      Local raw/processed data (git-ignored)
research_summary/          Jupyter notebook with research results and visualizations
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # then fill in WRDS_USERNAME / WRDS_PASSWORD
```

## Running the pipeline

```powershell
python scripts/01_collect_fundamentals.py
python scripts/02_build_linking_table.py
python scripts/03_build_ceo_sample.py
python scripts/04_collect_ceo_sc.py
python scripts/05_collect_backtest_data.py
python scripts/06_build_connection_panel.py
python scripts/07_merge_panel.py
python scripts/08_merge_returns.py
python scripts/09_feature_engineering_connections.py
python scripts/10_run_research.py
python scripts/11_run_econometrics_and_portfolio.py
```

## Tests

```powershell
pytest
```
