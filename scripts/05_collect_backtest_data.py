"""Entry point: collect backtesting return data (CRSP) from WRDS."""

from ceo_sc.data.backtest_data import collect_and_save

if __name__ == "__main__":
    collect_and_save("data/raw/backtest_returns.parquet")
