"""Entry point: collect company fundamental data (S&P 1500) from WRDS."""

from ceo_sc.data.fundamentals import collect_and_save

if __name__ == "__main__":
    collect_and_save("data/raw/fundamentals.parquet")