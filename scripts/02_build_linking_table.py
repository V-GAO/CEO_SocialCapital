"""Entry point: build the company <-> CEO/director linking table."""

from ceo_sc.data.linking import collect_and_save

if __name__ == "__main__":
    collect_and_save("data/raw/linking_table.parquet")
