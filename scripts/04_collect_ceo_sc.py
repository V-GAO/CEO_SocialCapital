"""Entry point: collect CEO social capital (BoardEx) data from WRDS in chunks."""

from ceo_sc.data.ceo_social_capital import collect_ceo_sc

if __name__ == "__main__":
    collect_ceo_sc()
