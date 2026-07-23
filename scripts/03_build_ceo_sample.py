"""Entry point: build a CEO sample from BoardEx Organization Composition,
restricted to companies resolved in the linking table.

Run after 03_build_linking_table.py.
"""

from ceo_sc.data.ceo_sample import collect_and_save

if __name__ == "__main__":
    collect_and_save()
