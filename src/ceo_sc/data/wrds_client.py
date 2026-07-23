"""Thin wrapper around the WRDS Python API.

Credentials are read from environment variables (``WRDS_USERNAME`` /
``WRDS_PASSWORD``) via a local ``.env`` file (see ``.env.example``).
Never hard-code credentials into source files or configs.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional, Union

import pandas as pd
from dotenv import load_dotenv

from ceo_sc.utils.logging_utils import get_logger

logger = get_logger(__name__)

load_dotenv()


class WRDSClient:
    """Lazily-connecting wrapper around ``wrds.Connection``.

    Usage
    -----
    >>> with WRDSClient() as db:
    ...     df = db.query("SELECT * FROM comp.company LIMIT 10")
    """

    def __init__(self, username: Optional[str] = None) -> None:
        self._username = username or os.environ.get("WRDS_USERNAME") or None
        self._conn: Any = None

    def connect(self) -> "WRDSClient":
        if self._conn is not None:
            return self
        import wrds  # imported lazily so the package is optional at import time

        logger.info("Connecting to WRDS (username=%s)...", self._username or "<from .pgpass>")
        self._conn = wrds.Connection(wrds_username=self._username)
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "WRDSClient":
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def query(self, sql: str, params: Optional[dict[str, Any]] = None,
              chunksize: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Run a query against WRDS.

        If ``chunksize`` is given, returns an iterator of DataFrames (each up
        to ``chunksize`` rows) instead of materializing the full result set
        in memory at once. Use this for large result sets to avoid OOM.
        """
        if self._conn is None:
            self.connect()
        return self._conn.raw_sql(
            sql, params=params, chunksize=chunksize,
            return_iter=chunksize is not None,
        )

    def get_table(
        self,
        library: str,
        table: str,
        columns: Optional[list[str]] = None,
        obs: Optional[int] = None,
    ) -> pd.DataFrame:
        if self._conn is None:
            self.connect()
        return self._conn.get_table(library=library, table=table, columns=columns, obs=obs)
