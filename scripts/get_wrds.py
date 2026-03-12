import pandas as pd
import s3fs
import numpy as np
import wrds

db = wrds.Connection(wrds_username='vincent0358')

query_ccm = """
SELECT gvkey,
       lpermno AS permno,
       lpermco AS permco,
       linktype,
       linkprim,
       linkdt,
       linkenddt
FROM crsp.ccmxpf_linktable
WHERE (linktype IN ('LU', 'LC'))
    AND linkprim IN ('P', 'C')
    AND usedflag = 1
"""

query_crsp_m = f"""
SELECT permno, date, ret, shrout, prc
FROM crsp.msf
WHERE permno IN ({permno_list})
    AND date >= '2004-12-31'
"""

query_comp_a = """
SELECT gvkey, datadate, at, ceq, sale, ni
FROM comp.funda
WHERE indfmt = 'INDL'
    AND datadate >= '2004-12-31'
    AND datafmt = 'STD'
    AND popsrc = 'D'
    AND consol = 'C'
"""

def get_wrds(permno_list, queries):
    permno_list = permno_list
    # queries can be tuple of strings
