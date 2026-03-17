import pandas as pd
import s3fs
import numpy as np
import wrds

# credentials and files
db = wrds.Connection(wrds_username='vincent0358')
id_file = "s3://buc-vin0358/US_isin_cusip_permno_ticker_expand.csv"

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
WHERE permno IN ({permnos})
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

isin_cusip_expand = pd.read_csv(id_file)
unique_id = isin_cusip_expand['permno'].unique()
unique_id_df = pd.DataFrame(unique_id)
unique_id_df.columns = ['permno']
permno_list = ','.join(str(key) for key in unique_id_df['permno'])

def get_wrds(
    db,
    permnos=permno_list,
    start_date='2004-12-31',
    end_date='2025-12-31',
    data_freq='annual' # data_freq can be annual(funda) or quarterly(fundq); for fundamental data
):
    #permno_list = permno_list
    # queries can be tuple of strings
    #queries = queries

    # 1. Load CCM link table
    ccm = db.raw_sql(query_ccm)
    ccm = ccm.dropna(subset=['permno'])
    ccm['permno'] = ccm['permno'].astype(int)

    # 2. Load CRSP monthly data
    if permnos is not None:
        #placeholders = ",".(join["%s"] * len(permnos))
        crsp_m = db.raw_sql(query_crsp_m)
    else:
        None

    crsp_m['permno'] = crsp_m['permno'].astype(int)
    
    # 3. Load COMPUSTAT fundamentals
    table = "comp.funda" if data_freq=='annual' else 'comp.fundq'

    comp = db.raw_sql(query_comp_a)

    # Merge the tables fetched
    comp_ccm = comp.merge(ccm, on='gvkey', how='left')
    
    comp_ccm = comp_ccm[
        (comp_ccm['datadate'] >= comp_ccm['linkdt'])&
        ((comp_ccm['datadate'] <= comp_ccm['linkenddt']) | comp_ccm['linkenddt'].isna())
    ]

    comp_ccm = comp_ccm.dropna(subset=['permno'])

        # merge comp_ccm with crsp on permno
    merged = comp_ccm.merge(crsp_m, on='permno', how='right')

    # only keep monthly data
    merged['date_dt'] = pd.to_datetime(merged['date'], errors='coerce')
    merged['year'] = merged['date_dt'].dt.year
    merged['month'] = merged['date_dt'].dt.month

    # dropna
    merged = merged.sort_values(by=['gvkey', 'date_dt'])
    merged_use = merged.dropna(subset=['gvkey', 'year', 'month', 'ret', 'shrout', 'prc', 'at', 'ceq', 'sale', 'ni'])
    # drop duplicates
    merged_use = merged_use.drop_duplicates(subset=['gvkey', 'permno', 'year', 'month'], keep='first')

    return merged_use # return clean df

get_wrds(db, permnos=permno_list)
merge_use.to_csv("s3://buc-vin0358/wrds_merged_firstTry.csv")

