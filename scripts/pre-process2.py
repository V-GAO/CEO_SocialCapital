import pandas as pd
import s3fs
import numpy as np

trucost_use = pd.read_csv('s3://buc-vin0358/trucost_use.csv')

def quantile_sort(group, cols):
    result = group.copy()
    for col in cols:
        result[col] = pd.qcut(group[col], q=5, labels=False, duplicates='drop')
    return result

# perform nested sorting by FinancialYear, Country, GICS_Industry_Name
result = trucost_use.groupby(['FinancialYear', 'Country', 'GICS_Industry_Name']).apply(
    lambda x: quantile_sort(x, cols_to_sort)
    ).reset_index(drop=True)

result.to_csv('s3://buc-vin0358/trucost_use_sorted_countryIndustry.csv', index=False)