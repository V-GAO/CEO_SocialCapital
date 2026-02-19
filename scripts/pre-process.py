import pandas as pd
import s3fs
import numpy as np

file = "s3://buc-vin0358/Data_Trucost.csv"
df = pd.read_csv(file)

cols_origin = ['TCUID',
'Company',
'ISIN',
'Financial Year',
'GICS Sector Code',
'GICS Sector Name',
'GICS Industry Group Code',
'GICS Industry Group Name',
'GICS Industry Code',
'GICS Industry Name',
'GICS Sub Industry Code',
'GICS Sub Industry Name',
'GICS Description',
'Country',
'Carbon-Scope 1  (tonnes CO2e)',
'Carbon-Scope 2  (tonnes CO2e)',
'Carbon-Scope 3 (tonnes CO2e)',
'Carbon-First Tier Indirect (tonnes CO2e)',
'Carbon-Direct+First Tier Indirect (tonnes CO2e)',
'Carbon Intensity-Scope 1 (tonnes CO2e/USD mn)',
'Carbon Intensity-Scope 2 (tonnes CO2e/USD mn)',
'Carbon Intensity-Scope 3 (tonnes CO2e/USD mn)',
'Carbon Intensity-Direct (tonnes CO2e/USD mn)',
'Carbon Intensity-First Tier Indirect (tonnes CO2e/USD mn)',
'Carbon Intensity-Direct+First Tier Indirect (tonnes CO2e/USD mn)',
'Total-Direct (USD mn)',
'Total-Indirect (USD mn)',
'Total-Direct+Indirect (USD mn)',
'GHG-Direct (USD mn)',
'GHG-Indirect (USD mn)',
'GHG-Total (USD mn)',
'GHG-Direct Impact Ratio (%)',
'GHG-Indirect Impact Ratio (%)',
'GHG-Total Impact Ratio (%)']
cols_rename = {
    'TCUID': 'TCUID',
'Company': 'Company',
'ISIN': 'ISIN',
'Financial Year': 'FinancialYear',
'GICS Sector Code': 'GICS_Sector_Code',
'GICS Sector Name': 'GICS_Sector_Name',
'GICS Industry Group Code': 'GICS_Industry_Group_Code',
'GICS Industry Group Name': 'GICS_Industry_Group_Name',
'GICS Industry Code': 'GICS_Industry_Code',
'GICS Industry Name': 'GICS_Industry_Name',
'GICS Sub Industry Code': 'GICS_Sub_Industry_Code',
'GICS Sub Industry Name': 'GICS_Sub_Industry_Name',
'Country': 'Country',
'Carbon-Scope 1  (tonnes CO2e)': 'CarbonScope1(CO2e)',
'Carbon-Scope 2  (tonnes CO2e)': 'CarbonScope2(CO2e)',
'Carbon-Scope 3 (tonnes CO2e)': 'CarbonScope3(CO2e)',
'Carbon-First Tier Indirect (tonnes CO2e)': 'Carbon_FirstTierIndirect(CO2e)',
'Carbon-Direct+First Tier Indirect (tonnes CO2e)': 'Carbon_DirectAndFirstTierIndirect(CO2e)',
'Carbon Intensity-Scope 1 (tonnes CO2e/USD mn)': 'CarbonIntensityScope1(CO2e/USDmn)',
'Carbon Intensity-Scope 2 (tonnes CO2e/USD mn)': 'CarbonIntensityScope2(CO2e/USDmn)',
'Carbon Intensity-Scope 3 (tonnes CO2e/USD mn)': 'CarbonIntensityScope3(CO2e/USDmn)',
'Carbon Intensity-Direct (tonnes CO2e/USD mn)': 'CarbonIntensityDirect(CO2e/USDmn)',
'Carbon Intensity-First Tier Indirect (tonnes CO2e/USD mn)': 'CarbonIntensityFirstTierIndirect(CO2e/USDmn)',
'Carbon Intensity-Direct+First Tier Indirect (tonnes CO2e/USD mn)': 'CarbonIntensity_DirectAndFirstTierIndirect(CO2e/USDmn)',
'Total-Direct (USD mn)': 'TotalDirect(USDmn)',
'Total-Indirect (USD mn)': 'TotalIndirect(USDmn)',
'Total-Direct+Indirect (USD mn)': 'Total_DirectAndIndirect(USDmn)',
'GHG-Direct (USD mn)': 'GHG_Direct(USDmn)',
'GHG-Indirect (USD mn)': 'GHG_Indirect(USDmn)',
'GHG-Total (USD mn)': 'GHG_Total(USDmn)',
'GHG-Direct Impact Ratio (%)': 'GHG_DirectImpactRatio',
'GHG-Indirect Impact Ratio (%)': 'GHG_IndirectImpactRatio',
'GHG-Total Impact Ratio (%)': 'GHG_TotalImpactRatio'
}

trucost_use = df[cols_origin]
trucost_use = trucost_use.rename(columns=cols_rename)
cols_to_sort = list(trucost_use.iloc[:, 14:34])
trucost_use = trucost_use.dropna(subset=cols_to_sort, how='all')

trucost_use.to_csv('s3://buc-vin0358/trucost_use.csv', index=False)

def quantile_sort(group, cols):
    result = group.copy()
    for col in cols:
        result[col] = pd.qcut(group[col], q=5, labels=False, duplicates='drop')
    return result

# perform nested sorting by FinancialYear, Country, GICS_Industry_Name
result = trucost_use.groupby(['FinancialYear', 'Country', 'GICS_Industry_Name']).apply(
    lambda x: quantile_sort(x, cols_to_sort)
    ).reset_index(drop=True)

result.to_csv('s3://buc-vin0358/trucost_use_sorted.csv', index=False)