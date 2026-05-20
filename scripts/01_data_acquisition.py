"""
Step 2.1 + 2.2: Pull Census tract geometries and ACS demographic data.
Corresponds to notebook 01_data_acquisition.ipynb from the Phase 2 workflow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygris
import geopandas as gpd
import pandas as pd
import requests
from constants import STATE_FIPS, COUNTY_FIPS, CRS_PROJECTED

# ── Step 2.1: Tract geometries ──────────────────────────────────────────────

print("=== Step 2.1: Pulling TIGER/Line tract geometries ===")

tracts = pygris.tracts(state='TX', county=COUNTY_FIPS, year=2022)
tracts = tracts.to_crs(CRS_PROJECTED)
tracts = tracts[['GEOID', 'geometry']].copy()

print(f"Tract count: {len(tracts)}")

tracts.to_file('data/raw/tracts.geojson', driver='GeoJSON')
print("Saved: data/raw/tracts.geojson")

# ── Step 2.2: ACS demographic data ──────────────────────────────────────────

print("\n=== Step 2.2: Pulling ACS 5-year estimates ===")

API_KEY = os.environ.get('CENSUS_API_KEY')
if not API_KEY:
    raise EnvironmentError("CENSUS_API_KEY environment variable is not set.")

YEAR = 2022

ACS_VARS = {
    'B01001_001E': 'total_pop',
    'B01001_007E': 'male_20_24',
    'B01001_008E': 'male_25_29',
    'B01001_009E': 'male_30_34',
    'B01001_031E': 'female_20_24',
    'B01001_032E': 'female_25_29',
    'B01001_033E': 'female_30_34',
    'B01001_015E': 'male_45_49',
    'B01001_016E': 'male_50_54',
    'B01001_017E': 'male_55_59',
    'B01001_018E': 'male_60_64',
    'B01001_039E': 'female_45_49',
    'B01001_040E': 'female_50_54',
    'B01001_041E': 'female_55_59',
    'B01001_042E': 'female_60_64',
    'B01001_020E': 'male_65_74',
    'B01001_021E': 'male_75_84',
    'B01001_044E': 'female_65_74',
    'B01001_045E': 'female_75_84',
    'B11001_001E': 'total_households',
    'B11001_002E': 'family_households',
    'B11001_007E': 'nonfamily_households',
    'B11001_008E': 'single_person_households',
    'B19013_001E': 'median_hh_income',
    'B17001_001E': 'poverty_universe',
    'B17001_002E': 'below_poverty',
    'B05001_001E': 'citizenship_universe',
    'B05001_006E': 'noncitizen',
    'B05006_001E': 'foreign_born_total',
    'B16004_001E': 'language_universe',
    'B16004_003E': 'spanish_speak_very_well',
    'B16004_004E': 'spanish_speak_well',
    'B16004_005E': 'spanish_speak_not_well',
    'B03003_003E': 'hispanic_or_latino',
    'B15003_001E': 'education_universe',
    'B15003_022E': 'bachelors_degree',
    'B15003_023E': 'masters_degree',
    'B15003_024E': 'professional_degree',
    'B15003_025E': 'doctoral_degree',
    'B25003_001E': 'tenure_universe',
    'B25003_002E': 'owner_occupied',
    'B25003_003E': 'renter_occupied',
    'B18101_001E': 'disability_universe',
    'B18101_004E': 'disability_male_18_34',
    'B18101_007E': 'disability_male_35_64',
    'B18101_010E': 'disability_male_65_74',
    'B18101_023E': 'disability_female_18_34',
    'B18101_026E': 'disability_female_35_64',
    'B18101_029E': 'disability_female_65_74',
    'B27001_001E': 'insurance_universe',
    'B27001_005E': 'uninsured_male_18_24',
    'B27001_008E': 'uninsured_male_25_34',
    'B27001_033E': 'uninsured_female_18_24',
    'B27001_036E': 'uninsured_female_25_34',
}

# Census API limit is 50 variables per request; split into batches of 49
all_var_keys = list(ACS_VARS.keys())
batch_size = 49
batches = [all_var_keys[i:i + batch_size] for i in range(0, len(all_var_keys), batch_size)]

print(f"Variables: {len(all_var_keys)}, batches: {len(batches)}")

batch_dfs = []
for batch_num, batch_vars in enumerate(batches, 1):
    var_string = ','.join(batch_vars)
    url = (
        f"https://api.census.gov/data/{YEAR}/acs/acs5"
        f"?get={var_string}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}"
        f"&key={API_KEY}"
    )
    print(f"  Batch {batch_num}/{len(batches)} ({len(batch_vars)} vars)...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    batch_dfs.append(df)

# Merge batches on the geography columns
geo_cols = ['state', 'county', 'tract']
acs_df = batch_dfs[0]
for df in batch_dfs[1:]:
    acs_df = acs_df.merge(df, on=geo_cols, how='outer')

acs_df = acs_df.rename(columns=ACS_VARS)
acs_df['GEOID'] = acs_df['state'] + acs_df['county'] + acs_df['tract']

acs_df = acs_df[acs_df['county'].isin(COUNTY_FIPS)].copy()

numeric_cols = list(ACS_VARS.values())
for col in numeric_cols:
    if col in acs_df.columns:
        acs_df[col] = pd.to_numeric(acs_df[col], errors='coerce')

acs_df = acs_df.replace(-666666666, pd.NA)

print(f"ACS tract count: {len(acs_df)}")

acs_df.to_csv('data/raw/acs_raw.csv', index=False)
print("Saved: data/raw/acs_raw.csv")

# ── QA check ────────────────────────────────────────────────────────────────

tract_geoids = set(tracts['GEOID'])
acs_geoids   = set(acs_df['GEOID'])

in_geom_not_acs = tract_geoids - acs_geoids
in_acs_not_geom = acs_geoids - tract_geoids

print(f"\n=== QA ===")
print(f"Tracts in geometry, missing from ACS : {len(in_geom_not_acs)}")
print(f"Tracts in ACS, missing from geometry : {len(in_acs_not_geom)}")

if in_geom_not_acs:
    print(f"  Sample GEOIDs missing from ACS: {list(in_geom_not_acs)[:5]}")
if in_acs_not_geom:
    print(f"  Sample GEOIDs missing from geometry: {list(in_acs_not_geom)[:5]}")

print("\nStep 2.1 + 2.2 complete.")
