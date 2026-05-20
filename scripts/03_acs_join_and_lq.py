"""
Step 2.6: Join ACS data to parish territories, build composite segment indices,
and calculate location quotients for all 7 ministry segments.
Corresponds to notebook 03_acs_join_and_lq.ipynb from the Phase 2 workflow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import pandas as pd
import numpy as np
from constants import CRS_PROJECTED

print("=== Step 2.6: ACS join and location quotient calculation ===")

tracts  = gpd.read_file('data/processed/tracts_assigned.geojson')
acs_df  = pd.read_csv('data/raw/acs_raw.csv', dtype={'GEOID': str})

# Join ACS to tract geometries
tracts_acs = tracts.merge(acs_df, on='GEOID', how='left')

count_vars = [
    'total_pop', 'total_households',
    'male_20_24', 'male_25_29', 'male_30_34',
    'female_20_24', 'female_25_29', 'female_30_34',
    'male_45_49', 'male_50_54', 'male_55_59', 'male_60_64',
    'female_45_49', 'female_50_54', 'female_55_59', 'female_60_64',
    'male_65_74', 'male_75_84', 'female_65_74', 'female_75_84',
    'family_households', 'nonfamily_households', 'single_person_households',
    'poverty_universe', 'below_poverty',
    'citizenship_universe', 'noncitizen', 'foreign_born_total',
    'language_universe', 'spanish_speak_very_well', 'spanish_speak_well',
    'spanish_speak_not_well', 'hispanic_or_latino',
    'education_universe', 'bachelors_degree', 'masters_degree',
    'professional_degree', 'doctoral_degree',
    'tenure_universe', 'owner_occupied', 'renter_occupied',
    'disability_universe', 'disability_male_18_34', 'disability_male_35_64',
    'disability_male_65_74', 'disability_female_18_34',
    'disability_female_35_64', 'disability_female_65_74',
    'insurance_universe', 'uninsured_male_18_24', 'uninsured_male_25_34',
    'uninsured_female_18_24', 'uninsured_female_25_34',
]

# Only sum columns that actually exist in the merged dataframe
count_vars = [c for c in count_vars if c in tracts_acs.columns]

parish_acs = (
    tracts_acs
    .groupby('parish_name')[count_vars]
    .sum()
    .reset_index()
)

# Weighted average for median household income
tracts_acs['income_weight'] = (
    tracts_acs['median_hh_income'] * tracts_acs['total_households']
)
income_agg = (
    tracts_acs
    .groupby('parish_name')[['income_weight', 'total_households']]
    .sum()
    .reset_index()
)
income_agg['median_hh_income_wtd'] = (
    income_agg['income_weight'] / income_agg['total_households']
)
parish_acs = parish_acs.merge(
    income_agg[['parish_name', 'median_hh_income_wtd']],
    on='parish_name',
)

# ── Segment indices ──────────────────────────────────────────────────────────

def safe_div(num, denom):
    return np.where(denom > 0, num / denom, 0)

# S1: Young unattached adults
parish_acs['pct_age_18_34'] = safe_div(
    parish_acs['male_20_24'] + parish_acs['male_25_29'] + parish_acs['male_30_34'] +
    parish_acs['female_20_24'] + parish_acs['female_25_29'] + parish_acs['female_30_34'],
    parish_acs['total_pop']
)
parish_acs['pct_single_hh'] = safe_div(
    parish_acs['single_person_households'],
    parish_acs['total_households']
)
parish_acs['idx_s1'] = (parish_acs['pct_age_18_34'] + parish_acs['pct_single_hh']) / 2

# S2: Young families
parish_acs['pct_family_hh'] = safe_div(
    parish_acs['family_households'],
    parish_acs['total_households']
)
parish_acs['pct_age_25_44_approx'] = safe_div(
    parish_acs['male_25_29'] + parish_acs['male_30_34'] +
    parish_acs['female_25_29'] + parish_acs['female_30_34'],
    parish_acs['total_pop']
)
parish_acs['idx_s2'] = (parish_acs['pct_family_hh'] + parish_acs['pct_age_25_44_approx']) / 2

# S3: Hispanic / immigrant
parish_acs['pct_hispanic'] = safe_div(
    parish_acs['hispanic_or_latino'],
    parish_acs['total_pop']
)
parish_acs['pct_foreign_born'] = safe_div(
    parish_acs['foreign_born_total'],
    parish_acs['total_pop']
)
parish_acs['pct_spanish'] = safe_div(
    parish_acs['spanish_speak_very_well'] + parish_acs['spanish_speak_well'] +
    parish_acs['spanish_speak_not_well'],
    parish_acs['language_universe']
)
parish_acs['idx_s3'] = (
    parish_acs['pct_hispanic'] + parish_acs['pct_foreign_born'] + parish_acs['pct_spanish']
) / 3

# S4: Economically vulnerable
parish_acs['pct_below_poverty'] = safe_div(
    parish_acs['below_poverty'],
    parish_acs['poverty_universe']
)
parish_acs['pct_uninsured'] = safe_div(
    parish_acs['uninsured_male_18_24'] + parish_acs['uninsured_male_25_34'] +
    parish_acs['uninsured_female_18_24'] + parish_acs['uninsured_female_25_34'],
    parish_acs['insurance_universe']
)
parish_acs['pct_renter'] = safe_div(
    parish_acs['renter_occupied'],
    parish_acs['tenure_universe']
)
income_min = parish_acs['median_hh_income_wtd'].min()
income_max = parish_acs['median_hh_income_wtd'].max()
parish_acs['income_vulnerability'] = 1 - (
    (parish_acs['median_hh_income_wtd'] - income_min) / (income_max - income_min)
)
parish_acs['idx_s4'] = (
    parish_acs['pct_below_poverty'] + parish_acs['pct_uninsured'] +
    parish_acs['pct_renter'] + parish_acs['income_vulnerability']
) / 4

# S5: Established older families
parish_acs['pct_age_45_64'] = safe_div(
    parish_acs['male_45_49'] + parish_acs['male_50_54'] +
    parish_acs['male_55_59'] + parish_acs['male_60_64'] +
    parish_acs['female_45_49'] + parish_acs['female_50_54'] +
    parish_acs['female_55_59'] + parish_acs['female_60_64'],
    parish_acs['total_pop']
)
parish_acs['pct_owner'] = safe_div(
    parish_acs['owner_occupied'],
    parish_acs['tenure_universe']
)
parish_acs['idx_s5'] = (
    parish_acs['pct_age_45_64'] + parish_acs['pct_family_hh'] + parish_acs['pct_owner']
) / 3

# S6: Senior adults
parish_acs['pct_age_65_plus'] = safe_div(
    parish_acs['male_65_74'] + parish_acs['male_75_84'] +
    parish_acs['female_65_74'] + parish_acs['female_75_84'],
    parish_acs['total_pop']
)
parish_acs['pct_disability'] = safe_div(
    parish_acs['disability_male_65_74'] + parish_acs['disability_female_65_74'],
    parish_acs['disability_universe']
)
parish_acs['idx_s6'] = (parish_acs['pct_age_65_plus'] + parish_acs['pct_disability']) / 2

# S7: Educated young professionals
parish_acs['pct_college'] = safe_div(
    parish_acs['bachelors_degree'] + parish_acs['masters_degree'] +
    parish_acs['professional_degree'] + parish_acs['doctoral_degree'],
    parish_acs['education_universe']
)
parish_acs['idx_s7'] = (
    parish_acs['pct_college'] + parish_acs['pct_renter'] + parish_acs['pct_age_25_44_approx']
) / 3

# ── Location quotients ───────────────────────────────────────────────────────

print("\nSegment location quotient summary:")
for sid in ['s1', 's2', 's3', 's4', 's5', 's6', 's7']:
    idx_col = f'idx_{sid}'
    lq_col  = f'lq_{sid}'
    diocese_avg = parish_acs[idx_col].mean()
    parish_acs[lq_col] = parish_acs[idx_col] / diocese_avg
    print(f"  Segment {sid} — diocese avg index: {diocese_avg:.4f}  "
          f"(LQ range: {parish_acs[lq_col].min():.2f}–{parish_acs[lq_col].max():.2f})")

parish_acs.to_csv('data/processed/parish_acs.csv', index=False)
print("\nSaved: data/processed/parish_acs.csv")
print(f"Parishes in output: {len(parish_acs)}")
print("\nStep 2.6 complete.")
