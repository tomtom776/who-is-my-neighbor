"""
Step 2.8: Join LQ scores to parish territory polygons, simplify geometry,
and export the final GeoJSON for the Phase 3 web map.

Uses unsnapped dissolved boundaries (data/processed/parish_territories.geojson)
directly — TxDOT road snapping was skipped due to topology issues it introduced.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import pandas as pd
from constants import CRS_GEOGRAPHIC, SIMPLIFY_TOLERANCE

print("=== Step 2.8: Producing final output GeoJSON ===")

_territories_path = 'data/processed/parish_territories.geojson'
print(f"Loading territories from: {_territories_path}")
parish_territories = gpd.read_file(_territories_path)
parish_acs = pd.read_csv('data/processed/parish_acs.csv')

lq_cols  = [f'lq_s{i}' for i in range(1, 8)]
pct_cols = [
    'pct_age_18_34', 'pct_single_hh',
    'pct_family_hh', 'pct_age_25_44_approx',
    'pct_hispanic', 'pct_foreign_born', 'pct_spanish',
    'pct_below_poverty', 'pct_uninsured', 'pct_renter',
    'pct_age_45_64', 'pct_owner',
    'pct_age_65_plus', 'pct_disability',
    'pct_college',
    'median_hh_income_wtd', 'total_pop', 'total_households',
]

# Only join columns that exist in parish_acs
available_pct = [c for c in pct_cols if c in parish_acs.columns]
join_cols = ['parish_name'] + lq_cols + available_pct

final = parish_territories.merge(
    parish_acs[join_cols],
    on='parish_name',
    how='left',
)

# Round LQ values and percentage fields
for col in lq_cols:
    final[col] = final[col].round(3)
for col in available_pct:
    if col in final.columns:
        final[col] = final[col].round(4)

# Simplify geometry for web performance (preserve_topology prevents gaps)
final['geometry'] = final.geometry.simplify(
    tolerance=SIMPLIFY_TOLERANCE,
    preserve_topology=True,
)

# Reproject to WGS84 for GeoJSON export
final = final.to_crs(CRS_GEOGRAPHIC)

# Repair any invalids introduced by simplification or reprojection
invalid_mask = ~final.geometry.is_valid
if invalid_mask.any():
    print(f"Repairing {invalid_mask.sum()} invalid geometries (buffer(0)).")
    final.loc[invalid_mask, 'geometry'] = final.loc[invalid_mask, 'geometry'].buffer(0)

# Validity checks
assert final.geometry.is_valid.all(), "Invalid geometries in final output — fix before export"
assert final['parish_name'].notna().all(), "Null parish names in final output"

null_lq = final[lq_cols].isnull().any(axis=1).sum()
if null_lq > 0:
    print(f"WARNING: {null_lq} parishes have null LQ values (check ACS join).")

final.to_file('data/output/parish_territories_final.geojson', driver='GeoJSON')

approx_kb = len(final.to_json()) / 1024
print("Export complete.")
print(f"  Parishes  : {len(final)}")
print(f"  Fields    : {list(final.columns)}")
print(f"  File size : ~{approx_kb:.0f} KB")
print("\nStep 2.8 complete.")
