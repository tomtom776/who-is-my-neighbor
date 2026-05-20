"""
Step 2.4 + 2.5: Assign each Census tract to its nearest parish, then dissolve
tracts into parish territory polygons.
Corresponds to notebook 02_tract_assignment.ipynb from the Phase 2 workflow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import pandas as pd
from constants import CRS_PROJECTED

# ── Step 2.4: Nearest-neighbor assignment ────────────────────────────────────

print("=== Step 2.4: Assigning tracts to nearest parish ===")

tracts   = gpd.read_file('data/raw/tracts.geojson').to_crs(CRS_PROJECTED)
parishes = gpd.read_file('data/raw/parishes_archgh.geojson').to_crs(CRS_PROJECTED)

# Standardize the name column (OSM exports 'name')
if 'name' in parishes.columns and 'parish_name' not in parishes.columns:
    parishes = parishes.rename(columns={'name': 'parish_name'})

# Keep parish_id and city alongside parish_name so same-named parishes stay distinct
parishes = parishes[['parish_id', 'parish_name', 'city', 'geometry']].copy()

print(f"Tracts loaded   : {len(tracts)}")
print(f"Parishes loaded : {len(parishes)}")

# Tract centroids (computed in projected CRS — distances in meters)
tracts['centroid'] = tracts.geometry.centroid
centroids = tracts[['GEOID', 'centroid']].copy().set_geometry('centroid')

# Nearest-neighbor spatial join — join on full parish geometry including parish_id
assigned = gpd.sjoin_nearest(
    centroids,
    parishes[['parish_id', 'parish_name', 'city', 'geometry']],
    how='left',
    distance_col='dist_to_parish_m',
)

# Drop duplicates that arise when two parishes are equidistant
assigned = assigned.drop_duplicates(subset='GEOID', keep='first')

# Merge assignment back to full tract GeoDataFrame
tracts = tracts.merge(
    assigned[['GEOID', 'parish_id', 'parish_name', 'city', 'dist_to_parish_m']],
    on='GEOID',
    how='left',
)
tracts = tracts.drop(columns=['centroid'])

assigned_count  = tracts['parish_id'].notna().sum()
unique_parishes = tracts['parish_id'].nunique()
print(f"Tracts assigned   : {assigned_count} of {len(tracts)}")
print(f"Unique parishes   : {unique_parishes}")

# Flag tracts that are very far from any parish (may indicate missing parish point)
far_tracts = tracts[tracts['dist_to_parish_m'] > 15000]
if len(far_tracts) > 0:
    print(f"\nWARNING: {len(far_tracts)} tracts are >15 km from their assigned parish:")
    print(far_tracts[['GEOID', 'parish_id', 'parish_name', 'dist_to_parish_m']].to_string(index=False))

tracts.to_file('data/processed/tracts_assigned.geojson', driver='GeoJSON')
print("\nSaved: data/processed/tracts_assigned.geojson")

# ── Step 2.5: Dissolve tracts into parish territories ────────────────────────

print("\n=== Step 2.5: Dissolving tracts into parish territories ===")

# Dissolve by parish_id (unique) not parish_name (may collide across cities)
parish_territories = (
    tracts
    .dissolve(by='parish_id', aggfunc={'parish_name': 'first', 'city': 'first'})
    .reset_index()
)[['parish_id', 'parish_name', 'city', 'geometry']].copy()

print(f"Parish territory count: {len(parish_territories)}")

# Geometry validity check
invalid = parish_territories[~parish_territories.geometry.is_valid]
print(f"Invalid geometries    : {len(invalid)}")
if len(invalid) > 0:
    parish_territories['geometry'] = parish_territories.geometry.buffer(0)
    print("  Geometry repair applied (buffer(0)).")

parish_territories.to_file('data/processed/parish_territories.geojson', driver='GeoJSON')
print("Saved: data/processed/parish_territories.geojson")

print("\nStep 2.4 + 2.5 complete.")
