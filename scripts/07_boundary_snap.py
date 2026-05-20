"""
Step 2.7: Snap parish territory boundaries to major TxDOT roads.

Loads TxDOT_Roadways.shp, filters for on-system major routes (IH/US/SH/SL),
then uses Shapely's snap() to pull territory polygon boundaries onto nearby
road lines within SNAP_TOLERANCE meters.

Input : data/processed/parish_territories.geojson
Output: data/processed/parish_territories_snapped.geojson
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union, snap
import warnings
warnings.filterwarnings('ignore', message='.*Measured.*')

from constants import CRS_PROJECTED, SNAP_TOLERANCE

MAJOR_PREFIXES = {'IH', 'US', 'SH', 'SL'}

# ── Load parish territories ───────────────────────────────────────────────────

print("=== Step 2.7: Snapping boundaries to major roads ===")

territories = gpd.read_file('data/processed/parish_territories.geojson').to_crs(CRS_PROJECTED)
print(f"Parish territories loaded: {len(territories)}")

study_bbox = territories.total_bounds  # (minx, miny, maxx, maxy) in projected CRS
study_box_wgs = (
    gpd.GeoDataFrame(geometry=[territories.to_crs('EPSG:3857').unary_union.envelope],
                     crs='EPSG:3857')
    .to_crs('EPSG:3857')
    .total_bounds
)

# ── Load and filter TxDOT roads ───────────────────────────────────────────────

print("Loading TxDOT_Roadways.shp (this may take a moment)...")

# bbox filter in the shapefile's native CRS (EPSG:3857) avoids loading all 277 MB
bbox_3857 = (
    gpd.GeoDataFrame(geometry=gpd.GeoSeries(territories.to_crs('EPSG:3857').unary_union.envelope),
                     crs='EPSG:3857')
    .total_bounds
)
roads_raw = gpd.read_file(
    'data/raw/TxDOT_Roadways.shp',
    bbox=tuple(bbox_3857),
)
print(f"  Rows in bounding box: {len(roads_raw)}")

major_roads = roads_raw[
    (roads_raw['SYSTEM'] == 'On') &
    (roads_raw['RTE_PRFX'].isin(MAJOR_PREFIXES))
].copy()
print(f"  Major-road segments (IH/US/SH/SL, on-system): {len(major_roads)}")

if len(major_roads) == 0:
    print("ERROR: No major road segments found. Check SYSTEM/RTE_PRFX column values.")
    sys.exit(1)

major_roads = major_roads.to_crs(CRS_PROJECTED)
road_union = unary_union(major_roads.geometry)
print(f"  Road union built.")

# ── Snap each territory boundary to road network ──────────────────────────────

print("Snapping territory boundaries...")

snapped_geoms = []
for idx, row in territories.iterrows():
    geom = row.geometry
    snapped = snap(geom, road_union, tolerance=SNAP_TOLERANCE)
    if not snapped.is_valid:
        snapped = snapped.buffer(0)
    snapped_geoms.append(snapped)

territories['geometry'] = snapped_geoms
territories = territories.set_geometry('geometry')

invalid_count = (~territories.geometry.is_valid).sum()
if invalid_count > 0:
    print(f"WARNING: {invalid_count} invalid geometries after snap — applying buffer(0) repair.")
    territories['geometry'] = territories.geometry.apply(
        lambda g: g.buffer(0) if not g.is_valid else g
    )

# ── Save ─────────────────────────────────────────────────────────────────────

territories.to_file('data/processed/parish_territories_snapped.geojson', driver='GeoJSON')
print(f"Saved: data/processed/parish_territories_snapped.geojson")
print(f"  Parishes: {len(territories)}")
print("\nStep 2.7 complete.")
