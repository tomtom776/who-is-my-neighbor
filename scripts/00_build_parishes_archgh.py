"""
Converts the authoritative Archdiocese of Galveston-Houston parish list
(data/raw/parishes_archgh.csv) into a GeoJSON point file used downstream
by 02_tract_assignment.py.

CSV schema: id, name, city, latitude, longitude, source
Output schema: parish_id, name, city, geometry (Point, EPSG:4326)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

print("=== Building parishes_archgh.geojson from authoritative CSV ===")

df = pd.read_csv('data/raw/parishes_archgh.csv')
print(f"Rows in CSV: {len(df)}")

missing_coords = df[df['latitude'].isna() | df['longitude'].isna()]
if len(missing_coords) > 0:
    print(f"WARNING: {len(missing_coords)} rows missing coordinates — skipping:")
    print(missing_coords[['id', 'name']].to_string(index=False))
    df = df.dropna(subset=['latitude', 'longitude'])

df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)

parishes = gpd.GeoDataFrame(
    df[['id', 'name', 'city', 'geometry']].rename(columns={'id': 'parish_id'}),
    geometry='geometry',
    crs='EPSG:4326',
)

parishes.to_file('data/raw/parishes_archgh.geojson', driver='GeoJSON')
print(f"Saved: data/raw/parishes_archgh.geojson  ({len(parishes)} parishes)")
print("\nSample names:")
for n in parishes['name'].head(10).tolist():
    print(f"  {n}")
