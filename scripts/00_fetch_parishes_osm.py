"""
Fetches Catholic church/parish points for the Archdiocese of Galveston-Houston
from the OpenStreetMap Overpass API. Saves to data/raw/parishes_osm.geojson.

The bounding box covers all 10 Archdiocese counties:
  South ~28.9N, North ~31.1N, West ~97.0W, East ~94.3W
"""
import requests
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CRS_PROJECTED

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

# Bounding box: (south, west, north, east)
BBOX = "28.9,-97.0,31.1,-94.3"

QUERY = f"""
[out:json][timeout:90];
(
  node["amenity"="place_of_worship"]["religion"="christian"]["denomination"="catholic"]({BBOX});
  way["amenity"="place_of_worship"]["religion"="christian"]["denomination"="catholic"]({BBOX});
  relation["amenity"="place_of_worship"]["religion"="christian"]["denomination"="catholic"]({BBOX});
);
out center tags;
"""

print("Querying Overpass API for Catholic churches in the Archdiocese area...")
headers = {
    "Accept": "application/json, */*",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "parish-analysis/1.0 (research project)",
}
response = requests.post(
    OVERPASS_URL,
    data=f"data={requests.utils.quote(QUERY)}",
    headers=headers,
    timeout=120,
)
response.raise_for_status()
result = response.json()

elements = result.get("elements", [])
print(f"Raw OSM elements returned: {len(elements)}")

records = []
for elem in elements:
    tags = elem.get("tags", {})
    name = tags.get("name", "")
    if not name:
        continue  # skip unnamed features

    # Get coordinates — nodes have lat/lon directly; ways/relations use center
    if elem["type"] == "node":
        lat, lon = elem["lat"], elem["lon"]
    else:
        center = elem.get("center", {})
        if not center:
            continue
        lat, lon = center["lat"], center["lon"]

    records.append({
        "osm_id": elem["id"],
        "osm_type": elem["type"],
        "name": name,
        "geometry": Point(lon, lat),
    })

df = pd.DataFrame(records)
parishes = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

print(f"Named Catholic churches found: {len(parishes)}")
print("\nSample names:")
for n in parishes["name"].head(20).tolist():
    print(f"  {n}")

parishes.to_file("data/raw/parishes_osm.geojson", driver="GeoJSON")
print(f"\nSaved: data/raw/parishes_osm.geojson  ({len(parishes)} features)")
