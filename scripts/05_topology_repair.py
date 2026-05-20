"""
Step 2.9: Topology repair for final parish territory output.

Strategy: additive-only repairs that cannot shrink or relocate existing
territory geometry, plus a single conservative pass for tiny precision overlaps.

Passes applied (in order):
  1. make_valid()           — self-intersections / degenerate rings
  2. Fill holes             — removes interior rings from every polygon
  3. Isolated-part fix      — disconnected fragments reassigned to the adjacent
                              parish with the longest shared boundary;
                              'main' part selected by proximity to parish point
  4. Precision overlap fix  — resolves overlapping pairs ONLY when the
                              intersection < 0.1 % of the smaller territory,
                              preventing runaway subtraction on large territories
  5. Gap / sliver fill      — morphological close (50 m) detects slivers and
                              gaps; only genuinely adjacent gaps are filled
  6. Fill holes + make_valid (final clean-up)

NOTE: vertex snapping is intentionally omitted.  The 100 m snap tolerance
distorts large rural territories (outer-county tracts assigned to a downtown
church span hundreds of km²) by moving their vertices across the shared edge
to a different part of the neighbour polygon, causing catastrophic area loss.
Gap fill is sufficient for closing simplification slivers.

Run after 04_final_output.py.  Overwrites the same file in-place.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

try:
    from shapely import make_valid          # Shapely 2.x
except ImportError:
    from shapely.validation import make_valid  # Shapely 1.x

from constants import CRS_PROJECTED, CRS_GEOGRAPHIC

INPUT         = 'data/output/parish_territories_final.geojson'
OUTPUT        = 'data/output/parish_territories_final.geojson'
PARISHES_PATH = 'data/raw/parishes_archgh.geojson'

# Overlap resolution: only resolve if the intersection is smaller than this
# fraction of the smaller territory.  Keeps precision-level overlaps (slivers
# created by per-polygon simplify) while skipping any large disputed areas
# that would shrink a territory dramatically.
MAX_OVERLAP_FRACTION = 0.001   # 0.1 % of the smaller territory

# Gap fill: morphological close detects gaps up to 2 × CLOSE_TOL wide.
CLOSE_TOL  = 50      # metres — catches gaps from 50 m simplify tolerance
MAX_GAP_M2 = 100_000 # m²     — gaps larger than this are real uncovered areas
TOUCH_TOL  = 5.0     # metres — gap must be this close to a territory to fill

# Isolated-part reassignment: only move parts SMALLER than this threshold.
# Large parts (e.g. 300 km² rural tracts assigned to a downtown parish) are
# legitimate territories even when disconnected — they must not be moved.
# Small parts (< 20 km²) are likely checkerboard artifacts from the nearest-
# neighbour assignment or floating-point dissolve issues.
MAX_ISOLATED_AREA_M2 = 20e6  # 20 km²


def extract_polygons(geom):
    """
    Return only the polygon content of any geometry.

    Polygon          — remove interior rings (fill holes)
    MultiPolygon     — remove interior rings from each part; drop zero-area parts
    GeometryCollection — extract Polygon/MultiPolygon sub-geometries, discard
                         LineStrings and Points that arise as degenerate shared
                         edges from union() calls; QGIS cannot render GCols as
                         polygon features
    Other / empty    — returned unchanged
    """
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior) if geom.area > 0 else geom

    elif geom.geom_type == 'MultiPolygon':
        parts = [Polygon(p.exterior) for p in geom.geoms if p.area > 0]
        if not parts:
            return geom
        return MultiPolygon(parts) if len(parts) > 1 else parts[0]

    elif geom.geom_type == 'GeometryCollection':
        polys = []
        for part in geom.geoms:
            if part.geom_type == 'Polygon' and part.area > 0:
                polys.append(Polygon(part.exterior))
            elif part.geom_type == 'MultiPolygon':
                polys.extend([Polygon(p.exterior) for p in part.geoms if p.area > 0])
        if not polys:
            return geom
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]

    return geom


print("=== Step 2.9: Topology repair ===\n")

# ─── Load ─────────────────────────────────────────────────────────────────────
gdf = gpd.read_file(INPUT).to_crs(CRS_PROJECTED)
n_start = len(gdf)
print(f"Loaded {n_start} parish territories.")
print(f"  MultiPolygon : {(gdf.geometry.geom_type == 'MultiPolygon').sum()}")
print(f"  Invalid      : {(~gdf.geometry.is_valid).sum()}")

data_cols = [c for c in gdf.columns if c != 'geometry']

parishes_pts = gpd.read_file(PARISHES_PATH).to_crs(CRS_PROJECTED)
if 'name' in parishes_pts.columns and 'parish_name' not in parishes_pts.columns:
    parishes_pts = parishes_pts.rename(columns={'name': 'parish_name'})
parish_pt_map = dict(zip(parishes_pts['parish_name'], parishes_pts.geometry))

# ─── Pass 1: make_valid ───────────────────────────────────────────────────────
gdf['geometry'] = gdf['geometry'].apply(make_valid)
print("\nPass 1 (make_valid) done.")

# ─── Pass 2: fill interior holes ─────────────────────────────────────────────
print("\nPass 2 — fill interior holes ...")

def _count_holes(geom):
    if geom.geom_type == 'Polygon':
        return len(list(geom.interiors))
    elif geom.geom_type == 'MultiPolygon':
        return sum(len(list(p.interiors)) for p in geom.geoms)
    return 0

total_holes = int(sum(gdf.geometry.apply(_count_holes)))
gdf['geometry'] = gdf['geometry'].apply(extract_polygons).apply(make_valid)
print(f"  Interior rings removed: {total_holes}")

# ─── Pass 3: isolated-part reassignment ──────────────────────────────────────
print("\nPass 3 — isolated-part reassignment ...")

parts = gdf.copy().explode(index_parts=False).reset_index(drop=True)
parts['_area'] = parts.geometry.area

def _main_score(row):
    pt = parish_pt_map.get(row['parish_name'])
    if pt is not None:
        return row.geometry.distance(pt)
    return -row['_area']

parts['_score']   = parts.apply(_main_score, axis=1)
main_idx          = parts.groupby('parish_name')['_score'].idxmin()
parts['_is_main'] = parts.index.isin(main_idx)

isolated = parts[~parts['_is_main']]
print(f"  Isolated parts: {len(isolated)}")

if len(isolated) > 0:
    sindex     = parts.sindex
    reassigned = 0

    skipped_large = 0
    for row_idx in isolated.index:
        iso_geom   = parts.loc[row_idx, 'geometry']
        old_parish = parts.loc[row_idx, 'parish_name']

        # Keep large parts — they are legitimate rural territories, not artifacts
        if parts.loc[row_idx, '_area'] > MAX_ISOLATED_AREA_M2:
            skipped_large += 1
            continue

        cands = gpd.GeoDataFrame()
        for buf in (500, 2_000, 10_000):
            cand_pos = list(sindex.query(iso_geom.buffer(buf)))
            cands    = parts.iloc[cand_pos]
            cands    = cands[cands['parish_name'] != old_parish]
            if len(cands):
                break

        if len(cands) == 0:
            print(f"  WARNING: no neighbour found for isolated part of '{old_parish}'.")
            continue

        def _shared(row):
            try:
                return iso_geom.buffer(500).intersection(row.geometry).length
            except Exception:
                return 0.0

        shared = cands.apply(_shared, axis=1)
        best   = cands.loc[shared.idxmax(), 'parish_name']
        parts.loc[row_idx, 'parish_name'] = best
        reassigned += 1

    print(f"  Reassigned: {reassigned} small parts (<20 km²).")
    print(f"  Kept large: {skipped_large} parts (>=20 km2 -- legitimate rural territories).")

parts_sorted = parts.sort_values('_is_main', ascending=False)
agg          = {c: 'first' for c in data_cols if c != 'parish_name'}
gdf          = parts_sorted.dissolve(by='parish_name', aggfunc=agg).reset_index()
gdf['geometry'] = gdf['geometry'].apply(make_valid).apply(extract_polygons)
print(f"  MultiPolygons after: {(gdf.geometry.geom_type == 'MultiPolygon').sum()}")

# ─── Pass 4: precision overlap resolution ────────────────────────────────────
# Uses the ORIGINAL geometry snapshot for intersection detection so that
# resolving pair (i, j) does not cascade into inflated intersections for
# subsequent pairs involving i or j.
# Only resolves overlaps < MAX_OVERLAP_FRACTION of the smaller territory to
# prevent large territory losses from mis-detected "overlaps".
print(f"\nPass 4 — precision overlap resolution (max {MAX_OVERLAP_FRACTION*100:.1f}% of smaller territory) ...")

orig_geoms = list(gdf.geometry)   # fixed snapshot — intersection detection reads from here
work_geoms = list(gdf.geometry)   # modified during resolution
sindex     = gdf.sindex           # built on orig_geoms positions

n_found = n_resolved = n_skipped_large = 0

for i in range(len(gdf)):
    g_orig_i = orig_geoms[i]
    name_i   = gdf.iloc[i]['parish_name']
    pt_i     = parish_pt_map.get(name_i)

    cand_pos = list(sindex.query(g_orig_i))
    for j in cand_pos:
        if j <= i:
            continue
        g_orig_j = orig_geoms[j]

        try:
            inter = g_orig_i.intersection(g_orig_j)
        except Exception:
            continue

        if inter.is_empty or inter.area < 1.0:   # < 1 m² — floating-point noise
            continue

        n_found += 1
        threshold = MAX_OVERLAP_FRACTION * min(g_orig_i.area, g_orig_j.area)

        if inter.area > threshold:
            n_skipped_large += 1
            continue   # too large — do not risk distorting the territory

        name_j = gdf.iloc[j]['parish_name']
        pt_j   = parish_pt_map.get(name_j)
        centroid = inter.centroid

        if pt_i is not None and pt_j is not None:
            give_to_i = pt_i.distance(centroid) <= pt_j.distance(centroid)
        else:
            give_to_i = g_orig_i.area >= g_orig_j.area

        try:
            if give_to_i:
                work_geoms[j] = make_valid(work_geoms[j].difference(inter))
            else:
                work_geoms[i] = make_valid(work_geoms[i].difference(inter))
            n_resolved += 1
        except Exception as e:
            print(f"  WARNING: could not resolve '{name_i}'/'{name_j}': {e}")

gdf['geometry'] = [extract_polygons(g) for g in work_geoms]
print(f"  Overlapping pairs found : {n_found}")
print(f"  Resolved (small)        : {n_resolved}")
print(f"  Skipped (too large)     : {n_skipped_large}")

# ─── Pass 5: gap / sliver fill ────────────────────────────────────────────────
# Purely additive — only adds area to the nearest territory, never subtracts.
# Gaps from per-polygon simplify (up to 2 × CLOSE_TOL ≈ 100 m wide) are caught.
# The adjacency check (TOUCH_TOL) prevents filling legitimate non-coverage areas
# (rural land, water bodies) that are not touching any territory boundary.
print(f"\nPass 5 — gap / sliver fill (close={CLOSE_TOL} m, max {MAX_GAP_M2/1e6:.3f} km²) ...")

all_union = unary_union(gdf.geometry)
closed    = all_union.buffer(CLOSE_TOL).buffer(-CLOSE_TOL)
gaps      = closed.difference(all_union)

if gaps.is_empty:
    print("  No gaps detected.")
else:
    if gaps.geom_type == 'Polygon':
        gap_list = [gaps]
    elif gaps.geom_type == 'MultiPolygon':
        gap_list = list(gaps.geoms)
    else:
        gap_list = [g for g in getattr(gaps, 'geoms', []) if g.geom_type == 'Polygon']

    small_gaps = [g for g in gap_list if 0 < g.area < MAX_GAP_M2]
    print(f"  Gap polygons : {len(gap_list)} total, {len(small_gaps)} within size threshold.")

    sindex  = gdf.sindex
    filled  = 0
    skipped = 0

    for gap_geom in small_gaps:
        cand_pos = list(sindex.query(gap_geom.buffer(CLOSE_TOL)))
        if not cand_pos:
            continue
        cands    = gdf.iloc[cand_pos]
        touching = cands.geometry.intersects(gap_geom.buffer(TOUCH_TOL))
        if not touching.any():
            skipped += 1
            continue
        touching_cands = cands[touching]
        nearest_i = touching_cands.geometry.distance(gap_geom).idxmin()
        merged    = gdf.loc[nearest_i, 'geometry'].union(gap_geom.buffer(TOUCH_TOL))
        # union() can produce GeometryCollection (polygon + degenerate edge lines);
        # extract_polygons strips non-polygon parts before storing
        gdf.loc[nearest_i, 'geometry'] = extract_polygons(make_valid(merged))
        filled += 1

    print(f"  Filled: {filled} | Skipped (non-adjacent): {skipped}")

# ─── Pass 6: final hole fill + make_valid ────────────────────────────────────
gdf['geometry'] = gdf['geometry'].apply(extract_polygons).apply(make_valid)
print("\nPass 6 (fill holes + make_valid) done.")

# ─── Area sanity check ────────────────────────────────────────────────────────
pre  = gpd.read_file('data/processed/parish_territories.geojson').to_crs(CRS_PROJECTED)
pre['area_orig'] = pre.geometry.area
gdf['area_new']  = gdf.geometry.area
check = gdf[['parish_name','area_new']].merge(pre[['parish_name','area_orig']], on='parish_name')
check['pct_chg'] = (check['area_new'] - check['area_orig']) / check['area_orig'] * 100
bad = check[check['pct_chg'].abs() > 5].sort_values('pct_chg')
if len(bad):
    print(f"\nWARNING: {len(bad)} territories changed by >5% vs original dissolved data:")
    print(bad[['parish_name','area_orig','area_new','pct_chg']].to_string(index=False))
else:
    print("\nArea sanity check passed: all territories within 5% of original size.")

gdf = gdf.drop(columns=['area_new'], errors='ignore')

# ─── Summary ──────────────────────────────────────────────────────────────────
invalid_count = (~gdf.geometry.is_valid).sum()
multi_count   = (gdf.geometry.geom_type == 'MultiPolygon').sum()
print(f"\nFinal state:")
print(f"  Parishes      : {len(gdf)}  (was {n_start})")
print(f"  Invalid geoms : {invalid_count}")
print(f"  MultiPolygons : {multi_count}")
if multi_count > 0:
    print("  NOTE: remaining MultiPolygons may be legitimate coastal/island territories.")

# ─── Export ───────────────────────────────────────────────────────────────────
gdf = gdf.to_crs(CRS_GEOGRAPHIC)
gdf.to_file(OUTPUT, driver='GeoJSON')
print(f"\nSaved: {OUTPUT}")
print("Step 2.9 complete.")
