"""Interactive Mapping with Folium

This script creates interactive maps using Folium to visualize the distribution of hospitals in Peru.
It includes a choropleth map showing hospital density by district and a proximity analysis for Lima and Loreto
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

# --- Rutas (ajustadas a tu repo) ---
ROOT = Path(__file__).resolve().parents[2]
HOSP_CSV = ROOT / "_data" / "operational_hospitals.csv"
DIST_DIR = ROOT / "_data" / "DISTRITOS"
OUT_DIR  = ROOT / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Utilidades ---
def first_shp(folder: Path):
    shps = [p for p in folder.rglob("*") if p.suffix.lower() == ".shp"]
    return shps[0] if shps else None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

def looks_like_latlon(lat: pd.Series, lon: pd.Series) -> bool:
    # ¿parecen grados?
    ok_lat = lat.between(-90, 90).mean() > 0.95
    ok_lon = lon.between(-180, 180).mean() > 0.95
    return bool(ok_lat and ok_lon)

def to_wgs84_from_best_utm(df_xy: pd.DataFrame, xcol: str, ycol: str) -> gpd.GeoDataFrame:
    """
    Intenta 32717, 32718, 32719 y escoge la que produzca más puntos dentro de rangos de Perú.
    Devuelve GeoDataFrame en EPSG:4326.
    """
    candidates = [32717, 32718, 32719]
    best = None
    best_valid = -1
    for epsg in candidates:
        g = gpd.GeoDataFrame(
            df_xy,
            geometry=gpd.points_from_xy(df_xy[xcol], df_xy[ycol]),
            crs=f"EPSG:{epsg}"
        ).to_crs("EPSG:4326")
        lat = g.geometry.y
        lon = g.geometry.x
        valid = (lat.between(-19, 1) & lon.between(-85, -67)).sum()
        if valid > best_valid:
            best_valid = valid
            best = g
    if best is None:
        raise ValueError("No pude convertir UTM a WGS84.")
    # filtra a Perú
    lat = best.geometry.y
    lon = best.geometry.x
    best = best[lat.between(-19, 1) & lon.between(-85, -67)].copy()
    return best

# --- 1) Cargar hospitales y normalizar coordenadas a EPSG:4326 ---
h = pd.read_csv(HOSP_CSV, encoding="utf-8-sig")
# columnas NORTE/ESTE (de tu CSV)
if not {"NORTE", "ESTE"}.issubset(h.columns):
    raise ValueError(f"Tu CSV debe tener columnas 'NORTE' y 'ESTE'. Columnas: {list(h.columns)}")

h["NORTE"] = to_num(h["NORTE"])
h["ESTE"]  = to_num(h["ESTE"])
h = h.dropna(subset=["NORTE", "ESTE"]).copy()

# ¿son grados o UTM?
if looks_like_latlon(h["NORTE"], h["ESTE"]):
    # ¡sorpresa! NORTE/ESTE vienen ya como lat/lon (poco probable con esos nombres)
    g_h = gpd.GeoDataFrame(h, geometry=gpd.points_from_xy(h["ESTE"], h["NORTE"]), crs="EPSG:4326")
    print("[INFO] NORTE/ESTE parecen lat/lon (grados).")
else:
    # Tratar como UTM y auto-detectar zona mejor (17S/18S/19S)
    print("[INFO] NORTE/ESTE parecen UTM. Probando zonas 17S/18S/19S…")
    g_h = to_wgs84_from_best_utm(h, "ESTE", "NORTE")  # x=ESTE, y=NORTE (UTM → WGS84)
print(f"[DEBUG] hospitales válidos (WGS84): {len(g_h)}")

# nombre para tooltip
name_hcol = None
for c in ["Nombre del establecimiento", "NOMBRE DEL ESTABLECIMIENTO", "NOMBRE", "ESTABLECIMIENTO"]:
    if c in g_h.columns:
        name_hcol = c
        break

# --- 2) Cargar distritos (EPSG:4326) + reparar geometrías ---
dist_shp = first_shp(DIST_DIR)
if not dist_shp:
    raise FileNotFoundError("No encontré shapefile en _data/DISTRITOS/")
g_dist = gpd.read_file(dist_shp)
g_dist = g_dist.to_crs("EPSG:4326") if g_dist.crs else g_dist.set_crs("EPSG:4326")

# Reparar geometrías (polígonos inválidos) y filtrar nulos
g_dist = g_dist[g_dist.geometry.notnull()].copy()
g_dist["geometry"] = g_dist.buffer(0)

if "UBIGEO" not in g_dist.columns:
    g_dist["UBIGEO"] = g_dist.index.astype(str)
print(f"[DEBUG] distritos: {len(g_dist)} polígonos | CRS={g_dist.crs}")
print(f"[DEBUG] bounds distritos: {g_dist.total_bounds}")  # [minx, miny, maxx, maxy]
print(f"[DEBUG] bounds hospitales: {g_h.total_bounds}")

# --- 3) Conteo por distrito: within → intersects → nearest (fallback) ---
def _count_by(method: str):
    if method == "within":
        j = gpd.sjoin(g_h, g_dist, how="left", predicate="within")
    elif method == "intersects":
        j = gpd.sjoin(g_h, g_dist, how="left", predicate="intersects")
    elif method == "nearest":
        # distancia máxima ~ 0.01° ≈ 1.1 km (ajustable)
        j = gpd.sjoin_nearest(g_h, g_dist, how="left", distance_col="dist_deg", max_distance=0.02)
    else:
        raise ValueError("method inválido")
    c = j.groupby(j.index_right).size()
    return c

# --- Sanity check: ¿están los ejes invertidos? (x debería ser lon ∈ [-85,-67], y lat ∈ [-19,1]) ---
bx_min, by_min, bx_max, by_max = g_h.total_bounds
if (-19 <= bx_min <= 1) and (-85 <= by_min <= -67):  # x luce como lat y y como lon
    # Intercambiar: nuevo x = lon (antes y), nuevo y = lat (antes x)
    g_h = g_h.set_geometry(
        gpd.points_from_xy(g_h.geometry.y, g_h.geometry.x),
        crs="EPSG:4326"
    )
    print("[FIX] Coordenadas invertidas detectadas. Se intercambiaron x↔y.")
    print("[DEBUG] bounds hospitales (corregidos):", g_h.total_bounds)

# 3.1 within
counts = _count_by("within")
total = int(counts.sum())
print(f"[DEBUG] suma conteos (within): {total}")

# 3.2 fallback intersects si está en cero
if total == 0:
    counts = _count_by("intersects")
    total = int(counts.sum())
    print(f"[DEBUG] suma conteos (intersects): {total}")

# 3.3 fallback nearest si sigue en cero
if total == 0:
    try:
        counts = _count_by("nearest")
        total = int(counts.sum())
        print(f"[DEBUG] suma conteos (nearest<=0.02°): {total}")
    except Exception as e:
        print("[WARN] sjoin_nearest no disponible/falló:", e)

# Aplicar conteos al GeoDataFrame de distritos
g_dist["hosp_count"] = g_dist.index.map(counts).fillna(0).astype(int)
print(f"[DEBUG] distritos con >=1 hospital: {(g_dist['hosp_count']>0).sum()}")

# --- 4) Construir mapa Folium ---
m = folium.Map(location=[-9.19, -75.02], zoom_start=5, tiles="cartodbpositron")

folium.Choropleth(
    geo_data=g_dist.to_json(),
    data=g_dist,
    columns=["UBIGEO", "hosp_count"],
    key_on="feature.properties.UBIGEO",
    fill_color="YlOrRd",
    fill_opacity=0.85,
    line_opacity=0.2,
    legend_name="Hospitales operativos por distrito"
).add_to(m)

name_dcol = next((c for c in ["DISTRITO","NOMB_DIST","NOMBRE"] if c in g_dist.columns), "UBIGEO")
folium.GeoJson(
    g_dist,
    name="Distritos",
    style_function=lambda x: {"fillOpacity": 0, "color": "#00000000"},
    tooltip=folium.GeoJsonTooltip(
        fields=[name_dcol, "hosp_count"],
        aliases=["Distrito", "Hospitales"],
        sticky=True
    )
).add_to(m)

# MarkerCluster de hospitales
mc = MarkerCluster(name="Hospitales").add_to(m)
for _, r in g_h.iterrows():
    folium.Marker(
        location=[r.geometry.y, r.geometry.x],
        tooltip=str(r[name_hcol]) if name_hcol else "Hospital"
    ).add_to(mc)

folium.LayerControl(collapsed=False).add_to(m)

out = OUT_DIR / "national_choropleth_hospitals.html"
m.save(str(out))
print(f"✅ Mapa guardado en {out}")

