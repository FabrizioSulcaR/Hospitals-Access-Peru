""" Proximity AnalysiS Lima Loreto

This script performs a proximity analysis of hospitals to population centers in the departments of Lima and Loreto, Peru.
it generates interactive maps showing the most and least served population centers within a specified radius."""

import sys, os, time
print("[START FILE]", __file__, flush=True)
print("[CWD]", os.getcwd(), flush=True)
print("[PYTHON]", sys.executable, flush=True)

# _data/scripts/2.2.proximity_lim_loreto.py
import sys, time
from pathlib import Path
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from math import radians, sin, cos, asin, sqrt

# ------------------ RUTAS ------------------
ROOT = Path(__file__).resolve().parents[2]
HOSP_CSV = ROOT / "_data" / "operational_hospitals.csv"  # CSV con NORTE/ESTE (limpio y operativo)
CP_DIR   = ROOT / "_data" / "CCPP_0"                      # Shapefile Centros Poblados
DIST_DIR = ROOT / "_data" / "DISTRITOS"                   # Shapefile Distritos
OUT_DIR  = ROOT / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RADIUS_KM = 10.0
TILES = "cartodbpositron"
MAX_TIES = 3  # máximo de empates por tipo (mín/máx) para no saturar el mapa

# ------------------ HELPERS ------------------
def first_shp(folder: Path):
    shps = [p for p in folder.rglob("*") if p.suffix.lower() == ".shp"]
    return shps[0] if shps else None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

def looks_like_latlon(lat: pd.Series, lon: pd.Series) -> bool:
    ok_lat = lat.between(-90, 90).mean() > 0.95
    ok_lon = lon.between(-180, 180).mean() > 0.95
    return bool(ok_lat and ok_lon)

def to_wgs84_from_best_utm(df_xy: pd.DataFrame, xcol: str, ycol: str) -> gpd.GeoDataFrame:
    # Prueba UTM 17S/18S/19S, elige la que produce más puntos en Perú y convierte a WGS84
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
    lat = best.geometry.y
    lon = best.geometry.x
    best = best[lat.between(-19, 1) & lon.between(-85, -67)].copy()
    return best

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2*R*asin(sqrt(a))

def count_within_radius(center_lat, center_lon, hosp_df, radius_km=10.0) -> int:
    cnt = 0
    for la, lo in zip(hosp_df["lat"].values, hosp_df["lon"].values):
        if haversine_km(center_lat, center_lon, la, lo) <= radius_km:
            cnt += 1
    return cnt

# UBIGEO → Departamento (fallback)
DEPT_CODE_MAP = {
    "01":"AMAZONAS","02":"ANCASH","03":"APURIMAC","04":"AREQUIPA","05":"AYACUCHO",
    "06":"CAJAMARCA","07":"CALLAO","08":"CUSCO","09":"HUANCAVELICA","10":"HUANUCO",
    "11":"ICA","12":"JUNIN","13":"LA LIBERTAD","14":"LAMBAYEQUE","15":"LIMA",
    "16":"LORETO","17":"MADRE DE DIOS","18":"MOQUEGUA","19":"PASCO","20":"PIURA",
    "21":"PUNO","22":"SAN MARTIN","23":"TACNA","24":"TUMBES","25":"UCAYALI"
}

def detect_dept_column(gdf: gpd.GeoDataFrame):
    for c in ["DEPARTAMENTO","DEPARTAMEN","DEPARTAM","DPTO","REGION","REGIÓN","NOM_DEP","NOMBDEP","DEPA_NOMB","DEPART"]:
        if c in gdf.columns:
            return c
    if "UBIGEO" in gdf.columns:
        codes = gdf["UBIGEO"].astype(str).str[:2]
        gdf["__DEPT__"] = codes.map(DEPT_CODE_MAP).fillna("DESCONOCIDO")
        return "__DEPT__"
    return None

# ------------------ CARGAS ------------------
def load_districts_wgs84() -> gpd.GeoDataFrame:
    shp = first_shp(DIST_DIR)
    if not shp:
        raise FileNotFoundError("No encontré shapefile de Distritos.")
    d = gpd.read_file(shp)
    d = d.to_crs("EPSG:4326") if d.crs else d.set_crs("EPSG:4326")
    d = d[d.geometry.notnull()].copy()
    d["geometry"] = d.buffer(0)  # reparar geometrías
    dept_col = detect_dept_column(d)
    if not dept_col:
        raise ValueError("No pude identificar/derivar la columna de departamento en Distritos.")
    d["__DEPT__"] = d[dept_col].astype(str).str.upper().str.strip()
    return d

def load_hospitals_wgs84() -> pd.DataFrame:
    if not HOSP_CSV.exists():
        raise FileNotFoundError(f"No encuentro {HOSP_CSV}")

    df = pd.read_csv(HOSP_CSV, encoding="utf-8-sig")

    # Validación de columnas de coordenadas
    if not {"NORTE", "ESTE"}.issubset(df.columns):
        raise ValueError(f"Se requieren columnas 'NORTE' y 'ESTE'. Columnas: {list(df.columns)}")

    # Normaliza números
    df["NORTE"] = to_num(df["NORTE"])
    df["ESTE"]  = to_num(df["ESTE"])
    df = df.dropna(subset=["NORTE", "ESTE"]).copy()

    # A WGS84 (lat/lon)
    if looks_like_latlon(df["NORTE"], df["ESTE"]):
        g = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["ESTE"], df["NORTE"]),
            crs="EPSG:4326"
        )
    else:
        g = to_wgs84_from_best_utm(df, "ESTE", "NORTE")

    # Si los ejes vinieron invertidos (seguridad)
    bx_min, by_min, _, _ = g.total_bounds
    if (-19 <= bx_min <= 1) and (-85 <= by_min <= -67):
        g = g.set_geometry(gpd.points_from_xy(g.geometry.y, g.geometry.x), crs="EPSG:4326")

    g["lat"] = g.geometry.y
    g["lon"] = g.geometry.x

    # --- nombre hospital robusto ---
    candidates = [
        "Nombre del establecimiento", "NOMBRE DEL ESTABLECIMIENTO",
        "NOMBRE_ESTABLECIMIENTO", "NOMBRE", "ESTABLECIMIENTO", "RAZON_SOCIAL"
    ]
    inst_candidates = ["INSTITUCIÓN", "INSTITUCION", "INSTITUTION"]
    dep_candidates  = ["DEPARTAMENTO", "DEPARTAMEN", "DEPARTAM", "DPTO"]
    dist_candidates = ["DISTRITO", "NOMB_DIST", "DISTRITO_NOM"]

    def first_col(cols):
        for c in cols:
            if c in g.columns:
                return c
        return None

    name_col = first_col(candidates)
    inst_col = first_col(inst_candidates)
    dep_col  = first_col(dep_candidates)
    dist_col = first_col(dist_candidates)

    if name_col:
        base = (
            g[name_col]
            .astype(str)
            .fillna("")
            .str.strip()
            .replace({"nan": "", "None": ""})
        )
    else:
        base = pd.Series([""] * len(g), index=g.index)

    # Construye fallback combinando columnas secundarias (fila a fila)
    cols_to_concat = []
    if inst_col: cols_to_concat.append(g[inst_col].astype(str).fillna("").str.strip())
    if dep_col:  cols_to_concat.append(g[dep_col].astype(str).fillna("").str.strip())
    if dist_col: cols_to_concat.append(g[dist_col].astype(str).fillna("").str.strip())

    if cols_to_concat:
        joined = pd.concat(cols_to_concat, axis=1).fillna("").astype(str)
        # Junta solo valores no vacíos por fila
        fallback = joined.apply(lambda r: " - ".join([x for x in r.values if x]), axis=1)
    else:
        fallback = pd.Series([""], index=g.index)

    # Completa base con fallback cuando base está vacía
    label = base.where(base.str.len() > 0, fallback)
    label = label.replace({"": "Hospital sin nombre"}).fillna("Hospital sin nombre")

    g["__name_h__"] = label.astype(str)

    return g[["lat", "lon", "__name_h__"]].copy()


def load_cpop_wgs84(dist_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    shp = first_shp(CP_DIR)
    if not shp:
        raise FileNotFoundError(f"No encontré shapefile en {CP_DIR}")
    g = gpd.read_file(shp)
    g = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
    g = g[g.geometry.notnull()].copy()

    # nombre CP
    name_cp = None
    for c in ["NOMBRE","NOMBRE_CP","NOMBRE_CCPP","CCPP","CENTRO_POBLADO","NOM_CCPP","NOMCP","NOM_CC_POB"]:
        if c in g.columns:
            name_cp = c; break
    if not name_cp:
        name_cp = g.columns[0]
    g["__name__"] = g[name_cp].astype(str)

    # asignar dept por sjoin con distritos
    sj = gpd.sjoin(g, dist_gdf[["__DEPT__","geometry"]], how="left", predicate="within")
    g["__dept__"] = sj["__DEPT__"].astype(str).str.upper().str.strip().fillna("DESCONOCIDO")

    g["lat"] = g.geometry.y; g["lon"] = g.geometry.x
    return g[["__dept__","__name__","lat","lon","geometry"]].copy()

# ------------------ PROXIMIDAD POR DEPARTAMENTO ------------------
def proximity_map(
    dept_name: str,
    hosp_all_df: pd.DataFrame,          # TODOS los hospitales (lat/lon/__name_h__)
    cpop_gdf: gpd.GeoDataFrame,         # CCPP con __dept__/__name__/lat/lon
    dist_gdf: gpd.GeoDataFrame,         # Distritos con __DEPT__
    radius_km=10.0
):
    dept = dept_name.upper().strip()

    # 1) Polígonos del departamento (para encuadre del mapa)
    dept_poly = dist_gdf[dist_gdf["__DEPT__"] == dept]
    if dept_poly.empty:
        raise ValueError(f"No encontré polígonos de departamento {dept}")

    # 2) CCPP del departamento
    csub = cpop_gdf[cpop_gdf["__dept__"] == dept].copy()
    if csub.empty:
        raise ValueError(f"No hay Centros Poblados para {dept}")

    # 3) Conteo de hospitales ≤ radio (usando TODOS los hospitales)
    print(f"[INFO] {dept}: CCPP a evaluar = {len(csub)} | hospitales totales = {len(hosp_all_df)}", flush=True)
    results = []
    for i, (_, row) in enumerate(csub.iterrows(), 1):
        if i % 200 == 0 or i == 1:
            print(f"[..] {dept}: procesando CP {i}/{len(csub)}", flush=True)
        n = count_within_radius(row["lat"], row["lon"], hosp_all_df, radius_km)
        results.append((row["__name__"], row["lat"], row["lon"], n))

    results.sort(key=lambda x: x[3])
    min_count = results[0][3]
    max_count = results[-1][3]
    mins = [r for r in results if r[3] == min_count][:MAX_TIES]
    maxs = [r for r in results if r[3] == max_count][:MAX_TIES]

    # 4) Mapa ajustado al departamento
    m = folium.Map(tiles=TILES)
    bounds = dept_poly.total_bounds  # [minx, miny, maxx, maxy]
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # 5) Círculos rojos (mínimos)
    for k, cp in enumerate(mins, 1):
        folium.Circle(
            location=[cp[1], cp[2]],
            radius=int(radius_km * 1000),
            color="red", fill=False, weight=2, opacity=0.9,
            tooltip=f"Centro poblado: {cp[0]} — {cp[3]} hosp ≤ {radius_km} km (mín #{k})",
            popup=folium.Popup(
                f"<b>Centro poblado (mín #{k}):</b> {cp[0]}<br>"
                f"<b>Hospitales ≤ {radius_km} km:</b> {cp[3]}",
                max_width=340
            )
        ).add_to(m)

    # 6) Círculos verdes (máximos)
    for k, cp in enumerate(maxs, 1):
        folium.Circle(
            location=[cp[1], cp[2]],
            radius=int(radius_km * 1000),
            color="green", fill=False, weight=2, opacity=0.9,
            tooltip=f"Centro poblado: {cp[0]} — {cp[3]} hosp ≤ {radius_km} km (máx #{k})",
            popup=folium.Popup(
                f"<b>Centro poblado (máx #{k}):</b> {cp[0]}<br>"
                f"<b>Hospitales ≤ {radius_km} km:</b> {cp[3]}",
                max_width=340
            )
        ).add_to(m)

    # 7) Resaltar hospitales dentro de cada círculo (rojo/verde) – TODOS los hospitales
    for label, group in [("min", mins), ("max", maxs)]:
        col = "red" if label == "min" else "green"
        for cp in group:
            for _, r in hosp_all_df.iterrows():
                if haversine_km(cp[1], cp[2], r["lat"], r["lon"]) <= radius_km:
                    folium.CircleMarker(
                        location=[r["lat"], r["lon"]],
                        radius=4,
                        color=col,
                        fill=True,
                        fill_opacity=0.9,
                        tooltip=folium.Tooltip(r["__name_h__"], sticky=True, direction="top"),
                        popup=folium.Popup(f"<b>Hospital:</b> {r['__name_h__']}", max_width=260)
                    ).add_to(m)

    # (opcional) Capa con todos los hospitales del dpto en gris (contexto)
    try:
        hosp_g = gpd.GeoDataFrame(hosp_all_df, geometry=gpd.points_from_xy(hosp_all_df["lon"], hosp_all_df["lat"]), crs="EPSG:4326")
        dept_hosp = gpd.sjoin(hosp_g, dept_poly[["__DEPT__","geometry"]], how="inner", predicate="within")
        layer = folium.FeatureGroup(name=f"Hospitales en {dept} (contexto)", show=False)
        for _, r in dept_hosp.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=2,
                color="#888888",
                fill=True,
                fill_opacity=0.5,
                tooltip=r["__name_h__"]
            ).add_to(layer)
        layer.add_to(m)
    except Exception as e:
        print(f"[WARN] capa de contexto hospitales {dept} no añadida: {e}")

    folium.LayerControl(collapsed=False).add_to(m)

    # 8) CSV con extremos (mins + maxs)
    summary = pd.DataFrame(mins + maxs, columns=["cp_name", "lat", "lon", "hosp_10km"])
    summary.insert(0, "dept", dept)
    summary["extreme"] = (["min"] * len(mins)) + (["max"] * len(maxs))
    csv_path = OUT_DIR / f"proximity_{dept.lower()}_extremes_{int(radius_km)}km.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📄 CSV extremos {dept} → {csv_path}")

    # 9) Guardar HTML
    out = OUT_DIR / f"proximity_{dept.lower()}_{int(radius_km)}km.html"
    m.save(str(out))
    print(f"✅ Proximidad {dept} → {out}")

    return {"mins": mins, "maxs": maxs}

if __name__ == "__main__":
    print("[RUN] Proximity analysis (Lima & Loreto) — R=10 km", flush=True)

    # Cargas
    dist_gdf = load_districts_wgs84()
    hosp_df  = load_hospitals_wgs84()      # columnas: lat, lon, __name_h__
    cpop_gdf = load_cpop_wgs84(dist_gdf)   # columnas: __dept__, __name__, lat, lon, geometry

    # Parámetros
    radius_km = RADIUS_KM  # 10.0 por defecto
    depts = ["LIMA", "LORETO"]

    results = {}
    for d in depts:
        print(f"\n=== {d} ===")
        res = proximity_map(
            dept_name=d,
            hosp_all_df=hosp_df,
            cpop_gdf=cpop_gdf,
            dist_gdf=dist_gdf,
            radius_km=radius_km
        )
        results[d] = res
        # Log breve:
        mins = ", ".join([f"{n} ({c} hosp)" for n, _, _, c in res["mins"]])
        maxs = ", ".join([f"{n} ({c} hosp)" for n, _, _, c in res["maxs"]])
        print(f"[{d}] Min densidad: {mins}")
        print(f"[{d}] Max densidad: {maxs}")

    print("\n[OK] Mapas HTML y CSV generados en 'assets/'.")



#Breve análisis escrito: 
# Lima: Los resultados muestran que los centros poblados con mayor número de hospitales en un radio de 10 km se concentran 
# en la capital y su periferia inmediata. Esto confirma la alta concentración urbana y la facilidad de acceso a servicios de
#  salud en áreas metropolitanas densamente pobladas. La infraestructura hospitalaria está relativamente próxima entre sí, 
# lo que mejora la accesibilidad pero también refleja una distribución centralizada que favorece a la capital frente a 
# provincias más alejadas.
# 
# Loreto: En contraste, los centros poblados de Loreto evidencian baja densidad hospitalaria en un radio de 10 km. Esto se explica
#  por la dispersión geográfica de la población y las condiciones propias de la Amazonía: grandes distancias, presencia de ríos como
#  principales vías de transporte, y una infraestructura sanitaria más limitada. Los resultados refuerzan el reto de la accesibilidad 
# en esta región: incluso centros poblados de importancia tienen escasos hospitales cercanos, lo que genera mayores barreras para la 
# atención oportuna.."""