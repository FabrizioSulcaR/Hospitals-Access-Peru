"""Aplication Deplyment with Streamlit
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

# =========================
# RUTAS (desde _data/scripts/)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # raíz del repo
DATA_DIR     = PROJECT_ROOT / "_data"
HOSP_CSV     = DATA_DIR / "operational_hospitals.csv"
DIST_DIR     = DATA_DIR / "DISTRITOS"
ASSETS       = PROJECT_ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ==============
# CONFIG STREAMLIT
# ==============
st.set_page_config(page_title="Acceso a Hospitales — Perú", layout="wide", page_icon="🌍")
st.title("Acceso a Hospitales — Perú")

# =========================
# HELPERS CARGA Y LIMPIEZA
# =========================
@st.cache_data(show_spinner=False)
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

@st.cache_data(show_spinner=False)
def load_districts_wgs84() -> gpd.GeoDataFrame:
    shps = [p for p in DIST_DIR.rglob("*") if p.suffix.lower() == ".shp"]
    if not shps:
        raise FileNotFoundError(f"No se encontró shapefile en {DIST_DIR}")
    g = gpd.read_file(shps[0])
    g = g.to_crs(4326) if g.crs else g.set_crs(4326)
    g = g[g.geometry.notnull()].copy()
    # sanear geometrías
    g["geometry"] = g.buffer(0)

    candidates = ["DEPARTAMENTO","DEPARTAMEN","DEPARTAM","DPTO","REGION","REGIÓN","NOM_DEP","NOMBDEP","DEPA_NOMB","DEPART"]
    dept_col = next((c for c in candidates if c in g.columns), None)
    if dept_col:
        g["__DEPT__"] = g[dept_col].astype(str).str.upper().str.strip()
    elif "UBIGEO" in g.columns:
        DEPT_CODE_MAP = {
            "01":"AMAZONAS","02":"ANCASH","03":"APURIMAC","04":"AREQUIPA","05":"AYACUCHO",
            "06":"CAJAMARCA","07":"CALLAO","08":"CUSCO","09":"HUANCAVELICA","10":"HUANUCO",
            "11":"ICA","12":"JUNIN","13":"LA LIBERTAD","14":"LAMBAYEQUE","15":"LIMA",
            "16":"LORETO","17":"MADRE DE DIOS","18":"MOQUEGUA","19":"PASCO","20":"PIURA",
            "21":"PUNO","22":"SAN MARTIN","23":"TACNA","24":"TUMBES","25":"UCAYALI"
        }
        g["__DEPT__"] = g["UBIGEO"].astype(str).str[:2].map(DEPT_CODE_MAP).fillna("DESCONOCIDO")
    else:
        g["__DEPT__"] = "DESCONOCIDO"
    return g

@st.cache_data(show_spinner=False)
def load_hospitals_points() -> gpd.GeoDataFrame:
    if not HOSP_CSV.exists():
        raise FileNotFoundError(f"No se encontró {HOSP_CSV}")
    df = pd.read_csv(HOSP_CSV, encoding="utf-8-sig")
    if not {"NORTE","ESTE"}.issubset(df.columns):
        raise ValueError("Se requieren columnas NORTE y ESTE en el CSV.")

    df["NORTE"] = to_num(df["NORTE"]); df["ESTE"] = to_num(df["ESTE"])
    df = df.dropna(subset=["NORTE","ESTE"]).copy()

    # Heurística WGS84 vs UTM
    lat_like = df["NORTE"].between(-90, 90).mean() > 0.95
    lon_like = df["ESTE"].between(-180, 180).mean() > 0.95
    if lat_like and lon_like:
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["ESTE"], df["NORTE"]), crs=4326)
    else:
        best, best_valid = None, -1
        for epsg in (32717, 32718, 32719):
            tmp = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["ESTE"], df["NORTE"]), crs=f"EPSG:{epsg}").to_crs(4326)
            lat = tmp.geometry.y; lon = tmp.geometry.x
            valid = (lat.between(-19, 1) & lon.between(-85, -67)).sum()
            if valid > best_valid:
                best, best_valid = tmp, valid
        g = best.loc[best.geometry.y.between(-19,1) & best.geometry.x.between(-85,-67)].copy()

    # Etiqueta robusta
    g["lat"] = g.geometry.y; g["lon"] = g.geometry.x
    name_candidates = ["Nombre del establecimiento","NOMBRE DEL ESTABLECIMIENTO","NOMBRE_ESTABLECIMIENTO","NOMBRE","ESTABLECIMIENTO","RAZON_SOCIAL"]
    inst_candidates = ["INSTITUCIÓN","INSTITUCION","INSTITUTION"]
    dep_candidates  = ["DEPARTAMENTO","DEPARTAMEN","DEPARTAM","DPTO"]
    dis_candidates  = ["DISTRITO","NOMB_DIST","DISTRITO_NOM"]

    def first_col(cols):
        for c in cols:
            if c in g.columns: return c
        return None
    name_col = first_col(name_candidates)
    inst_col = first_col(inst_candidates)
    dep_col  = first_col(dep_candidates)
    dis_col  = first_col(dis_candidates)

    if name_col:
        base = g[name_col].astype(str).fillna("").str.strip().replace({"nan":"", "None":""})
    else:
        base = pd.Series([""]*len(g), index=g.index)

    parts = []
    if inst_col: parts.append(g[inst_col].astype(str).fillna("").str.strip())
    if dep_col:  parts.append(g[dep_col].astype(str).fillna("").str.strip())
    if dis_col:  parts.append(g[dis_col].astype(str).fillna("").str.strip())
    if parts:
        joined = pd.concat(parts, axis=1).fillna("").astype(str)
        fallback = joined.apply(lambda r: " - ".join([x for x in r.values if x]), axis=1)
    else:
        fallback = pd.Series([""], index=g.index)

    label = base.where(base.str.len()>0, fallback)
    label = label.replace({"": "Hospital sin nombre"}).fillna("Hospital sin nombre")
    g["label"] = label
    return g

# =========================================
# AGREGACIONES (sin cache para evitar hashing de GeoDataFrame)
# =========================================
def hospitals_by_department(hosp_gdf: gpd.GeoDataFrame, dist_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # sjoin robusto
    s = gpd.sjoin(hosp_gdf.set_crs(4326), dist_gdf[["__DEPT__","geometry"]],
                  how="left", predicate="within")
    s["__DEPT__"] = s["__DEPT__"].fillna("DESCONOCIDO")
    return (s.groupby("__DEPT__", dropna=False)
              .size()
              .reset_index(name="hospitals")
              .sort_values("hospitals", ascending=False))

def hospitals_by_district(hosp_gdf: gpd.GeoDataFrame, dist_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    d = dist_gdf.copy()
    name_candidates = ["NOMBDIST","NOMB_DIST","DISTRITO","NOM_DIST","NOMBRE","NOMBDISTRI"]
    dist_name = next((c for c in name_candidates if c in d.columns), None)
    if not dist_name:
        dist_name = "__DIST__"
        d[dist_name] = d.index.astype(str)

    # Usa intersects si sospechas problemas de geometría:
    s = gpd.sjoin(hosp_gdf.set_crs(4326),
                  d[[dist_name,"__DEPT__","geometry"]],
                  how="left", predicate="intersects")
    s[dist_name] = s[dist_name].fillna("SIN NOMBRE")
    return (s.groupby([dist_name,"__DEPT__"])
              .size()
              .reset_index(name="hospitals")
              .sort_values("hospitals", ascending=False))

# =========================================
# GRÁFICOS (Matplotlib)
# =========================================
def plot_bar_generic(df: pd.DataFrame, x_col: str, y_col: str,
                     title: str, rotate_x: int = 80,
                     width: float = 12, height: float = 5):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    df = df.copy()
    df[x_col] = df[x_col].astype(str)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col)
    ax.tick_params(axis='x', labelrotation=rotate_x)
    fig.tight_layout()
    return fig

def static_map_points_over_districts(hosp_gdf: gpd.GeoDataFrame, dist_gdf: gpd.GeoDataFrame,
                                     title="Hospitales operativos sobre límites distritales"):
    fig, ax = plt.subplots(figsize=(9, 9))
    dist_gdf.boundary.plot(ax=ax, linewidth=0.4, color="#666666")
    hosp_gdf.plot(ax=ax, markersize=2, alpha=0.7, color="#1f77b4")
    ax.set_title(title); ax.set_axis_off(); fig.tight_layout()
    return fig

# ==================
# CARGA DE DATOS
# ==================
dist_gdf = load_districts_wgs84()
hosp_gdf = load_hospitals_points()

# =========================================
# PESTAÑAS
# =========================================
tab1, tab2, tab3 = st.tabs(["📁 Descripción de Datos", "🗺️ Mapas Estáticos", "🌍 Mapas Dinámicos"])

# ----------------------------
# TAB 1 — Descripción de Datos
# ----------------------------
with tab1:
    st.subheader("Descripción de los Datos")

    total_hosp = len(hosp_gdf)
    dept_count = dist_gdf["__DEPT__"].nunique()
    hosp_by_dept = hospitals_by_department(hosp_gdf, dist_gdf)
    max_dept = hosp_by_dept["hospitals"].max() if not hosp_by_dept.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Hospitales", f"{total_hosp:,}")
    col2.metric("Departamentos", f"{dept_count}")
    col3.metric("Máximo en un Dpto.", f"{max_dept}")

    st.markdown("---")
    st.markdown("""
**Unidad de análisis:** establecimientos hospitalarios públicos **operativos** en Perú.  
**Fuentes:** MINSA — IPRESS (subset operativo), shapefiles de **Centros Poblados** y **Distritos**.  
**Reglas de filtrado:** solo hospitales **operativos** con coordenadas válidas reproyectadas a **EPSG:4326**.
    """)

    st.markdown("#### Vista rápida del dataset de hospitales")
    st.dataframe(hosp_gdf.drop(columns=["geometry"]).head(25), use_container_width=True)

    st.markdown("#### Distribución por Distrito (Top 20)")
    dist_df = hospitals_by_district(hosp_gdf, dist_gdf)
    if dist_df.empty:
        st.warning("No se encontraron cruces hospital–distrito. Revisa CRS y geometrías.")
    else:
        st.dataframe(dist_df.head(20), use_container_width=True)
        dist_name_col = dist_df.columns[0]
        fig_top = plot_bar_generic(
            dist_df.head(20).sort_values("hospitals", ascending=False),
            x_col=dist_name_col, y_col="hospitals",
            title="Top 20 distritos por número de hospitales"
        )
        if fig_top is None:
            st.error("No se pudo construir el gráfico (DF vacío o columnas faltantes).")
        else:
            st.pyplot(fig_top, clear_figure=True)

# ------------------------------------------------------
# TAB 2 — Mapas Estáticos (GeoPandas + HTML embebido)
# ------------------------------------------------------
with tab2:
    st.subheader("Mapas Estáticos & Análisis por Departamento")

    st.markdown("#### Resumen por Departamento")
    by_dept = hospitals_by_department(hosp_gdf, dist_gdf)
    st.dataframe(by_dept, use_container_width=True)

    st.markdown("#### Gráfico de Barras (Departamentos)")
    fig_bar = plot_bar_generic(by_dept, x_col="__DEPT__", y_col="hospitals",
                               title="Hospitales operativos por departamento",
                               rotate_x=70, width=12, height=5)
    if fig_bar is not None:
        st.pyplot(fig_bar, clear_figure=True)

    st.markdown("#### Mapa Estático (GeoPandas)")
    fig_static = static_map_points_over_districts(hosp_gdf, dist_gdf)
    st.pyplot(fig_static, clear_figure=True)

    st.markdown("#### Coropleta Nacional (HTML incrustado)")
    chor_html = ASSETS / "national_choropleth_hospitals.html"
    if chor_html.exists():
        components.html(chor_html.read_text(encoding="utf-8"), height=650, scrolling=True)
    else:
        st.warning(f"No se encontró {chor_html}. Coloca el archivo en esa ruta.")

# -----------------------------
# TAB 3 — Mapas Dinámicos (Proximidad)
# -----------------------------
with tab3:
    st.subheader("Mapas Dinámicos — Proximidad (Folium)")
    st.caption("Círculos de 10 km: verde = mayor densidad hospitalaria, rojo = menor densidad.")

    lima_html   = ASSETS / "proximity_lima_10km.html"
    loreto_html = ASSETS / "proximity_loreto_10km.html"
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Lima — concentración urbana y accesibilidad**")
        if lima_html.exists():
            components.html(lima_html.read_text(encoding="utf-8"), height=600, scrolling=True)
        else:
            st.warning(f"Falta {lima_html.name}. Ejecuta el script de proximidad primero.")

    with colB:
        st.markdown("**Loreto — dispersión geográfica y retos de accesibilidad**")
        if loreto_html.exists():
            components.html(loreto_html.read_text(encoding="utf-8"), height=600, scrolling=True)
        else:
            st.warning(f"Falta {loreto_html.name}. Ejecuta el script de proximidad primero.")
