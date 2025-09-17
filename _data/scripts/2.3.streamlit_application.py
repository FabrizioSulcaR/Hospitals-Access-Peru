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
RAW_IPRESS_CSV = DATA_DIR / "raw" / "IPRESS.csv"
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
def load_raw_ipress() -> pd.DataFrame:
    """Load raw IPRESS data with proper column names."""
    if not RAW_IPRESS_CSV.exists():
        raise FileNotFoundError(f"No se encontró {RAW_IPRESS_CSV}")
    
    df = pd.read_csv(RAW_IPRESS_CSV, encoding='utf-8', encoding_errors='replace')
    
    # Set proper column names (same as in filter_hospitals.py)
    df.columns = [
        'Institucion', 'Codigo Unico', 'Nombre del establecimiento',
        'Clasificación', 'Tipo', 'DEPARTAMENTO', 'Provincia', 'Distrito',
        'UBIGEO', 'Dirección', 'Código DISA', 'Código Red',
        'Código Microrred', 'DISA', 'Red', 'Microrred', 'Código UE',
        'Unidad Ejecutora', 'Categoría', 'Teléfono',
        'Tipo Doc.Categorización', 'Nro.Doc.Categorización', 'Horario',
        'Inicio de Actividad',
        'Director Médico y/o Responsable de la Atención de Salud', 'ESTADO',
        'Situación', 'Condición', 'Inspección', 'NORTE', 'ESTE', 'COTA',
        'CAMAS'
    ]
    
    return df

@st.cache_data(show_spinner=False)
def apply_hospital_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same filters as in 1.1.filter_hospitals.py"""
    
    # Filter for operational hospitals
    df_operational = df[df['Condición'].str.strip() == 'EN FUNCIONAMIENTO'].copy()
    
    # Filter for hospital classifications
    df_hospitals = df_operational[
        df_operational['Clasificación'].str.contains('HOSPITALES O CLINICAS DE ATENCION GENERAL', case=False, na=False) | 
        df_operational['Clasificación'].str.contains('HOSPITALES O CLINICAS DE ATENCION ESPECIALIZADA', case=False, na=False)
    ].copy()
    
    # Filter for public hospitals only
    df_public = df_hospitals[df_hospitals['Institucion'] != 'PRIVADO'].copy()
    
    # Filter for valid coordinates
    df_public['NORTE'] = pd.to_numeric(df_public['NORTE'], errors='coerce')
    df_public['ESTE'] = pd.to_numeric(df_public['ESTE'], errors='coerce')
    df_valid = df_public.dropna(subset=['NORTE', 'ESTE']).copy()
    
    return df_valid

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

# Load raw IPRESS data for Tab 1
try:
    raw_ipress = load_raw_ipress()
    st.session_state.raw_data_loaded = True
except FileNotFoundError:
    st.session_state.raw_data_loaded = False
    st.error("⚠️ Raw IPRESS data not found. Please ensure _data/raw/IPRESS.csv exists.")

# =========================================
# PESTAÑAS
# =========================================
tab1, tab2, tab3 = st.tabs(["🗂️ Descripción de Datos", "🗺️ Mapas Estáticos", "🌍 Mapas Dinámicos"])

# ----------------------------
# TAB 1 — Descripción de Datos
# ----------------------------
with tab1:
    st.header("🗂️ Descripción de Datos")
    
    if st.session_state.raw_data_loaded:
        # Data filtering section
        st.subheader("📊 Exploración del Dataset IPRESS")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 🎛️ Filtros de Datos")
            
            # Filter options
            show_all_data = st.checkbox("Mostrar todos los establecimientos", value=False)
            
            if not show_all_data:
                st.info("**Filtros aplicados automáticamente:**\n"
                       "✅ Solo establecimientos operativos\n"
                       "✅ Solo hospitales públicos\n"
                       "✅ Solo con coordenadas válidas")
                
                # Apply filters
                filtered_data = apply_hospital_filters(raw_ipress)
                data_to_show = filtered_data
                
                # Show filtering statistics
                st.markdown("### 📈 Estadísticas de Filtrado")
                total_records = len(raw_ipress)
                operational = len(raw_ipress[raw_ipress['Condición'].str.strip() == 'EN FUNCIONAMIENTO'])
                hospitals = len(raw_ipress[
                    (raw_ipress['Condición'].str.strip() == 'EN FUNCIONAMIENTO') &
                    (raw_ipress['Clasificación'].str.contains('HOSPITALES O CLINICAS', case=False, na=False))
                ])
                public_hospitals = len(filtered_data)
                
                st.metric("Total establecimientos", f"{total_records:,}")
                st.metric("Establecimientos operativos", f"{operational:,}")
                st.metric("Hospitales operativos", f"{hospitals:,}")
                st.metric("Hospitales públicos operativos", f"{public_hospitals:,}")
                
            else:
                st.warning("Mostrando **todos** los establecimientos del registro IPRESS")
                data_to_show = raw_ipress
        
        with col1:
            st.markdown("### 📋 Datos del Registro IPRESS")
            
            # Search functionality
            search_term = st.text_input("🔍 Buscar por nombre de establecimiento:", "")
            
            if search_term:
                mask = data_to_show['Nombre del establecimiento'].str.contains(search_term, case=False, na=False)
                display_data = data_to_show[mask]
                st.info(f"Mostrando {len(display_data)} registros que coinciden con '{search_term}'")
            else:
                display_data = data_to_show
            
            # Department filter
            departments = sorted(display_data['DEPARTAMENTO'].dropna().unique())
            selected_dept = st.selectbox("🏛️ Filtrar por departamento:", ["Todos"] + departments)
            
            if selected_dept != "Todos":
                display_data = display_data[display_data['DEPARTAMENTO'] == selected_dept]
            
            # Display the data
            st.markdown(f"**Registros mostrados:** {len(display_data):,}")
            
            # Select columns to display
            key_columns = [
                'Nombre del establecimiento', 'Institucion', 'Clasificación', 
                'DEPARTAMENTO', 'Provincia', 'Distrito', 'Condición', 'CAMAS'
            ]
            
            # Show data with pagination
            if len(display_data) > 0:
                # Pagination
                rows_per_page = st.selectbox("Registros por página:", [25, 50, 100, 200], index=0)
                total_pages = (len(display_data) - 1) // rows_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input("Página:", min_value=1, max_value=total_pages, value=1)
                    start_idx = (page - 1) * rows_per_page
                    end_idx = start_idx + rows_per_page
                    page_data = display_data.iloc[start_idx:end_idx]
                    st.caption(f"Página {page} de {total_pages}")
                else:
                    page_data = display_data
                
                st.dataframe(
                    page_data[key_columns], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar datos filtrados (CSV)",
                    data=csv,
                    file_name=f"ipress_filtered_{selected_dept.lower() if selected_dept != 'Todos' else 'all'}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No se encontraron registros con los filtros aplicados.")
    
    else:
        st.error("No se pudo cargar el dataset IPRESS. Verifica que el archivo _data/raw/IPRESS.csv existe.")
    
    st.markdown("---")
    
    # Analysis summary section
    st.subheader("📊 Resumen del Análisis")
    
    total_hosp = len(hosp_gdf)
    dept_count = dist_gdf["__DEPT__"].nunique()
    hosp_by_dept = hospitals_by_department(hosp_gdf, dist_gdf)
    max_dept = hosp_by_dept["hospitals"].max() if not hosp_by_dept.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏥 Hospitales Analizados", f"{total_hosp:,}")
    col2.metric("🗺️ Departamentos", f"{dept_count}")
    col3.metric("📈 Máximo en un Dpto.", f"{max_dept}")
    col4.metric("📍 Distritos", f"{len(dist_gdf):,}")

    st.markdown("""
    ### 🎯 Metodología de Análisis
    
    **Unidad de análisis:** Establecimientos hospitalarios públicos **operativos** en Perú.  
    
    **Fuentes de datos:**
    - 🏥 **MINSA - IPRESS**: Registro Nacional de Establecimientos de Salud
    - 🗺️ **INEI**: Límites distritales y centros poblados
    - 📍 **IGN**: Datos geográficos de referencia
    
    **Criterios de filtrado aplicados:**
    1. ✅ Solo establecimientos con condición **"EN FUNCIONAMIENTO"**
    2. ✅ Solo clasificados como **hospitales** (general o especializada)
    3. ✅ Solo instituciones **públicas** (excluye privados)
    4. ✅ Solo con **coordenadas válidas** para análisis geoespacial
    
    **Sistema de coordenadas:**
    - 🔧 **Análisis**: UTM Zone 18S (EPSG:32718) para cálculos métricos
    - 🌐 **Visualización**: WGS84 (EPSG:4326) para mapas web
    """)


# ------------------------------------------------------
# TAB 2 — Mapas Estáticos & Análisis por Departamento  
# ------------------------------------------------------
with tab2:
    st.header("🗺️ Mapas Estáticos & Análisis por Departamento")
   
    st.markdown("---")
    
    # Static Maps Section
    st.subheader("🗺️ Mapas Estáticos Generados")
    
    # District Analysis Maps
    st.markdown("### 📍 Análisis por Distrito")
    
    district_maps = [
        ("map1_hospital_count.png", "Mapa 1: Conteo de Hospitales por Distrito", 
         "Distribución del número de hospitales por distrito en todo el territorio peruano."),
        ("map2_zero_hospitals.png", "Mapa 2: Distritos sin Hospitales", 
         "Identificación de distritos que no cuentan con hospitales públicos operativos."),
        ("map3_top10_hospitals.png", "Mapa 3: Top 10 Distritos con Más Hospitales", 
         "Los 10 distritos con mayor concentración de hospitales públicos.")
    ]
    
    # Display district maps in a grid
    for i in range(0, len(district_maps), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(district_maps):
                map_file, title, description = district_maps[i + j]
                map_path = ASSETS / "district_analysis" / map_file
                
                with col:
                    st.markdown(f"#### {title}")
                    if map_path.exists():
                        st.image(str(map_path), caption=description, use_column_width=True)
                    else:
                        st.warning(f"⚠️ Imagen no encontrada: {map_file}")
                        st.info("Ejecuta el script `1.2.hospital_maps.py` para generar este mapa.")
    
    st.markdown("---")
    
    # Department Analysis Maps
    st.markdown("### 🏛️ Análisis Departamental")
    
    dept_maps = [
        ("department_hospital_barchart.png", "Gráfico de Barras Departamental", 
         "Comparación del número de hospitales entre departamentos del Perú."),
        ("department_hospital_choropleth.png", "Mapa Coroplético Departamental", 
         "Visualización coroplética de la densidad hospitalaria por departamento.")
    ]
    
    cols = st.columns(2)
    for i, (map_file, title, description) in enumerate(dept_maps):
        map_path = ASSETS / "department_analysis" / map_file
        
        with cols[i]:
            st.markdown(f"#### {title}")
            if map_path.exists():
                st.image(str(map_path), caption=description, use_column_width=True)
            else:
                st.warning(f"⚠️ Imagen no encontrada: {map_file}")
                st.info("Ejecuta el script `1.3.hospital_by_department.py` para generar este mapa.")
    
    st.markdown("---")
    
    # Additional Information
    st.markdown("---")
    st.markdown("### ℹ️ Información Adicional")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎨 Mapas Estáticos Generados con:**
        - **GeoPandas**: Procesamiento geoespacial
        - **Matplotlib**: Visualización estática
        - **Seaborn**: Paletas de colores
        
        **📊 Análisis Incluido:**
        - Conteo por distrito y departamento
        - Identificación de áreas sin servicios
        - Ranking de concentración hospitalaria
        """)
    
    with col2:
        st.info("""
        **🔧 Para Regenerar los Mapas:**
        ```bash
        cd _data/scripts
        python 1.2.hospital_maps.py
        python 1.3.hospital_by_department.py
        ```
        
        **📁 Ubicación de Archivos:**
        - `assets/district_analysis/`
        - `assets/department_analysis/`
        """)
    
# -----------------------------
# TAB 3 — Mapas Dinámicos (Folium)
# -----------------------------
with tab3:
    st.header("🌍 Mapas Dinámicos Interactivos")
    
    # National Choropleth Section
    st.subheader("🗺️ Mapa Nacional Interactivo")
    st.markdown("### 🏥 Coropleta Nacional + Marcadores de Hospitales")
    
    national_choropleth = ASSETS / "national_choropleth_hospitals.html"
    if national_choropleth.exists():
        st.info("""
        **🎨 Mapa interactivo nacional** generado con Folium que incluye:
        - 📊 **Coropleta**: Densidad hospitalaria por distrito (colores)
        - 📍 **Marcadores**: Ubicación exacta de cada hospital público
        - 🔍 **Interactividad**: Zoom, pan, tooltips informativos y clustering
        - 🎛️ **Controles**: Capas activables/desactivables
        """)
        
        # Load and enhance the national map HTML
        national_html = national_choropleth.read_text(encoding="utf-8")
        
        # Enhance the HTML with better styling for Streamlit embedding
        enhanced_national_html = national_html.replace(
            '<head>',
            '''<head>
            <style>
                .folium-map { 
                    width: 100% !important; 
                    height: 650px !important;
                    border: 2px solid #e6e6e6;
                    border-radius: 8px;
                }
                .leaflet-control-container {
                    font-size: 12px;
                }
            </style>'''
        )
        
        components.html(enhanced_national_html, height=680, scrolling=False)
        
        # Download option
        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                label="📥 Descargar Mapa Nacional (HTML)",
                data=national_html,
                file_name="national_choropleth_hospitals.html",
                mime="text/html"
            )
    else:
        st.warning("⚠️ Mapa coroplético nacional no encontrado.")
        st.info("""
        **Para generar el mapa nacional interactivo:**
        ```bash
        cd _data/scripts
        python 2.1.folium_interactive.py
        ```
        
        Este script creará un mapa interactivo que combina:
        - 🎨 Coropleta de densidad hospitalaria por distrito
        - 📍 Marcadores agrupados de hospitales individuales  
        - 🎛️ Controles de capas interactivos
        - 🔍 Tooltips informativos al hacer hover
        """)
    
    st.markdown("---")
    
    # Proximity Analysis Section
    st.subheader("📍 Análisis de Proximidad Regional")
    st.markdown("### 🎯 Accesibilidad Hospitalaria: Lima vs Loreto")
    
    st.info("""
    **Análisis de proximidad con buffers de 10 km:**
    - 🟢 **Verde**: Mayor densidad hospitalaria (mejor acceso)
    - 🔴 **Rojo**: Menor densidad hospitalaria (acceso limitado)  
    - 🔵 **Azul**: Centros poblados analizados
    - ⭕ **Círculos**: Buffers de 10 km de radio para análisis de accesibilidad
    """)

    lima_html = ASSETS / "proximity_lima_10km.html"
    loreto_html = ASSETS / "proximity_loreto_10km.html"
    
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### 🏙️ Lima — Concentración Urbana")
        st.caption("Análisis de accesibilidad en el departamento más poblado del Perú")
        
        if lima_html.exists():
            # Load and enhance Lima map
            lima_map_html = lima_html.read_text(encoding="utf-8")
            
            # Enhance with better default view and styling
            enhanced_lima_html = lima_map_html.replace(
                '<head>',
                '''<head>
                <style>
                    .folium-map { 
                        width: 100% !important; 
                        height: 500px !important;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                    }
                    .leaflet-control-container {
                        font-size: 11px;
                    }
                </style>'''
            )
            
            # Try to set better initial coordinates for Lima (approximately -12.0, -77.0)
            if '"location": [' in enhanced_lima_html:
                import re
                # Find and replace the location coordinates to center on Lima better
                enhanced_lima_html = re.sub(
                    r'"location":\s*\[[^\]]+\]',
                    '"location": [-12.046374, -77.042793]',
                    enhanced_lima_html,
                    count=1
                )
            
            # Set better zoom level for Lima
            if '"zoom_start": ' in enhanced_lima_html:
                enhanced_lima_html = re.sub(
                    r'"zoom_start":\s*\d+',
                    '"zoom_start": 9',
                    enhanced_lima_html,
                    count=1
                )
            
            components.html(enhanced_lima_html, height=520, scrolling=False)
            
            # Download button for Lima
            st.download_button(
                label="📥 Descargar Mapa Lima",
                data=lima_map_html,
                file_name="proximity_lima_10km.html",
                mime="text/html",
                key="lima_download"
            )
        else:
            st.warning(f"⚠️ Mapa de Lima no encontrado: {lima_html.name}")
            st.info("Ejecuta `python 2.2.proximity_lim_loreto.py` para generar este mapa.")

    with colB:
        st.markdown("#### 🌳 Loreto — Dispersión Geográfica")
        st.caption("Análisis de accesibilidad en la región amazónica más extensa")
        
        if loreto_html.exists():
            # Load and enhance Loreto map
            loreto_map_html = loreto_html.read_text(encoding="utf-8")
            
            # Enhance with better default view and styling
            enhanced_loreto_html = loreto_map_html.replace(
                '<head>',
                '''<head>
                <style>
                    .folium-map { 
                        width: 100% !important; 
                        height: 500px !important;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                    }
                    .leaflet-control-container {
                        font-size: 11px;
                    }
                </style>'''
            )
            
            # Try to set better initial coordinates for Loreto (approximately -4.0, -73.0)
            if '"location": [' in enhanced_loreto_html:
                import re
                enhanced_loreto_html = re.sub(
                    r'"location":\s*\[[^\]]+\]',
                    '"location": [-4.0, -73.0]',
                    enhanced_loreto_html,
                    count=1
                )
            
            # Set better zoom level for Loreto (larger region)
            if '"zoom_start": ' in enhanced_loreto_html:
                enhanced_loreto_html = re.sub(
                    r'"zoom_start":\s*\d+',
                    '"zoom_start": 7',
                    enhanced_loreto_html,
                    count=1
                )
            
            components.html(enhanced_loreto_html, height=520, scrolling=False)
            
            # Download button for Loreto
            st.download_button(
                label="📥 Descargar Mapa Loreto",
                data=loreto_map_html,
                file_name="proximity_loreto_10km.html",
                mime="text/html",
                key="loreto_download"
            )
        else:
            st.warning(f"⚠️ Mapa de Loreto no encontrado: {loreto_html.name}")
            st.info("Ejecuta `python 2.2.proximity_lim_loreto.py` para generar este mapa.")
    
    st.markdown("---")
    
    # Proximity Analysis Data Section
    st.subheader("📊 Datos del Análisis de Proximidad")
    
    # Load proximity analysis summary if available
    proximity_summary_csv = ASSETS / "proximity_analysis" / "proximity_analysis_summary.csv"
    extreme_cases_csv = ASSETS / "proximity_analysis" / "extreme_cases_summary.csv"
    
    if proximity_summary_csv.exists() or extreme_cases_csv.exists():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if extreme_cases_csv.exists():
                st.markdown("#### 🎯 Casos Extremos de Accesibilidad")
                extreme_df = pd.read_csv(extreme_cases_csv)
                st.dataframe(extreme_df, use_container_width=True)
            
            if proximity_summary_csv.exists():
                proximity_df = pd.read_csv(proximity_summary_csv)
                if len(proximity_df) > 0:
                    st.markdown("#### 📈 Estadísticas por Departamento")
                    if 'department' in proximity_df.columns and 'hospital_count' in proximity_df.columns:
                        summary_stats = proximity_df.groupby('department')['hospital_count'].agg([
                            'count', 'mean', 'min', 'max'
                        ]).round(2)
                        summary_stats.columns = ['Centros Poblados', 'Promedio', 'Mínimo', 'Máximo']
                        st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.markdown("#### 💾 Descargar Datos")
            
            if proximity_summary_csv.exists():
                st.download_button(
                    label="📥 Resumen Completo (CSV)",
                    data=proximity_summary_csv.read_text(encoding='utf-8'),
                    file_name="proximity_analysis_summary.csv",
                    mime="text/csv",
                    key="proximity_summary_download"
                )
            
            if extreme_cases_csv.exists():
                st.download_button(
                    label="📥 Casos Extremos (CSV)",
                    data=extreme_cases_csv.read_text(encoding='utf-8'),
                    file_name="extreme_cases_summary.csv",
                    mime="text/csv",
                    key="extreme_cases_download"
                )
            
            # Individual department data
            lima_csv = ASSETS / "proximity_analysis" / "proximity_analysis_lima.csv"
            if lima_csv.exists():
                st.download_button(
                    label="📥 Datos Lima (CSV)",
                    data=lima_csv.read_text(encoding='utf-8'),
                    file_name="proximity_analysis_lima.csv",
                    mime="text/csv",
                    key="lima_data_download"
                )
            
            loreto_csv = ASSETS / "proximity_analysis" / "proximity_analysis_loreto.csv"
            if loreto_csv.exists():
                st.download_button(
                    label="📥 Datos Loreto (CSV)",
                    data=loreto_csv.read_text(encoding='utf-8'),
                    file_name="proximity_analysis_loreto.csv",
                    mime="text/csv",
                    key="loreto_data_download"
                )
    else:
        st.warning("📊 Datos de análisis de proximidad no encontrados.")
        st.info("Ejecuta `python 1.4.proximity_analysis.py` para generar los datos de análisis.")
    
    # Technical Information
    st.markdown("---")
    st.markdown("### ℹ️ Información Técnica del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🛠️ Herramientas Utilizadas:**
        - **Folium**: Mapas interactivos web con Leaflet.js
        - **GeoPandas**: Análisis geoespacial y buffers
        - **Streamlit**: Interfaz web interactiva
        - **Pandas**: Procesamiento de datos tabulares
        
        **🎨 Características de los Mapas:**
        - Marcadores agrupados (clustering)
        - Tooltips informativos
        - Controles de zoom y pan
        - Capas activables/desactivables
        """)
    
    with col2:
        st.info("""
        **📏 Parámetros del Análisis:**
        - **Radio de Buffer**: 10 km
        - **CRS para Análisis**: UTM Zone 18S (EPSG:32718)
        - **CRS para Visualización**: WGS84 (EPSG:4326)
        - **Departamentos Analizados**: Lima y Loreto
        
        **🔧 Scripts para Regenerar:**
        ```bash
        cd _data/scripts
        python 2.1.folium_interactive.py    # Mapa nacional
        python 1.4.proximity_analysis.py    # Análisis proximidad
        python 2.2.proximity_lim_loreto.py  # Mapas regionales
        ```
        """)
