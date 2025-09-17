# Data Sources and Methodology

This document provides detailed information about the data sources, preprocessing steps, and methodological decisions used in the Hospital Access Analysis project.

## 📊 Data Sources

### 1. IPRESS Hospital Registry (MINSA)

**Source**: Ministry of Health (MINSA) - National Registry of Health Establishments
**File**: `_data/raw/IPRESS.csv`
**Format**: CSV with UTF-8 encoding
**Records**: ~3000+ health establishments (before filtering)

#### Key Fields:
- `Institucion`: Institution type (MINSA, ESSALUD, GOBIERNO REGIONAL, etc.)
- `Codigo Unico`: Unique establishment code
- `Nombre del establecimiento`: Hospital/establishment name
- `Clasificación`: Classification (hospitals vs. other health facilities)
- `DEPARTAMENTO`, `Provincia`, `Distrito`: Administrative divisions
- `UBIGEO`: Geographic identifier code
- `NORTE`, `ESTE`: Coordinates (mixed coordinate systems)
- `Condición`: Operational status
- `CAMAS`: Number of beds

#### Data Quality Issues:
- Mixed coordinate systems (UTM zones and WGS84)
- Inconsistent encoding (special characters)
- Missing coordinates for some establishments
- Operational status inconsistencies

### 2. District Boundaries (INEI)

**Source**: National Institute of Statistics and Informatics (INEI)
**File**: `_data/DISTRITOS/DISTRITOS.shp` (+ associated files)
**Format**: Shapefile
**Records**: 1,874 districts

#### Key Fields:
- `IDDIST`: District identifier (UBIGEO)
- `DISTRITO`: District name
- `geometry`: Polygon geometry

#### Coordinate System:
- Original CRS: Various (typically UTM or geographic)
- Standardized to: EPSG:4326 (WGS84)

### 3. Population Centers (IGN)

**Source**: National Geographic Institute (IGN)
**File**: `_data/CCPP_0/CCPP_IGN100K.shp`
**Format**: Shapefile
**Scale**: 1:100,000

#### Key Fields:
- Population center identifiers
- Administrative divisions
- `geometry`: Point or polygon geometry

#### Usage:
- Proximity analysis reference points
- Population distribution analysis
- Accessibility calculations

## 🔄 Data Preprocessing Pipeline

### Step 1: Hospital Data Filtering (`1.1.filter_hospitals.py`)

#### Filtering Criteria:
1. **Operational Status**: Only establishments with `Condición == 'EN FUNCIONAMIENTO'`
2. **Hospital Classification**: Only facilities classified as:
   - "HOSPITALES O CLINICAS DE ATENCION GENERAL"
   - "HOSPITALES O CLINICAS DE ATENCION ESPECIALIZADA"
3. **Public Institutions**: Exclude private facilities (`Institucion != 'PRIVADO'`)
4. **Valid Coordinates**: Remove records with missing or invalid coordinates

#### Coordinate System Handling:
```python
# Check coordinate ranges for Peru bounds
# Latitude: -18.5 to 0, Longitude: -81.5 to -68
valid_standard = (
    (df['ESTE'] >= -82) & (df['ESTE'] <= -68) &
    (df['NORTE'] >= -19) & (df['NORTE'] <= 1)
).sum()

valid_swapped = (
    (df['NORTE'] >= -82) & (df['NORTE'] <= -68) &
    (df['ESTE'] >= -19) & (df['ESTE'] <= 1)
).sum()
```

#### Output:
- **Result**: 243 operational public hospitals
- **File**: `_data/operational_hospitals.csv`
- **Quality**: All records have valid coordinates and operational status

### Step 2: Coordinate System Standardization

#### Challenge:
Input data contains mixed coordinate systems:
- Some coordinates in WGS84 (decimal degrees)
- Others in UTM zones 17S, 18S, or 19S (meters)

#### Solution Strategy:
1. **Heuristic Detection**: Analyze coordinate ranges to identify likely system
2. **UTM Zone Testing**: For metric coordinates, test zones 32717, 32718, 32719
3. **Validation**: Count points falling within Peru's bounds after transformation
4. **Best Fit Selection**: Choose transformation yielding most valid points

#### Implementation:
```python
def to_wgs84_from_best_utm(df_xy, xcol, ycol):
    candidates = [32717, 32718, 32719]
    best = None
    best_valid = -1
    for epsg in candidates:
        g = gpd.GeoDataFrame(
            df_xy, 
            geometry=gpd.points_from_xy(df_xy[xcol], df_xy[ycol]),
            crs=f"EPSG:{epsg}"
        ).to_crs("EPSG:4326")
        
        # Count valid points within Peru bounds
        lat, lon = g.geometry.y, g.geometry.x
        valid = (lat.between(-19, 1) & lon.between(-85, -67)).sum()
        
        if valid > best_valid:
            best, best_valid = g, valid
    return best
```

## 🗺️ Spatial Analysis Methodology

### Proximity Analysis

#### Objective:
Identify population centers with limited hospital access using buffer analysis.

#### Parameters:
- **Buffer Distance**: 10 kilometers
- **Analysis CRS**: EPSG:32718 (UTM Zone 18S) for accurate metric calculations
- **Target Departments**: Lima (urban) and Loreto (rural/remote)

#### Process:
1. **Centroid Calculation**: Compute centroids of population center polygons
2. **Buffer Creation**: Generate 10km circular buffers around centroids
3. **Hospital Counting**: Count hospitals within each buffer
4. **Extreme Case Identification**:
   - **Isolation**: Centers with fewest hospitals (minimum)
   - **Concentration**: Centers with most hospitals (maximum)

#### Mathematical Approach:
```
For each population center i:
  1. centroid_i = geometry_i.centroid
  2. buffer_i = centroid_i.buffer(10000)  # 10km in meters
  3. hospitals_i = COUNT(hospitals WHERE ST_Within(hospital, buffer_i))
  4. accessibility_score_i = hospitals_i
```

### Choropleth Mapping

#### District-Level Analysis:
1. **Spatial Join**: Match hospitals to districts using point-in-polygon
2. **Aggregation**: Count hospitals per district
3. **Classification**: Categorize districts by hospital count
4. **Visualization**: Color-coded maps showing distribution

#### Department-Level Analysis:
1. **Aggregation**: Sum hospital counts by department
2. **Ranking**: Order departments by hospital count
3. **Visualization**: Bar charts and choropleth maps

## 🎯 Quality Assurance

### Data Validation Steps:

1. **Coordinate Validation**:
   - Check all coordinates fall within Peru's bounds
   - Verify CRS transformations produce sensible results
   - Manual spot-checking of major hospitals

2. **Administrative Boundary Matching**:
   - Verify hospital-district assignments
   - Check for hospitals falling outside district boundaries
   - Handle edge cases near boundaries

3. **Statistical Validation**:
   - Compare results with known hospital distributions
   - Validate proximity analysis results against manual calculations
   - Cross-check department totals

### Known Limitations:

1. **Coordinate Accuracy**: Original coordinates may have varying precision
2. **Temporal Mismatch**: Hospital data and boundary data may be from different time periods
3. **Administrative Changes**: District boundaries may have changed since data collection
4. **Hospital Classification**: Some facilities may be misclassified in original data

## 📈 Analysis Parameters

### Buffer Distance Justification:
- **10km chosen** based on:
  - Typical travel distance for healthcare access in developing countries
  - Balance between local access and regional coverage
  - Computational feasibility for analysis

### Department Selection:
- **Lima**: Represents urban, high-density scenario
- **Loreto**: Represents rural, low-density, geographically challenging scenario
- Provides maximum contrast for comparative analysis

### Filtering Criteria Rationale:
- **Public hospitals only**: Focus on publicly accessible healthcare
- **Operational status**: Ensure facilities are actually providing services
- **Hospital classification**: Exclude primary care centers to focus on hospital-level care

## 🔍 Reproducibility Notes

### Software Versions:
- Python 3.8+
- GeoPandas 0.10+
- Pandas 1.3+
- Folium 0.12+
- Streamlit 1.0+

### Random Seeds:
- No randomization used in analysis
- Results are fully deterministic

### Environment Dependencies:
- GDAL/OGR for spatial data reading
- PROJ for coordinate system transformations
- Shapely for geometric operations

---

**Last Updated**: September 2025
**Data Collection Period**: 2023-2024 (estimated)
**Analysis Period**: September 2025
