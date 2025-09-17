# Analysis Scripts Documentation

This document provides detailed documentation for all analysis scripts in the `_data/scripts/` directory, including their purpose, inputs, outputs, and usage instructions.

## 📁 Script Overview

The analysis pipeline consists of 7 main scripts organized in two phases:

### Phase 1: Data Processing & Static Analysis (1.x)
1. **1.1.filter_hospitals.py** - Data preprocessing and filtering
2. **1.2.hospital_maps.py** - Static district-level maps
3. **1.3.hospital_by_department.py** - Department-level analysis
4. **1.4.proximity_analysis.py** - Proximity buffer analysis

### Phase 2: Interactive Visualization & Web App (2.x)
5. **2.1.folium_interactive.py** - Interactive national maps
6. **2.2.proximity_lim_loreto.py** - Regional proximity maps
7. **2.3.streamlit_application.py** - Web application

---

## 📋 Detailed Script Documentation

### 1.1.filter_hospitals.py

**Purpose**: Filters raw IPRESS data to extract operational public hospitals with valid coordinates.

#### Inputs:
- `_data/raw/IPRESS.csv` - Raw hospital registry from MINSA

#### Outputs:
- `_data/operational_hospitals.csv` - Cleaned hospital dataset

#### Filtering Criteria:
1. **Operational Status**: `Condición == 'EN FUNCIONAMIENTO'`
2. **Hospital Classification**: Contains "HOSPITALES O CLINICAS DE ATENCION"
3. **Public Institutions**: `Institucion != 'PRIVADO'`
4. **Valid Coordinates**: Non-null NORTE and ESTE values

#### Usage:
```bash
cd _data/scripts
python 1.1.filter_hospitals.py
```

#### Key Functions:
- **Column Standardization**: Sets proper column names for consistency
- **Coordinate Validation**: Removes records with missing/invalid coordinates
- **Data Quality Reporting**: Prints filtering statistics at each step

---

### 1.2.hospital_maps.py

**Purpose**: Generates static choropleth maps showing hospital distribution by district.

#### Inputs:
- `_data/operational_hospitals.csv` - Filtered hospital data
- `_data/DISTRITOS/DISTRITOS.shp` - District boundaries shapefile

#### Outputs:
- `assets/district_analysis/map1_hospital_count.png` - Hospital count choropleth
- `assets/district_analysis/map2_zero_hospitals.png` - Districts with no hospitals
- `assets/district_analysis/map3_top10_hospitals.png` - Top 10 districts by hospital count

#### Features:
- **Spatial Join**: Matches hospitals to districts using point-in-polygon
- **Color Schemes**: Uses appropriate color palettes for different visualizations
- **Missing Data Handling**: Identifies and highlights districts without hospitals

#### Usage:
```bash
python 1.2.hospital_maps.py
```

#### Technical Details:
- **CRS Handling**: Ensures both datasets use EPSG:4326
- **Geometry Validation**: Handles invalid geometries and coordinate system issues
- **Export Settings**: High-quality PNG output with proper DPI settings

---

### 1.3.hospital_by_department.py

**Purpose**: Performs department-level analysis and generates summary visualizations.

#### Inputs:
- `_data/operational_hospitals.csv` - Filtered hospital data
- `_data/DISTRITOS/DISTRITOS.shp` - Administrative boundaries

#### Outputs:
- `assets/department_analysis/department_hospital_summary.csv` - Department statistics
- `assets/department_analysis/department_hospital_barchart.png` - Bar chart visualization
- `assets/department_analysis/department_hospital_choropleth.png` - Department choropleth

#### Analysis Components:
- **Aggregation**: Counts hospitals by department
- **Ranking**: Orders departments by hospital count
- **Statistical Summary**: Calculates mean, median, and distribution metrics

#### Usage:
```bash
python 1.3.hospital_by_department.py
```

---

### 1.4.proximity_analysis.py

**Purpose**: Performs comprehensive proximity analysis using 10km buffers around population centers.

#### Inputs:
- `_data/operational_hospitals.csv` - Hospital locations
- `_data/CCPP_0/CCPP_IGN100K.shp` - Population centers shapefile

#### Outputs:
- `assets/proximity_analysis/proximity_analysis_summary.csv` - Complete analysis results
- `assets/proximity_analysis/proximity_analysis_lima.csv` - Lima-specific data
- `assets/proximity_analysis/proximity_analysis_loreto.csv` - Loreto-specific data
- `assets/proximity_analysis/extreme_cases_summary.csv` - Isolation/concentration cases
- `assets/proximity_analysis/*.html` - Interactive maps for extreme cases

#### Methodology:
1. **Buffer Creation**: 10km circular buffers around population center centroids
2. **Hospital Counting**: Count hospitals within each buffer using spatial intersection
3. **Extreme Case Identification**: Find areas with minimum/maximum hospital access
4. **Coordinate System Handling**: UTM Zone 18S for metric calculations, WGS84 for visualization

#### Usage:
```bash
python 1.4.proximity_analysis.py
```

#### Key Parameters:
- **Buffer Distance**: 10 kilometers
- **Target Departments**: Lima (urban) and Loreto (rural)
- **Analysis CRS**: EPSG:32718 (UTM Zone 18S)

---

### 2.1.folium_interactive.py

**Purpose**: Creates interactive national choropleth map with hospital markers using Folium.

#### Inputs:
- `_data/operational_hospitals.csv` - Hospital locations
- `_data/DISTRITOS/DISTRITOS.shp` - District boundaries

#### Outputs:
- `assets/national_choropleth_hospitals.html` - Interactive national map

#### Features:
- **Choropleth Layer**: District-level hospital density coloring
- **Marker Clustering**: Grouped hospital markers for better performance
- **Interactive Controls**: Layer toggles, zoom, pan, and tooltips
- **Responsive Design**: Optimized for web embedding

#### Usage:
```bash
python 2.1.folium_interactive.py
```

#### Technical Implementation:
- **Coordinate System Detection**: Automatic detection and correction of coordinate systems
- **Color Palettes**: Sequential color schemes for density visualization  
- **Performance Optimization**: Marker clustering for large datasets
- **Cross-browser Compatibility**: Uses Leaflet.js for broad browser support

---

### 2.2.proximity_lim_loreto.py

**Purpose**: Generates regional proximity maps for Lima and Loreto departments.

#### Inputs:
- `_data/operational_hospitals.csv` - Hospital data
- Proximity analysis results from script 1.4

#### Outputs:
- `assets/proximity_lima_10km.html` - Lima proximity map
- `assets/proximity_loreto_10km.html` - Loreto proximity map
- `assets/proximity_*_extremes_10km.csv` - Extreme cases data

#### Map Features:
- **Buffer Visualization**: 10km circles around population centers
- **Color Coding**: Green (high access) to red (low access)
- **Comparative Analysis**: Side-by-side urban vs. rural accessibility

#### Usage:
```bash
python 2.2.proximity_lim_loreto.py
```

---

### 2.3.streamlit_application.py

**Purpose**: Web application providing interactive access to all analysis results.

#### Dependencies:
- All previous scripts' outputs
- Streamlit framework
- GeoPandas, Folium, Matplotlib

#### Features:

##### 🗂️ Tab 1: Data Description
- **Raw IPRESS Data Explorer**: Interactive table with filtering
- **Search Functionality**: Text search and department filtering
- **Download Options**: CSV export of filtered data
- **Filtering Statistics**: Real-time counts of applied filters
- **Methodology Documentation**: Analysis approach and data sources

##### 🗺️ Tab 2: Static Maps & Department Analysis
- **District Analysis Maps**: All three static maps from script 1.2
- **Department Analysis**: Summary tables and visualizations
- **Interactive Charts**: Department-level bar charts
- **Download Capabilities**: Access to generated CSV files

##### 🌍 Tab 3: Dynamic Maps
- **National Choropleth**: Interactive national map with markers
- **Proximity Maps**: Enhanced Lima and Loreto maps with better default views
- **Analysis Data**: Extreme cases and statistical summaries
- **Download Options**: All maps and data available for download

#### Usage:
```bash
cd _data/scripts
streamlit run 2.3.streamlit_application.py
```

#### Technical Features:
- **Caching**: Efficient data loading with Streamlit caching
- **Responsive Layout**: Optimized for different screen sizes
- **Error Handling**: Graceful handling of missing files
- **User Guidance**: Clear instructions for regenerating missing outputs

---

## 🔄 Execution Workflow

### Complete Analysis Pipeline:
```bash
cd _data/scripts

# Phase 1: Data Processing
python 1.1.filter_hospitals.py          # Filter raw data
python 1.2.hospital_maps.py             # Generate district maps  
python 1.3.hospital_by_department.py    # Department analysis
python 1.4.proximity_analysis.py        # Proximity analysis

# Phase 2: Interactive Visualization
python 2.1.folium_interactive.py        # National interactive map
python 2.2.proximity_lim_loreto.py      # Regional proximity maps

# Launch Web Application
streamlit run 2.3.streamlit_application.py
```

### Partial Execution:
Scripts can be run independently, but some depend on outputs from earlier scripts:

**Dependencies:**
- Scripts 1.2-1.4 require 1.1 (filtered hospital data)
- Script 2.1 requires 1.1 (hospital data and district boundaries)
- Script 2.2 requires 1.4 (proximity analysis results)
- Script 2.3 works with any available outputs (graceful degradation)

---

## 🛠️ Technical Requirements

### Python Version:
- Python 3.8 or higher

### Key Dependencies:
- **GeoPandas** (≥0.10.0): Geospatial data processing
- **Pandas** (≥1.3.0): Data manipulation
- **Matplotlib** (≥3.5.0): Static plotting
- **Folium** (≥0.12.0): Interactive web maps
- **Streamlit** (≥1.0.0): Web application framework

### System Requirements:
- **GDAL/OGR**: For shapefile reading (usually installed with GeoPandas)
- **PROJ**: For coordinate system transformations
- **Memory**: Minimum 4GB RAM recommended for processing national datasets

---

## 🔍 Troubleshooting

### Common Issues:

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Coordinate System Errors**:
   - Scripts automatically detect and handle mixed coordinate systems
   - Check that input shapefiles have valid CRS information

3. **Memory Issues**:
   - Large shapefiles may require more RAM
   - Consider using smaller subsets for testing

4. **File Not Found Errors**:
   - Ensure all required input files are present
   - Check file paths in script configurations

5. **Streamlit Port Conflicts**:
   ```bash
   streamlit run 2.3.streamlit_application.py --server.port 8502
   ```

### Performance Optimization:
- **Caching**: Streamlit automatically caches data loading
- **Chunking**: Large datasets are processed in chunks where possible
- **Geometry Simplification**: Complex polygons are simplified for web display

---

## 📊 Output File Reference

### Static Images (PNG):
- `assets/district_analysis/map*.png` - District-level analysis maps
- `assets/department_analysis/*.png` - Department-level visualizations

### Interactive Maps (HTML):
- `assets/national_choropleth_hospitals.html` - National interactive map
- `assets/proximity_*_10km.html` - Regional proximity maps
- `assets/proximity_analysis/*.html` - Extreme case maps

### Data Files (CSV):
- `_data/operational_hospitals.csv` - Main filtered dataset
- `assets/department_analysis/department_hospital_summary.csv` - Department statistics
- `assets/proximity_analysis/*.csv` - Proximity analysis results

### Configuration:
- All output paths are configurable in script headers
- Default output location: `assets/` directory
- Intermediate data: `_data/` directory

---

**Last Updated**: September 2025  
**Script Version**: 1.0  
**Compatibility**: Python 3.8+, tested on macOS and Linux
