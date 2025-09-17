# Hospital Access Analysis in Peru 🏥🇵🇪

A comprehensive geospatial analysis project examining the distribution and accessibility of operational public hospitals across Peru, with special focus on geographic disparities and proximity analysis for population centers.

## 📊 Project Overview

This project analyzes the accessibility and distribution of operational public hospitals in Peru using geospatial data science techniques. It combines hospital registry data (IPRESS) with administrative boundaries and population center data to provide insights into healthcare accessibility across different regions.

### Key Features

- **Hospital Distribution Analysis**: Visualization of hospital counts by department and district
- **Proximity Analysis**: Distance-based accessibility analysis for population centers
- **Interactive Visualizations**: Dynamic maps using Folium and static maps with GeoPandas/Matplotlib  
- **Web Application**: Streamlit-based dashboard for interactive exploration
- **Comparative Regional Analysis**: Focused comparison between Lima (urban) and Loreto (rural/remote)

### Research Questions Addressed

1. How are public hospitals distributed across Peru's departments and districts?
2. Which population centers have the least access to hospital services?
3. What are the geographic disparities in hospital accessibility?
4. How does urban vs. rural hospital accessibility compare?

## 🗂️ Project Structure

```
Hospitals-Access-Peru/
├── _data/                          # Data directory
│   ├── CCPP_0/                    # Population centers shapefile
│   ├── DISTRITOS/                 # Districts shapefile  
│   ├── raw/                       # Raw data files
│   │   └── IPRESS.csv            # Original hospital registry
│   ├── operational_hospitals.csv  # Cleaned hospital data
│   └── scripts/                   # Analysis scripts
│       ├── 1.1.filter_hospitals.py      # Data preprocessing
│       ├── 1.2.hospital_maps.py         # Static map generation
│       ├── 1.3.hospital_by_department.py # Department analysis
│       ├── 1.4.proximity_analysis.py    # Proximity calculations
│       ├── 2.1.folium_interactive.py    # Interactive maps
│       ├── 2.2.proximity_lim_loreto.py  # Lima-Loreto comparison
│       └── 2.3.streamlit_application.py # Web application
├── assets/                        # Generated outputs
│   ├── department_analysis/       # Department-level visualizations
│   ├── district_analysis/         # District-level maps
│   └── proximity_analysis/        # Proximity analysis results
├── src/                          # Additional source code
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Hospitals-Access-Peru.git
   cd Hospitals-Access-Peru
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data files**
   Ensure the following data files are present:
   - `_data/raw/IPRESS.csv` - Original hospital registry
   - `_data/DISTRITOS/DISTRITOS.shp` - Districts shapefile
   - `_data/CCPP_0/CCPP_IGN100K.shp` - Population centers shapefile

## 🔄 Analysis Workflow

### Step 1: Data Preprocessing
```bash
cd _data/scripts
python 1.1.filter_hospitals.py
```
Filters the raw IPRESS data to keep only operational public hospitals with valid coordinates.

### Step 2: Generate District Maps
```bash
python 1.2.hospital_maps.py
```
Creates static choropleth maps showing hospital distribution by district.

### Step 3: Department Analysis
```bash
python 1.3.hospital_by_department.py
```
Generates department-level analysis and visualizations.

### Step 4: Proximity Analysis
```bash
python 1.4.proximity_analysis.py
```
Performs 10km buffer analysis around population centers to identify areas with limited hospital access.

### Step 5: Interactive Maps
Running 2.1.folium_interactive.py IS MANDATORY as far the output was not uploaded to GitHub due to size constrains. 
```bash
python 2.1.folium_interactive.py
python 2.2.proximity_lim_loreto.py
```
Creates interactive Folium maps for web visualization.

### Step 6: Launch Web Application
```bash
python 2.3.streamlit_application.py
```
Launches the Streamlit dashboard at `http://localhost:8501`

## 📈 Key Findings

### Hospital Distribution
- **Total Hospitals**: 243 operational public hospitals analyzed
- **Geographic Concentration**: Lima department has the highest concentration
- **Rural Gaps**: Significant disparities in rural and remote areas

### Proximity Analysis (10km Buffer)
- **Lima Department**: Higher hospital density in urban areas
- **Loreto Department**: Large areas with limited hospital access
- **Extreme Cases**: Identified population centers with 0 hospitals within 10km radius

## 🛠️ Technical Details

### Data Sources
- **IPRESS (MINSA)**: National registry of health establishments
- **INEI**: District boundaries and population center shapefiles
- **IGN**: Geographic reference data

### Coordinate Systems
- **Input Data**: Mixed UTM zones (32717, 32718, 32719) and WGS84
- **Analysis**: UTM Zone 18S (EPSG:32718) for metric calculations
- **Visualization**: WGS84 (EPSG:4326) for web mapping

### Key Libraries
- **GeoPandas**: Geospatial data processing
- **Folium**: Interactive web mapping
- **Streamlit**: Web application framework
- **Matplotlib/Seaborn**: Static visualizations
- **Pandas**: Data manipulation

## 📊 Generated Outputs

### Static Maps (`assets/district_analysis/`)
- `map1_hospital_count.png`: Hospital count choropleth
- `map2_zero_hospitals.png`: Districts with no hospitals
- `map3_top10_hospitals.png`: Districts with highest hospital counts

### Interactive Maps (`assets/proximity_analysis/`)
- `lima_concentration_map.html`: Areas with highest hospital density
- `lima_isolation_map.html`: Areas with lowest hospital access
- `loreto_concentration_map.html`: Loreto concentration analysis
- `loreto_isolation_map.html`: Loreto isolation analysis

### Data Summaries
- `department_hospital_summary.csv`: Hospital counts by department
- `proximity_analysis_summary.csv`: Complete proximity analysis results
- `extreme_cases_summary.csv`: Isolation and concentration cases

## 🌐 Web Application Features

The Streamlit application provides:

1. **Data Overview**: Summary statistics and data quality metrics
2. **Static Maps**: Department and district-level visualizations  
3. **Interactive Maps**: Folium-based proximity analysis maps
4. **Comparative Analysis**: Lima vs. Loreto accessibility comparison

Access the application:
```bash
cd _data/scripts
streamlit run 2.3.streamlit_application.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MINSA (Ministry of Health)** for providing the IPRESS hospital registry
- **INEI (National Institute of Statistics)** for administrative boundary data
- **IGN (National Geographic Institute)** for geographic reference data
- **DataScienceUP** for project guidance and support

## 📧 Contact

For questions or collaboration opportunities:
- Project Repository: [https://github.com/yourusername/Hospitals-Access-Peru](https://github.com/yourusername/Hospitals-Access-Peru)
- Issues: [https://github.com/yourusername/Hospitals-Access-Peru/issues](https://github.com/yourusername/Hospitals-Access-Peru/issues)

---

**Keywords**: geospatial analysis, healthcare accessibility, Peru, hospitals, GIS, proximity analysis, data science, public health
