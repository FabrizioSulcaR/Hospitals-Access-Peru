#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Proximity Analysis of Operational Public Hospitals in Peru.

This script analyzes proximity between population centers and hospitals in Lima and Loreto departments:
1. Computes centroids of population centers
2. Creates 10km buffers around each population center
3. Counts hospitals within each buffer
4. Identifies isolation (fewest hospitals) and concentration (most hospitals) centers
5. Creates interactive Folium maps for visualization

Uses EPSG:32718 (UTM Zone 18S) for accurate metric calculations, then reprojects to EPSG:4326 for visualization.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOSPITALS_CSV = os.path.join(BASE_DIR, '_data', 'operational_hospitals.csv')
POPULATION_CENTERS_SHP = os.path.join(BASE_DIR, '_data', 'CCPP_0', 'CCPP_IGN100K.shp')
OUTPUT_DIR = os.path.join(BASE_DIR, 'assets')
PROXIMITY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'proximity_analysis')

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROXIMITY_OUTPUT_DIR, exist_ok=True)

# Analysis parameters
BUFFER_DISTANCE_KM = 10
TARGET_DEPARTMENTS = ['LIMA', 'LORETO']
UTM_CRS = 'EPSG:32718'  # UTM Zone 18S for Peru (metric calculations)
WGS84_CRS = 'EPSG:4326'  # WGS84 for visualization

def load_data():
    """Load hospitals and population centers data."""
    print("📊 LOADING DATA")
    print("=" * 50)
    
    # Load hospitals
    print("Loading operational hospitals data...")
    hospitals = pd.read_csv(HOSPITALS_CSV, encoding='utf-8', encoding_errors='replace')
    print(f"✅ Loaded {len(hospitals)} operational public hospitals")
    
    # Load population centers
    print("\nLoading population centers shapefile...")
    try:
        pop_centers = gpd.read_file(POPULATION_CENTERS_SHP)
        print(f"✅ Loaded {len(pop_centers)} population centers")
        print(f"Population centers CRS: {pop_centers.crs}")
        print(f"Population centers columns: {pop_centers.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error loading population centers: {e}")
        return None, None
    
    return hospitals, pop_centers

def prepare_hospitals_geodataframe(hospitals):
    """Convert hospitals DataFrame to GeoDataFrame with proper coordinates."""
    print("\n🏥 PREPARING HOSPITALS GEODATAFRAME")
    print("=" * 50)
    
    print("Converting hospitals to GeoDataFrame...")
    
    # Check for missing coordinates
    missing_coords = hospitals[['ESTE', 'NORTE']].isnull().any(axis=1).sum()
    print(f"Hospitals with missing coordinates: {missing_coords}")
    
    # Remove hospitals with missing coordinates
    hospitals_clean = hospitals.dropna(subset=['ESTE', 'NORTE']).copy()
    print(f"Hospitals with valid coordinates: {len(hospitals_clean)}")
    
    # Convert to numeric
    hospitals_clean['ESTE'] = pd.to_numeric(hospitals_clean['ESTE'], errors='coerce')
    hospitals_clean['NORTE'] = pd.to_numeric(hospitals_clean['NORTE'], errors='coerce')
    
    # Check coordinate ranges for Peru
    print(f"ESTE range: {hospitals_clean['ESTE'].min():.6f} to {hospitals_clean['ESTE'].max():.6f}")
    print(f"NORTE range: {hospitals_clean['NORTE'].min():.6f} to {hospitals_clean['NORTE'].max():.6f}")
    
    # Determine coordinate interpretation (similar to previous scripts)
    # Peru bounds: Latitude -18.5 to 0, Longitude -81.5 to -68
    valid_standard = (
        (hospitals_clean['ESTE'] >= -82) & (hospitals_clean['ESTE'] <= -68) &
        (hospitals_clean['NORTE'] >= -19) & (hospitals_clean['NORTE'] <= 1)
    ).sum()
    
    valid_swapped = (
        (hospitals_clean['NORTE'] >= -82) & (hospitals_clean['NORTE'] <= -68) &
        (hospitals_clean['ESTE'] >= -19) & (hospitals_clean['ESTE'] <= 1)
    ).sum()
    
    if valid_standard > valid_swapped:
        print("✅ Using standard interpretation: ESTE=longitude, NORTE=latitude")
        hospitals_gdf = gpd.GeoDataFrame(
            hospitals_clean,
            geometry=gpd.points_from_xy(hospitals_clean['ESTE'], hospitals_clean['NORTE']),
            crs=WGS84_CRS
        )
    else:
        print("✅ Using swapped interpretation: ESTE=latitude, NORTE=longitude")
        hospitals_gdf = gpd.GeoDataFrame(
            hospitals_clean,
            geometry=gpd.points_from_xy(hospitals_clean['NORTE'], hospitals_clean['ESTE']),
            crs=WGS84_CRS
        )
    
    print(f"Created GeoDataFrame with {len(hospitals_gdf)} hospitals")
    return hospitals_gdf

def prepare_population_centers(pop_centers, target_departments):
    """Prepare population centers for target departments."""
    print(f"\n🏘️ PREPARING POPULATION CENTERS")
    print("=" * 50)
    
    # Check available columns
    print("Available columns in population centers:")
    for col in pop_centers.columns:
        print(f"  - {col}")
    
    # Look for department-related columns
    dept_columns = [col for col in pop_centers.columns if 'DEPART' in col.upper() or 'DEPT' in col.upper() or col.upper() == 'DEP']
    print(f"Department-related columns: {dept_columns}")
    
    if not dept_columns:
        print("⚠️ No department column found. Checking first few rows...")
        print(pop_centers.head())
        return None
    
    # Use the first department column found
    dept_col = dept_columns[0]
    print(f"Using department column: {dept_col}")
    
    # Check unique departments
    unique_depts = pop_centers[dept_col].unique()
    print(f"Available departments: {sorted(unique_depts)}")
    
    # Filter for target departments
    filtered_centers = pop_centers[pop_centers[dept_col].isin(target_departments)].copy()
    print(f"Population centers in target departments: {len(filtered_centers)}")
    
    for dept in target_departments:
        count = len(filtered_centers[filtered_centers[dept_col] == dept])
        print(f"  - {dept}: {count} population centers")
    
    # Ensure proper CRS
    if filtered_centers.crs != WGS84_CRS:
        print(f"Converting population centers from {filtered_centers.crs} to {WGS84_CRS}")
        filtered_centers = filtered_centers.to_crs(WGS84_CRS)
    
    # Calculate centroids if needed
    print("Calculating centroids...")
    filtered_centers['centroid'] = filtered_centers.geometry.centroid
    
    return filtered_centers, dept_col

def calculate_proximity_analysis(hospitals_gdf, pop_centers, dept_col, buffer_distance_km=10):
    """Calculate proximity analysis for each population center."""
    print(f"\n📍 PROXIMITY ANALYSIS (Buffer: {buffer_distance_km}km)")
    print("=" * 60)
    
    # Convert to UTM for accurate metric calculations
    print(f"Converting to {UTM_CRS} for metric calculations...")
    hospitals_utm = hospitals_gdf.to_crs(UTM_CRS)
    pop_centers_utm = pop_centers.to_crs(UTM_CRS)
    
    # Create centroids in UTM
    pop_centers_utm['centroid_utm'] = pop_centers_utm.geometry.centroid
    
    # Convert buffer distance to meters
    buffer_distance_m = buffer_distance_km * 1000
    
    results = []
    
    for dept in TARGET_DEPARTMENTS:
        print(f"\nAnalyzing {dept} department...")
        
        # Filter data for current department
        dept_pop_centers = pop_centers_utm[pop_centers_utm[dept_col] == dept].copy()
        dept_hospitals = hospitals_utm[hospitals_utm['DEPARTAMENTO'] == dept].copy()
        
        print(f"  Population centers: {len(dept_pop_centers)}")
        print(f"  Hospitals: {len(dept_hospitals)}")
        
        if len(dept_pop_centers) == 0 or len(dept_hospitals) == 0:
            print(f"  ⚠️ Skipping {dept} - insufficient data")
            continue
        
        # Calculate buffers around population center centroids
        dept_pop_centers['buffer'] = dept_pop_centers['centroid_utm'].buffer(buffer_distance_m)
        
        # Count hospitals within each buffer
        for idx, pop_center in dept_pop_centers.iterrows():
            buffer_geom = pop_center['buffer']
            
            # Count hospitals within buffer
            hospitals_in_buffer = dept_hospitals[dept_hospitals.geometry.within(buffer_geom)]
            hospital_count = len(hospitals_in_buffer)
            
            # Convert centroid back to WGS84 for visualization
            centroid_utm = pop_center['centroid_utm']
            centroid_point_utm = gpd.GeoDataFrame([1], geometry=[centroid_utm], crs=UTM_CRS)
            centroid_point_wgs84 = centroid_point_utm.to_crs(WGS84_CRS)
            centroid_wgs84 = centroid_point_wgs84.geometry.iloc[0]
            
            # Get population center info
            pop_center_info = {
                'department': dept,
                'pop_center_id': idx,
                'hospital_count': hospital_count,
                'centroid_lat': centroid_wgs84.y,
                'centroid_lon': centroid_wgs84.x,
                'buffer_geometry': buffer_geom,
                'hospitals_in_buffer': hospitals_in_buffer
            }
            
            # Add any name/identifier columns if available
            for col in pop_centers.columns:
                if col not in ['geometry', 'centroid', 'buffer'] and col != dept_col:
                    pop_center_info[f'pop_center_{col}'] = pop_center[col]
            
            results.append(pop_center_info)
        
        print(f"  ✅ Analyzed {len(dept_pop_centers)} population centers")
    
    return results

def identify_extreme_cases(results):
    """Identify isolation and concentration cases for each department."""
    print(f"\n🎯 IDENTIFYING EXTREME CASES")
    print("=" * 50)
    
    extreme_cases = {}
    
    for dept in TARGET_DEPARTMENTS:
        dept_results = [r for r in results if r['department'] == dept]
        
        if not dept_results:
            print(f"⚠️ No results for {dept}")
            continue
        
        # Find isolation (minimum hospitals) and concentration (maximum hospitals)
        hospital_counts = [r['hospital_count'] for r in dept_results]
        min_hospitals = min(hospital_counts)
        max_hospitals = max(hospital_counts)
        
        # Get cases (there might be ties)
        isolation_cases = [r for r in dept_results if r['hospital_count'] == min_hospitals]
        concentration_cases = [r for r in dept_results if r['hospital_count'] == max_hospitals]
        
        extreme_cases[dept] = {
            'isolation': isolation_cases[0],  # Take first if multiple
            'concentration': concentration_cases[0],  # Take first if multiple
            'isolation_count': len(isolation_cases),
            'concentration_count': len(concentration_cases),
            'min_hospitals': min_hospitals,
            'max_hospitals': max_hospitals,
            'total_pop_centers': len(dept_results),
            'avg_hospitals': np.mean(hospital_counts)
        }
        
        print(f"\n{dept} Department:")
        print(f"  🔴 Isolation: {min_hospitals} hospitals ({len(isolation_cases)} centers)")
        print(f"  🟢 Concentration: {max_hospitals} hospitals ({len(concentration_cases)} centers)")
        print(f"  📊 Average: {np.mean(hospital_counts):.1f} hospitals per center")
        print(f"  📈 Range: {min_hospitals} - {max_hospitals} hospitals")
    
    return extreme_cases

def create_folium_map(case_info, case_type, dept, buffer_distance_km=10):
    """Create interactive Folium map for a specific case."""
    print(f"Creating {case_type} map for {dept}...")
    
    # Get case data
    centroid_lat = case_info['centroid_lat']
    centroid_lon = case_info['centroid_lon']
    hospital_count = case_info['hospital_count']
    hospitals_in_buffer = case_info['hospitals_in_buffer']
    
    # Create base map centered on the population center
    m = folium.Map(
        location=[centroid_lat, centroid_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add population center marker
    folium.Marker(
        location=[centroid_lat, centroid_lon],
        popup=f"Population Center<br>{case_type.title()}<br>{hospital_count} hospitals within {buffer_distance_km}km",
        icon=folium.Icon(color='blue', icon='home')
    ).add_to(m)
    
    # Add 10km buffer circle
    folium.Circle(
        location=[centroid_lat, centroid_lon],
        radius=buffer_distance_km * 1000,  # Convert to meters
        popup=f"{buffer_distance_km}km buffer",
        color='blue',
        fillColor='lightblue',
        fillOpacity=0.2,
        weight=2
    ).add_to(m)
    
    # Add hospitals within buffer
    if len(hospitals_in_buffer) > 0:
        # Convert hospitals back to WGS84 for mapping
        hospitals_wgs84 = hospitals_in_buffer.to_crs(WGS84_CRS)
        
        for idx, hospital in hospitals_wgs84.iterrows():
            folium.Marker(
                location=[hospital.geometry.y, hospital.geometry.x],
                popup=f"Hospital: {hospital['Nombre del establecimiento']}<br>Type: {hospital['Clasificación']}",
                icon=folium.Icon(color='red', icon='plus')
            ).add_to(m)
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:20px"><b>{dept} - {case_type.title()}</b></h3>
    <p align="center">{hospital_count} hospitals within {buffer_distance_km}km radius</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    filename = f"{dept.lower()}_{case_type}_map.html"
    filepath = os.path.join(PROXIMITY_OUTPUT_DIR, filename)
    m.save(filepath)
    print(f"  ✅ Map saved: {filepath}")
    
    return filepath

def save_results_to_csv(results, extreme_cases):
    """Save analysis results to CSV files."""
    print(f"\n💾 SAVING RESULTS")
    print("=" * 50)
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'department': result['department'],
            'pop_center_id': result['pop_center_id'],
            'hospital_count': result['hospital_count'],
            'centroid_lat': result['centroid_lat'],
            'centroid_lon': result['centroid_lon']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save overall summary
    summary_file = os.path.join(PROXIMITY_OUTPUT_DIR, 'proximity_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary saved: {summary_file}")
    
    # Save department-specific files
    for dept in TARGET_DEPARTMENTS:
        dept_data = summary_df[summary_df['department'] == dept]
        if len(dept_data) > 0:
            dept_file = os.path.join(PROXIMITY_OUTPUT_DIR, f'proximity_analysis_{dept.lower()}.csv')
            dept_data.to_csv(dept_file, index=False)
            print(f"✅ {dept} data saved: {dept_file}")
    
    # Save extreme cases summary
    extreme_summary = []
    for dept, cases in extreme_cases.items():
        extreme_summary.append({
            'department': dept,
            'isolation_hospitals': cases['min_hospitals'],
            'concentration_hospitals': cases['max_hospitals'],
            'avg_hospitals': cases['avg_hospitals'],
            'total_pop_centers': cases['total_pop_centers']
        })
    
    extreme_df = pd.DataFrame(extreme_summary)
    extreme_file = os.path.join(PROXIMITY_OUTPUT_DIR, 'extreme_cases_summary.csv')
    extreme_df.to_csv(extreme_file, index=False)
    print(f"✅ Extreme cases saved: {extreme_file}")

def main():
    """Main function to run proximity analysis."""
    print("📍 PROXIMITY ANALYSIS - POPULATION CENTERS & HOSPITALS")
    print("=" * 70)
    print(f"Target departments: {', '.join(TARGET_DEPARTMENTS)}")
    print(f"Buffer distance: {BUFFER_DISTANCE_KM}km")
    print(f"Metric CRS: {UTM_CRS}")
    print(f"Visualization CRS: {WGS84_CRS}")
    
    try:
        # Load data
        hospitals, pop_centers = load_data()
        if hospitals is None or pop_centers is None:
            print("❌ Failed to load data")
            return
        
        # Prepare hospitals GeoDataFrame
        hospitals_gdf = prepare_hospitals_geodataframe(hospitals)
        
        # Prepare population centers
        pop_centers_filtered, dept_col = prepare_population_centers(pop_centers, TARGET_DEPARTMENTS)
        if pop_centers_filtered is None:
            print("❌ Failed to prepare population centers")
            return
        
        # Calculate proximity analysis
        results = calculate_proximity_analysis(
            hospitals_gdf, pop_centers_filtered, dept_col, BUFFER_DISTANCE_KM
        )
        
        if not results:
            print("❌ No results from proximity analysis")
            return
        
        # Identify extreme cases
        extreme_cases = identify_extreme_cases(results)
        
        # Create Folium maps for extreme cases
        print(f"\n🗺️ CREATING INTERACTIVE MAPS")
        print("=" * 50)
        
        for dept, cases in extreme_cases.items():
            # Create isolation map
            create_folium_map(cases['isolation'], 'isolation', dept, BUFFER_DISTANCE_KM)
            
            # Create concentration map
            create_folium_map(cases['concentration'], 'concentration', dept, BUFFER_DISTANCE_KM)
        
        # Save results
        save_results_to_csv(results, extreme_cases)
        
        # Final summary
        print(f"\n✅ PROXIMITY ANALYSIS COMPLETED")
        print("=" * 70)
        print(f"📁 All outputs saved to: {PROXIMITY_OUTPUT_DIR}")
        
        for dept, cases in extreme_cases.items():
            print(f"\n🏷️ {dept} DEPARTMENT:")
            print(f"  🔴 Most isolated: {cases['min_hospitals']} hospitals within {BUFFER_DISTANCE_KM}km")
            print(f"  🟢 Highest concentration: {cases['max_hospitals']} hospitals within {BUFFER_DISTANCE_KM}km")
            print(f"  📊 Average: {cases['avg_hospitals']:.1f} hospitals per population center")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
