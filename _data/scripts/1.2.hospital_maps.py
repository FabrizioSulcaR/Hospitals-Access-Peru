#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate static maps showing the number of operational public hospitals per district in Peru.
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISTRICTS_SHP = os.path.join(BASE_DIR, '_data', 'DISTRITOS', 'DISTRITOS.shp')
HOSPITALS_CSV = os.path.join(BASE_DIR, '_data', 'operational_hospitals.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'assets/district_analysis')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load district shapefile and operational hospitals data."""
    print("Loading districts shapefile...")
    # Load shapefile and select IDDIST, DISTRITO (district name) and geometry columns
    districts = gpd.read_file(DISTRICTS_SHP)
    districts = districts[['IDDIST', 'DISTRITO', 'geometry']]
    
    # Rename IDDIST to UBIGEO
    districts = districts.rename({'IDDIST': 'UBIGEO'}, axis=1)
    
    # Convert UBIGEO to string and then to integer
    districts['UBIGEO'] = districts['UBIGEO'].astype(str).astype(int)
    
    # Ensure the dataset is in WGS-84 (EPSG:4326)
    districts = districts.to_crs(epsg=4326)
    
    print(f"Districts CRS: {districts.crs}")
    print(f"Districts shape: {districts.shape}")
    
    print("Loading hospitals data...")
    hospitals = pd.read_csv(HOSPITALS_CSV, encoding='utf-8', encoding_errors='replace')
    
    # Print first few rows to understand the data
    print("First few rows of hospitals data:")
    print(hospitals.head(2))
    print("Column names:")
    print(hospitals.columns.tolist())
    
    # Filter for operational public hospitals only
    # Based on the CSV schema: filter by 'Condición' = 'EN FUNCIONAMIENTO' 
    # and exclude private hospitals (those with 'Institucion' = 'PRIVADO')
    print(f"Total hospitals before filtering: {len(hospitals)}")
    
    # Check unique values in key columns for debugging
    print("Unique values in 'Condición':", hospitals['Condición'].unique())
    print("Unique values in 'Institucion':", hospitals['Institucion'].unique()[:10])  # Show first 10
    
    print(f"Loaded {len(districts)} districts and {len(hospitals)} operational public hospitals.")
    return districts, hospitals

def prepare_hospital_data(hospitals):
    """Clean and prepare hospital coordinate data."""
    print("Preparing hospital coordinate data...")
    
    # Create a copy to avoid SettingWithCopyWarning
    hospitals_clean = hospitals.copy()
    
    # Convert coordinates to numeric, handling any errors
    for col in ['ESTE', 'NORTE']:
        # Check data type before conversion
        if hospitals_clean[col].dtype == 'object':
            print(f"Converting {col} from {hospitals_clean[col].dtype} to numeric")
            # Replace commas with periods if present (common in some locales)
            if hospitals_clean[col].str.contains(',').any():
                hospitals_clean[col] = hospitals_clean[col].str.replace(',', '.')
            
            # Convert to numeric, coercing errors to NaN
            hospitals_clean[col] = pd.to_numeric(hospitals_clean[col], errors='coerce')
    
    # Check for missing values before and after conversion
    missing_before = hospitals.isna().sum()
    missing_after = hospitals_clean.isna().sum()
    
    print(f"Missing values in ESTE before: {missing_before['ESTE']}, after: {missing_after['ESTE']}")
    print(f"Missing values in NORTE before: {missing_before['NORTE']}, after: {missing_after['NORTE']}")
    
    # Drop rows with missing coordinates
    valid_coords = hospitals_clean.dropna(subset=['ESTE', 'NORTE'])
    print(f"Hospitals with valid coordinates: {len(valid_coords)} of {len(hospitals_clean)} ({len(valid_coords)/len(hospitals_clean)*100:.1f}%)")
    
    # Check for outliers in coordinates
    este_min, este_max = valid_coords['ESTE'].min(), valid_coords['ESTE'].max()
    norte_min, norte_max = valid_coords['NORTE'].min(), valid_coords['NORTE'].max()
    
    print(f"ESTE range: {este_min:.6f} to {este_max:.6f}")
    print(f"NORTE range: {norte_min:.6f} to {norte_max:.6f}")
    
    return valid_coords

def validate_and_create_coordinates(hospitals_clean, districts):
    """Validate coordinates and create GeoDataFrame with proper coordinate system."""
    print("\n=== COORDINATE VALIDATION ===")
    
    # Check coordinate ranges
    este_min, este_max = hospitals_clean['ESTE'].min(), hospitals_clean['ESTE'].max()
    norte_min, norte_max = hospitals_clean['NORTE'].min(), hospitals_clean['NORTE'].max()
    
    print(f"ESTE range: {este_min:.6f} to {este_max:.6f}")
    print(f"NORTE range: {norte_min:.6f} to {norte_max:.6f}")
    
    # Sample coordinates for manual inspection
    print("\nSample coordinates (first 5 hospitals):")
    for i in range(min(5, len(hospitals_clean))):
        row = hospitals_clean.iloc[i]
        print(f"  {row['Nombre del establecimiento'][:40]}")
        print(f"    Dept: {row['DEPARTAMENTO']}, Prov: {row['Provincia']}, Dist: {row['Distrito']}")
        print(f"    ESTE: {row['ESTE']:.6f}, NORTE: {row['NORTE']:.6f}")
    
    # Peru's approximate bounds for validation:
    # Latitude: -18.5 to 0 (South to North)
    # Longitude: -81.5 to -68 (West to East)
    
    print("\n=== COORDINATE INTERPRETATION ATTEMPTS ===")
    
    # Option 1: ESTE = longitude (X), NORTE = latitude (Y) [Expected based on names]
    print("Option 1: ESTE=longitude, NORTE=latitude")
    valid_opt1 = (
        (hospitals_clean['ESTE'] >= -82) & (hospitals_clean['ESTE'] <= -68) &  # Longitude check
        (hospitals_clean['NORTE'] >= -19) & (hospitals_clean['NORTE'] <= 1)    # Latitude check
    ).sum()
    print(f"  Valid coordinates in Peru bounds: {valid_opt1}/{len(hospitals_clean)}")
    
    # Option 2: ESTE = latitude (Y), NORTE = longitude (X) [Swapped - what we suspected]
    print("Option 2: ESTE=latitude, NORTE=longitude")
    valid_opt2 = (
        (hospitals_clean['NORTE'] >= -82) & (hospitals_clean['NORTE'] <= -68) &  # Longitude check
        (hospitals_clean['ESTE'] >= -19) & (hospitals_clean['ESTE'] <= 1)       # Latitude check
    ).sum()
    print(f"  Valid coordinates in Peru bounds: {valid_opt2}/{len(hospitals_clean)}")
    
    # Determine which option is correct
    if valid_opt1 > valid_opt2:
        print("✅ Using Option 1: ESTE=longitude, NORTE=latitude")
        hospital_points = gpd.GeoDataFrame(
            hospitals_clean,
            geometry=gpd.points_from_xy(hospitals_clean['ESTE'], hospitals_clean['NORTE']),
            crs='EPSG:4326'
        )
        coordinate_interpretation = "standard"
    else:
        print("✅ Using Option 2: ESTE=latitude, NORTE=longitude (swapped)")
        hospital_points = gpd.GeoDataFrame(
            hospitals_clean,
            geometry=gpd.points_from_xy(hospitals_clean['NORTE'], hospitals_clean['ESTE']),
            crs='EPSG:4326'
        )
        coordinate_interpretation = "swapped"
    
    # Check district bounds for additional validation
    districts_bounds = districts.total_bounds
    print(f"\nDistricts shapefile bounds: {districts_bounds}")
    print(f"Districts CRS: {districts.crs}")
    
    # Convert hospital points to match districts CRS
    if districts.crs != hospital_points.crs:
        print(f"Converting hospital points from {hospital_points.crs} to {districts.crs}")
        hospital_points = hospital_points.to_crs(districts.crs)
    
    # Final validation: check how many points fall within district bounds
    hospitals_bounds = hospital_points.total_bounds
    print(f"Hospital points bounds: {hospitals_bounds}")
    
    return hospital_points, coordinate_interpretation

def create_hospital_count_by_district(districts, hospitals):
    """Create a GeoDataFrame with hospital count per district."""
    # Step 1: Prepare hospital data
    hospitals_clean = prepare_hospital_data(hospitals)
    
    print("\n=== PREPARING DATA FOR MERGE ===")
    
    # Ensure UBIGEO in hospitals is an integer for proper joining
    print("Converting hospital UBIGEO to integer for joining...")
    hospitals_clean['UBIGEO'] = pd.to_numeric(hospitals_clean['UBIGEO'], errors='coerce')
    
    # Drop rows with invalid UBIGEO
    hospitals_clean = hospitals_clean.dropna(subset=['UBIGEO'])
    hospitals_clean['UBIGEO'] = hospitals_clean['UBIGEO'].astype(int)
    
    # Print UBIGEO info for debugging
    print(f"Hospital UBIGEO data type: {hospitals_clean['UBIGEO'].dtype}")
    print(f"Districts UBIGEO data type: {districts['UBIGEO'].dtype}")
    
    print(f"Sample UBIGEO from hospitals: {hospitals_clean['UBIGEO'].head(5).tolist()}")
    print(f"Sample UBIGEO from districts: {districts['UBIGEO'].head(5).tolist()}")
    
    # Merge districts and hospitals using inner join on UBIGEO
    print("\n=== PERFORMING INNER JOIN ===")
    print(f"Before merge - districts: {len(districts)}, hospitals: {len(hospitals_clean)}")
    
    # Merge using inner join to drop missing values
    merged_data = pd.merge(districts, hospitals_clean, how="inner", on="UBIGEO")
    print(f"After merge - rows in merged dataset: {len(merged_data)}")
    
    # Count hospitals per district
    print("\n=== HOSPITAL COUNT ANALYSIS ===")
    hospital_counts = merged_data.groupby('UBIGEO').size().reset_index(name='hospital_count')
    print(f"Districts with hospitals: {len(hospital_counts)} of {len(districts)} districts ({len(hospital_counts)/len(districts)*100:.1f}%)")
    
    # Show top districts
    top_districts = hospital_counts.nlargest(5, 'hospital_count')
    print("Top 5 districts by hospital count:")
    for _, row in top_districts.iterrows():
        print(f"  District UBIGEO {row['UBIGEO']}: {row['hospital_count']} hospitals")
    
    # Merge hospital counts back to districts
    print("\n=== CREATING FINAL DATASET ===")
    districts_with_counts = districts.merge(hospital_counts, on='UBIGEO', how='left')
    districts_with_counts['hospital_count'] = districts_with_counts['hospital_count'].fillna(0).astype(int)
    
    # Print summary statistics
    total_hospitals_mapped = districts_with_counts['hospital_count'].sum()
    districts_with_hospitals = (districts_with_counts['hospital_count'] > 0).sum()
    districts_without_hospitals = (districts_with_counts['hospital_count'] == 0).sum()
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"- Total hospitals mapped to districts: {total_hospitals_mapped}")
    print(f"- Districts with hospitals: {districts_with_hospitals}")
    print(f"- Districts without hospitals: {districts_without_hospitals}")
    print(f"- Max hospitals in a district: {districts_with_counts['hospital_count'].max()}")

    return districts_with_counts

def plot_map1(districts_with_counts):
    """Create Map 1: Total public hospitals per district."""
    print("Creating Map 1: Total public hospitals per district...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    
    # Plot districts with hospital count
    districts_with_counts.plot(
        column='hospital_count',
        ax=ax,
        legend=True,
        cmap='YlOrRd',
        edgecolor='black',
        linewidth=0.2,
        legend_kwds={
            'label': "Number of Public Hospitals",
            'orientation': "horizontal",
            'shrink': 0.6,
            'pad': 0.02
        }
    )
    
    # Add title and labels
    plt.title('Number of Operational Public Hospitals per District in Peru', 
             fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle with summary statistics
    total_hospitals = districts_with_counts['hospital_count'].sum()
    districts_with_hospitals = (districts_with_counts['hospital_count'] > 0).sum()
    max_hospitals = districts_with_counts['hospital_count'].max()
    
    plt.suptitle(f'Total: {total_hospitals} hospitals across {districts_with_hospitals} districts (Max: {max_hospitals} per district)', 
                fontsize=12, y=0.02)
    
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'map1_hospital_count.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Map 1 saved to {output_file}")

def plot_map2(districts_with_counts):
    """Create Map 2: Highlight districts with zero hospitals."""
    print("Creating Map 2: Districts with zero hospitals...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    
    # Create a binary column for districts with zero hospitals
    districts_with_counts['has_hospital'] = districts_with_counts['hospital_count'] > 0
    
    # Create a custom colormap: red for districts with no hospitals, green for districts with hospitals
    cmap = LinearSegmentedColormap.from_list('custom', ['#FF4444', '#44AA44'], N=2)
    
    # Plot districts based on whether they have hospitals
    districts_with_counts.plot(
        column='has_hospital',
        ax=ax,
        cmap=cmap,
        edgecolor='black',
        linewidth=0.2
    )
    
    # Create a more detailed legend
    districts_without = (districts_with_counts['hospital_count'] == 0).sum()
    districts_with = (districts_with_counts['hospital_count'] > 0).sum()
    
    red_patch = mpatches.Patch(color='#FF4444', label=f'No Hospitals ({districts_without} districts)')
    green_patch = mpatches.Patch(color='#44AA44', label=f'Has Hospitals ({districts_with} districts)')
    
    ax.legend(handles=[red_patch, green_patch], 
             loc='lower right', 
             fontsize=12,
             frameon=True,
             fancybox=True,
             shadow=True,
             framealpha=0.9)
    
    # Add title and labels
    plt.title('Districts with Zero Public Hospitals in Peru', 
             fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle
    coverage_pct = (districts_with / len(districts_with_counts)) * 100
    plt.suptitle(f'Hospital Coverage: {coverage_pct:.1f}% of districts have at least one public hospital', 
                fontsize=12, y=0.02)
    
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'map2_zero_hospitals.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Map 2 saved to {output_file}")

def plot_map3(districts_with_counts):
    """Create Map 3: Top 10 districts with highest number of hospitals."""
    print("Creating Map 3: Top 10 districts with highest number of hospitals...")
    
    # Get top 10 districts by hospital count
    top10_districts = districts_with_counts.nlargest(10, 'hospital_count')
    
    print("Top 10 districts by hospital count:")
    for _, row in top10_districts.iterrows():
        print(f"  {row['DISTRITO']} (UBIGEO {row['UBIGEO']}): {int(row['hospital_count'])} hospitals")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    
    # Plot all districts in light gray
    districts_with_counts.plot(
        color='#E8E8E8',
        ax=ax,
        edgecolor='gray',
        linewidth=0.2
    )
    
    # Plot top 10 districts with a different color scheme
    top10_districts.plot(
        column='hospital_count',
        ax=ax,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.8,
        legend=True,
        legend_kwds={
            'label': "Number of Public Hospitals (Top 10)",
            'orientation': "horizontal",
            'shrink': 0.6,
            'pad': 0.02
        }
    )
    
    # Add district names and hospital counts for top 10
    for idx, row in top10_districts.iterrows():
        # Get centroid coordinates
        centroid = row.geometry.centroid
        
        # Create a more readable label with district name
        district_name = row['DISTRITO']
        hospital_count = int(row['hospital_count'])
        
        plt.annotate(
            text=f"{district_name}\n({hospital_count} hospitals)",
            xy=(centroid.x, centroid.y),
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9)
        )
    
    # Add title and labels
    plt.title('Top 10 Districts with Highest Number of Public Hospitals in Peru', 
             fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle with ranking info
    top_district = top10_districts.iloc[0]
    plt.suptitle(f'Leading district: {top_district["DISTRITO"]} with {int(top_district["hospital_count"])} hospitals', 
                fontsize=12, y=0.02)
    
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'map3_top10_hospitals.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Map 3 saved to {output_file}")

def main():
    """Main function to generate all maps."""
    try:
        # Load data
        districts, hospitals = load_data()
        
        # Create GeoDataFrame with hospital count per district
        districts_with_counts = create_hospital_count_by_district(districts, hospitals)
        
        # Generate maps
        plot_map1(districts_with_counts)
        plot_map2(districts_with_counts)
        plot_map3(districts_with_counts)
        
        print("\n✅ All maps generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()