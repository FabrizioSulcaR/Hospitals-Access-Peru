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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTRICTS_SHP = os.path.join(BASE_DIR, '_data', 'shape_file', 'DISTRITOS.shp')
HOSPITALS_CSV = os.path.join(BASE_DIR, '_data', 'operational_hospitals.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'assets')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load district shapefile and operational hospitals data."""
    print("Loading districts shapefile...")
    districts = gpd.read_file(DISTRICTS_SHP)
    
    print("Loading hospitals data...")
    hospitals = pd.read_csv(HOSPITALS_CSV, encoding='utf-8', encoding_errors='replace')
    
    # Print first few rows to understand the data
    print("First few rows of hospitals data:")
    print(hospitals.head(2))
    
    # Filter for operational public hospitals only
    # Based on the CSV columns, we need to filter by 'Estado' column with value 'ACTIVADO'
    # and exclude private hospitals (those with 'InstituciÃ³n' as 'PRIVADO')
    hospitals = hospitals[
        (hospitals['Condición'] == 'EN FUNCIONAMIENTO') & 
        (hospitals['Institucion'] != 'PRIVADO')
    ]
    
    print(f"Loaded {len(districts)} districts and {len(hospitals)} operational public hospitals.")
    return districts, hospitals

def create_hospital_count_by_district(districts, hospitals):
    """Create a GeoDataFrame with hospital count per district."""
    # Create points from hospital coordinates
    # In the CSV, the coordinates are in columns 'ESTE' and 'NORTE'
    # Convert to numeric, handling any errors
    hospitals['ESTE'] = pd.to_numeric(hospitals['ESTE'], errors='coerce')
    hospitals['NORTE'] = pd.to_numeric(hospitals['NORTE'], errors='coerce')
    
    # Drop rows with missing coordinates
    hospitals = hospitals.dropna(subset=['ESTE', 'NORTE'])
    
    hospital_points = gpd.GeoDataFrame(
        hospitals,
        geometry=gpd.points_from_xy(hospitals['ESTE'], hospitals['NORTE']),
        crs=districts.crs
    )
    print("########################################################")
    print("This is hospital_points:")
    print(len(list(set(hospital_points['UBIGEO']))))
    print("########################################################")
    exit()

    # Count hospitals per district using spatial join
    joined = gpd.sjoin(hospital_points, districts, how="inner", predicate="within")
    hospital_counts = joined.groupby('UBIGEO').size().reset_index(name='hospital_count')

    # Merge counts back to districts
    districts_with_counts = districts.merge(hospital_counts, on='UBIGEO', how='left')
    districts_with_counts['hospital_count'] = districts_with_counts['hospital_count'].fillna(0).astype(int)
    
    return districts_with_counts

def plot_map1(districts_with_counts):
    """Create Map 1: Total public hospitals per district."""
    print("Creating Map 1: Total public hospitals per district...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    
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
            'orientation': "horizontal"
        }
    )
    
    # Add title and labels
    plt.title('Number of Operational Public Hospitals per District in Peru', fontsize=16)
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
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    
    # Create a binary column for districts with zero hospitals
    districts_with_counts['has_hospital'] = districts_with_counts['hospital_count'] > 0
    
    # Create a custom colormap: red for districts with no hospitals, green for districts with hospitals
    cmap = LinearSegmentedColormap.from_list('custom', ['#FF9999', '#66B266'], N=2)
    
    # Plot districts based on whether they have hospitals
    districts_with_counts.plot(
        column='has_hospital',
        ax=ax,
        cmap=cmap,
        edgecolor='black',
        linewidth=0.2
    )
    
    # Create legend
    red_patch = mpatches.Patch(color='#FF9999', label='No Hospitals')
    green_patch = mpatches.Patch(color='#66B266', label='Has Hospitals')
    ax.legend(handles=[red_patch, green_patch], loc='lower right')
    
    # Add title and labels
    plt.title('Districts with Zero Public Hospitals in Peru', fontsize=16)
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
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    
    # Plot all districts in light gray
    districts_with_counts.plot(
        color='lightgray',
        ax=ax,
        edgecolor='black',
        linewidth=0.2
    )
    
    # Plot top 10 districts with a different color scheme
    top10_districts.plot(
        column='hospital_count',
        ax=ax,
        cmap='plasma',
        edgecolor='black',
        linewidth=0.5,
        legend=True,
        legend_kwds={
            'label': "Number of Public Hospitals (Top 10)",
            'orientation': "horizontal"
        }
    )
    
    # Add district names for top 10
    for idx, row in top10_districts.iterrows():
        plt.annotate(
            text=f"{row['NOMBDIST']}: {int(row['hospital_count'])}",
            xy=(row.geometry.centroid.x, row.geometry.centroid.y),
            ha='center',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
    
    # Add title and labels
    plt.title('Top 10 Districts with Highest Number of Public Hospitals in Peru', fontsize=16)
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'map3_top10_hospitals.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Map 3 saved to {output_file}")

def main():
    """Main function to generate all maps."""
    # Load data
    districts, hospitals = load_data()
    
    # Create GeoDataFrame with hospital count per district
    districts_with_counts = create_hospital_count_by_district(districts, hospitals)
    
    # Generate maps
    plot_map1(districts_with_counts)
    plot_map2(districts_with_counts)
    plot_map3(districts_with_counts)
    
    print("All maps generated successfully!")

if __name__ == "__main__":
    main()
