#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Department-level Analysis of Operational Public Hospitals in Peru.

This script aggregates hospital data at the department level and provides:
1. Summary table sorted from highest to lowest
2. Bar chart visualization
3. Department-level choropleth map
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOSPITALS_CSV = os.path.join(BASE_DIR, '_data', 'operational_hospitals.csv')
DISTRICTS_SHP = os.path.join(BASE_DIR, '_data', 'DISTRITOS', 'DISTRITOS.shp')
OUTPUT_DIR = os.path.join(BASE_DIR, 'assets')
DEPARTMENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'department_analysis')

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEPARTMENT_OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load operational hospitals data and districts shapefile."""
    print("Loading operational hospitals data...")
    hospitals = pd.read_csv(HOSPITALS_CSV, encoding='utf-8', encoding_errors='replace')
    print(f"Loaded {len(hospitals)} operational public hospitals")
    
    print("\nLoading districts shapefile for department boundaries...")
    districts = gpd.read_file(DISTRICTS_SHP)
    print(f"Loaded {len(districts)} districts from shapefile")
    
    # Check available columns
    print(f"Hospital columns: {hospitals.columns.tolist()}")
    print(f"District columns: {districts.columns.tolist()}")
    
    return hospitals, districts

def create_department_aggregation(hospitals, districts):
    """Aggregate hospital data at the department level."""
    print("\n=== DEPARTMENT-LEVEL AGGREGATION ===")
    
    # Count hospitals per department from hospital data
    print("Aggregating hospitals by department...")
    hospital_dept_counts = hospitals.groupby('DEPARTAMENTO').size().reset_index(name='hospital_count')
    hospital_dept_counts = hospital_dept_counts.sort_values('hospital_count', ascending=False)
    
    print(f"Found hospitals in {len(hospital_dept_counts)} departments")
    
    # Get department geometries from districts shapefile
    print("Extracting department boundaries from districts...")
    # Group districts by department to create department polygons
    dept_geometries = districts.dissolve(by='DEPARTAMEN').reset_index()
    dept_geometries = dept_geometries[['DEPARTAMEN', 'geometry']]
    dept_geometries.columns = ['DEPARTAMENTO', 'geometry']
    
    print(f"Created geometries for {len(dept_geometries)} departments")
    
    # Standardize department names for merging
    print("Standardizing department names...")
    
    # Print unique department names from both sources for comparison
    print("Departments in hospital data:")
    print(sorted(hospital_dept_counts['DEPARTAMENTO'].unique()))
    print("\nDepartments in shapefile:")
    print(sorted(dept_geometries['DEPARTAMENTO'].unique()))
    
    # Merge hospital counts with department geometries
    print("\nMerging hospital counts with department geometries...")
    departments_with_counts = dept_geometries.merge(
        hospital_dept_counts, 
        on='DEPARTAMENTO', 
        how='left'
    )
    
    # Fill NaN values with 0 for departments without hospitals
    departments_with_counts['hospital_count'] = departments_with_counts['hospital_count'].fillna(0).astype(int)
    
    print(f"Final dataset: {len(departments_with_counts)} departments")
    
    return hospital_dept_counts, departments_with_counts

def create_summary_table(hospital_dept_counts):
    """Create and save summary table sorted from highest to lowest."""
    print("\n=== CREATING SUMMARY TABLE ===")
    
    # Add ranking
    summary_table = hospital_dept_counts.copy()
    summary_table['Rank'] = range(1, len(summary_table) + 1)
    summary_table = summary_table[['Rank', 'DEPARTAMENTO', 'hospital_count']]
    summary_table.columns = ['Rank', 'Department', 'Number of Hospitals']
    
    # Calculate statistics
    total_hospitals = summary_table['Number of Hospitals'].sum()
    avg_hospitals = summary_table['Number of Hospitals'].mean()
    median_hospitals = summary_table['Number of Hospitals'].median()
    
    # Identify highest and lowest
    highest_dept = summary_table.iloc[0]
    lowest_dept = summary_table.iloc[-1]
    
    print("DEPARTMENT-LEVEL HOSPITAL ANALYSIS")
    print("=" * 50)
    print(f"Total Hospitals: {total_hospitals}")
    print(f"Average per Department: {avg_hospitals:.1f}")
    print(f"Median per Department: {median_hospitals:.1f}")
    print(f"\nHighest: {highest_dept['Department']} ({highest_dept['Number of Hospitals']} hospitals)")
    print(f"Lowest: {lowest_dept['Department']} ({lowest_dept['Number of Hospitals']} hospitals)")
    print("\nComplete Ranking:")
    print(summary_table.to_string(index=False))
    
    # Save to CSV
    output_file = os.path.join(DEPARTMENT_OUTPUT_DIR, 'department_hospital_summary.csv')
    summary_table.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")
    
    return summary_table, highest_dept, lowest_dept

def create_bar_chart(hospital_dept_counts):
    """Create bar chart visualization of hospitals per department."""
    print("\n=== CREATING BAR CHART ===")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Create bar plot
    bars = ax.bar(
        range(len(hospital_dept_counts)), 
        hospital_dept_counts['hospital_count'],
        color=plt.cm.viridis(np.linspace(0, 1, len(hospital_dept_counts))),
        edgecolor='black',
        linewidth=0.5
    )
    
    # Customize the plot
    ax.set_xlabel('Departments', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Operational Public Hospitals', fontsize=14, fontweight='bold')
    ax.set_title('Operational Public Hospitals by Department in Peru', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels (rotate for better readability)
    ax.set_xticks(range(len(hospital_dept_counts)))
    ax.set_xticklabels(hospital_dept_counts['DEPARTAMENTO'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics text
    total_hospitals = hospital_dept_counts['hospital_count'].sum()
    avg_hospitals = hospital_dept_counts['hospital_count'].mean()
    
    stats_text = f'Total: {total_hospitals} hospitals | Average: {avg_hospitals:.1f} per department'
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    output_file = os.path.join(DEPARTMENT_OUTPUT_DIR, 'department_hospital_barchart.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to: {output_file}")

def create_choropleth_map(departments_with_counts):
    """Create department-level choropleth map."""
    print("\n=== CREATING CHOROPLETH MAP ===")
    
    # Convert to GeoDataFrame if not already
    if not isinstance(departments_with_counts, gpd.GeoDataFrame):
        departments_with_counts = gpd.GeoDataFrame(departments_with_counts)
    
    # Ensure proper CRS
    if departments_with_counts.crs is None:
        departments_with_counts = departments_with_counts.set_crs('EPSG:4326')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    
    # Create choropleth map
    departments_with_counts.plot(
        column='hospital_count',
        ax=ax,
        legend=True,
        cmap='YlOrRd',
        edgecolor='black',
        linewidth=1.0,
        legend_kwds={
            'label': "Number of Operational Public Hospitals",
            'orientation': "horizontal",
            'shrink': 0.6,
            'pad': 0.02
        }
    )
    
    # Add department labels
    for idx, row in departments_with_counts.iterrows():
        if row['hospital_count'] > 0:  # Only label departments with hospitals
            centroid = row.geometry.centroid
            plt.annotate(
                text=f"{row['DEPARTAMENTO']}\n({int(row['hospital_count'])})",
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
            )
    
    # Customize the map
    ax.set_title('Operational Public Hospitals by Department in Peru', 
                fontsize=18, fontweight='bold', pad=30)
    
    # Add statistics as subtitle
    total_hospitals = departments_with_counts['hospital_count'].sum()
    depts_with_hospitals = (departments_with_counts['hospital_count'] > 0).sum()
    max_hospitals = departments_with_counts['hospital_count'].max()
    
    stats_text = f'Total: {total_hospitals} hospitals across {depts_with_hospitals} departments (Max: {max_hospitals})'
    plt.suptitle(stats_text, fontsize=12, y=0.02)
    
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(DEPARTMENT_OUTPUT_DIR, 'department_hospital_choropleth.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Choropleth map saved to: {output_file}")

def create_comparison_analysis(hospital_dept_counts):
    """Create additional comparison analysis."""
    print("\n=== ADDITIONAL ANALYSIS ===")
    
    # Calculate percentiles
    q75 = hospital_dept_counts['hospital_count'].quantile(0.75)
    q50 = hospital_dept_counts['hospital_count'].quantile(0.50)
    q25 = hospital_dept_counts['hospital_count'].quantile(0.25)
    
    print(f"Hospital Distribution Quartiles:")
    print(f"75th percentile: {q75:.1f} hospitals")
    print(f"50th percentile (median): {q50:.1f} hospitals")
    print(f"25th percentile: {q25:.1f} hospitals")
    
    # Categorize departments
    high_coverage = hospital_dept_counts[hospital_dept_counts['hospital_count'] >= q75]
    medium_coverage = hospital_dept_counts[
        (hospital_dept_counts['hospital_count'] >= q25) & 
        (hospital_dept_counts['hospital_count'] < q75)
    ]
    low_coverage = hospital_dept_counts[hospital_dept_counts['hospital_count'] < q25]
    
    print(f"\nDepartment Categories:")
    print(f"High Coverage (≥{q75:.0f} hospitals): {len(high_coverage)} departments")
    for _, dept in high_coverage.iterrows():
        print(f"  - {dept['DEPARTAMENTO']}: {dept['hospital_count']} hospitals")
    
    print(f"\nMedium Coverage ({q25:.0f}-{q75:.0f} hospitals): {len(medium_coverage)} departments")
    for _, dept in medium_coverage.iterrows():
        print(f"  - {dept['DEPARTAMENTO']}: {dept['hospital_count']} hospitals")
    
    print(f"\nLow Coverage (<{q25:.0f} hospitals): {len(low_coverage)} departments")
    for _, dept in low_coverage.iterrows():
        print(f"  - {dept['DEPARTAMENTO']}: {dept['hospital_count']} hospitals")

def main():
    """Main function to run department-level analysis."""
    print("🏥 DEPARTMENT-LEVEL HOSPITAL ANALYSIS")
    print("=" * 60)
    
    try:
        # Load data
        hospitals, districts = load_data()
        
        # Create department aggregation
        hospital_dept_counts, departments_with_counts = create_department_aggregation(hospitals, districts)
        
        # Create summary table
        summary_table, highest_dept, lowest_dept = create_summary_table(hospital_dept_counts)
        
        # Create visualizations
        create_bar_chart(hospital_dept_counts)
        create_choropleth_map(departments_with_counts)
        
        # Additional analysis
        create_comparison_analysis(hospital_dept_counts)
        
        print(f"\n✅ Department-level analysis completed successfully!")
        print(f"📁 All outputs saved to: {DEPARTMENT_OUTPUT_DIR}")
        
        print(f"\n📊 KEY FINDINGS:")
        print(f"🥇 Highest: {highest_dept['Department']} ({highest_dept['Number of Hospitals']} hospitals)")
        print(f"🥉 Lowest: {lowest_dept['Department']} ({lowest_dept['Number of Hospitals']} hospitals)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
