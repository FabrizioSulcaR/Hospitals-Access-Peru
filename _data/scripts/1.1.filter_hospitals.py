#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script filters the IPRESS dataset to:
1. Keep only operational hospitals (with "EN FUNCIONAMIENTO" status)
2. Keep only records with valid coordinates (latitude and longitude)
3. Save the filtered dataframe to a file in the _data directory
"""

import pandas as pd
import os
import numpy as np

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, '_data', 'raw', 'IPRESS.csv')
    output_file = os.path.join(base_dir, '_data', 'operational_hospitals.csv')
    
    # Read the data
    print("Reading IPRESS dataset...")
    df = pd.read_csv(input_file, encoding='utf-8')
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
    # Print initial shape
    print(f"Initial dataset shape: {df.shape}")
    
    # Filter for operational hospitals
    print("Filtering for operational hospitals...")
    condition_col = 'Condición'
    df_operational = df[df[condition_col].str.strip() == 'EN FUNCIONAMIENTO']
    print(f"After operational status filter: {df_operational.shape}")

    # Check unique values in the classification column to handle potential encoding issues
    print("\nChecking hospital classifications...")
    print("Unique values in 'Clasificación':")
    print(df_operational['Clasificación'].unique())
    
    # Filter for hospital classifications using partial string matching to be safe with encoding
    print("\nFiltering for hospital classifications...")
    df_operational_hospitals = df_operational[
        df_operational['Clasificación'].str.contains('HOSPITALES O CLINICAS DE ATENCION GENERAL', case=False, na=False) | 
        df_operational['Clasificación'].str.contains('HOSPITALES O CLINICAS DE ATENCION ESPECIALIZADA', case=False, na=False)
    ]
    print(f"After hospital classification filter: {df_operational_hospitals.shape}")

    # Filter for public hospitals only
    print("\nFiltering for public hospitals...")
    df_operational_hospitals = df_operational_hospitals[
        (df_operational_hospitals['Institucion'] != 'PRIVADO')
    ]
    print(f"After public hospitals filter: {df_operational_hospitals.shape}")
    
    # Filter for valid coordinates
    print("Filtering for valid coordinates...")
    # Create a copy to avoid SettingWithCopyWarning
    df_operational_hospitals = df_operational_hospitals.copy()
    
    # Convert to numeric, coerce errors to NaN
    df_operational_hospitals['NORTE'] = pd.to_numeric(df_operational_hospitals['NORTE'], errors='coerce')
    df_operational_hospitals['ESTE'] = pd.to_numeric(df_operational_hospitals['ESTE'], errors='coerce')
    
    # Keep only rows where both coordinates are valid (not NaN)
    df_valid = df_operational_hospitals.dropna(subset=['NORTE', 'ESTE'])
    print(f"After valid coordinates filter: {df_valid.shape}")

    # Save to CSV
    print(f"Saving filtered dataset to {output_file}...")
    df_valid.to_csv(output_file, index=False, encoding='utf-8')
    
    print("Done!")
    
    # Return the dataframe for further use if needed
    return df_valid

if __name__ == "__main__":
    main()
