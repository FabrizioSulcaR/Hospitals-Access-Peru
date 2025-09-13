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
    input_file = os.path.join('_data', 'legacy/IPRESS.csv')
    output_file = os.path.join('_data', 'operational_hospitals.csv')
    
    # Read the data
    print("Reading IPRESS dataset...")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # Print initial shape
    print(f"Initial dataset shape: {df.shape}")
    
    # Filter for operational hospitals
    print("Filtering for operational hospitals...")
    # Find the correct column name for 'Condición'
    condition_col = None
    for col in df.columns:
        if 'Condici' in col:
            condition_col = col
            print(f"Found condition column: '{condition_col}'")
            break
    
    if condition_col is None:
        print("Could not find condition column. Available columns:", df.columns.tolist())
        return None
        
    df_operational = df[df[condition_col].str.strip() == 'EN FUNCIONAMIENTO']
    print(f"After operational filter: {df_operational.shape}")
    
    # Filter for valid coordinates
    print("Filtering for valid coordinates...")
    # Create a copy to avoid SettingWithCopyWarning
    df_operational = df_operational.copy()
    
    # Convert to numeric, coerce errors to NaN
    df_operational['NORTE'] = pd.to_numeric(df_operational['NORTE'], errors='coerce')
    df_operational['ESTE'] = pd.to_numeric(df_operational['ESTE'], errors='coerce')
    
    # Keep only rows where both coordinates are valid (not NaN)
    df_valid = df_operational.dropna(subset=['NORTE', 'ESTE'])
    print(f"After valid coordinates filter: {df_valid.shape}")

    df_valid.columns = [
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

    # Save to CSV
    print(f"Saving filtered dataset to {output_file}...")
    df_valid.to_csv(output_file, index=False, encoding='utf-8')
    
    print("Done!")
    
    # Return the dataframe for further use if needed
    return df_valid

if __name__ == "__main__":
    main()
