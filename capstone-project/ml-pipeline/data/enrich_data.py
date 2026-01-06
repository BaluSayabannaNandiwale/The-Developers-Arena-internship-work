"""
Data Enrichment Script
Adds missing required fields to the house_prices.csv dataset
"""
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path

def enrich_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Enrich the dataset with missing required fields.
    
    Args:
        input_path: Path to original house_prices.csv
        output_path: Path to save enriched dataset
        
    Returns:
        Enriched DataFrame
    """
    # Load original data
    df = pd.read_csv(input_path)
    
    # Map existing columns to required schema
    df = df.rename(columns={
        'Area': 'area_sqft',
        'Bedrooms': 'bedrooms',
        'Bathrooms': 'bathrooms',
        'Age': 'property_age',
        'Price': 'price_inr'
    })
    
    # Generate missing fields with realistic distributions
    np.random.seed(42)
    n = len(df)
    
    # Floor number (1-20, weighted towards lower floors)
    # Normalize probabilities to sum to 1
    floor_probs = [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04,
                   0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
    floor_probs = np.array(floor_probs) / np.sum(floor_probs)
    df['floor'] = np.random.choice(
        range(1, 21),
        size=n,
        p=floor_probs
    )
    
    # Total floors (correlated with property type and location)
    df['total_floors'] = df.apply(
        lambda row: np.random.randint(5, 25) if row['Location'] == 'City Center'
        else (np.random.randint(2, 8) if row['Property_Type'] == 'Villa'
              else np.random.randint(5, 15)),
        axis=1
    )
    
    # City (based on location patterns)
    city_mapping = {
        'City Center': 'Bangalore',
        'Suburb': np.random.choice(['Bangalore', 'Mumbai', 'Delhi'], p=[0.5, 0.3, 0.2]),
        'Rural': np.random.choice(['Bangalore', 'Pune', 'Hyderabad'], p=[0.4, 0.35, 0.25])
    }
    df['city'] = df['Location'].apply(
        lambda x: city_mapping[x] if isinstance(city_mapping[x], str)
        else city_mapping[x]
    )
    # Fix for Suburb and Rural
    df.loc[df['Location'] == 'Suburb', 'city'] = np.random.choice(
        ['Bangalore', 'Mumbai', 'Delhi'],
        size=df[df['Location'] == 'Suburb'].shape[0],
        p=[0.5, 0.3, 0.2]
    )
    df.loc[df['Location'] == 'Rural', 'city'] = np.random.choice(
        ['Bangalore', 'Pune', 'Hyderabad'],
        size=df[df['Location'] == 'Rural'].shape[0],
        p=[0.4, 0.35, 0.25]
    )
    
    # Facing direction
    df['facing'] = np.random.choice(
        ['East', 'West', 'North', 'South'],
        size=n,
        p=[0.3, 0.3, 0.2, 0.2]
    )
    
    # Furnishing status
    df['furnishing'] = np.random.choice(
        ['Furnished', 'Semi-Furnished', 'Unfurnished'],
        size=n,
        p=[0.3, 0.5, 0.2]
    )
    
    # Parking slots (correlated with bedrooms and property type)
    df['parking'] = df.apply(
        lambda row: min(row['bedrooms'], 2) if row['Property_Type'] == 'Apartment'
        else min(row['bedrooms'] + 1, 3),
        axis=1
    )
    
    # Amenities score (0-1, correlated with location and property type)
    df['amenities_score'] = df.apply(
        lambda row: np.random.beta(8, 2) if row['Location'] == 'City Center'
        else (np.random.beta(6, 4) if row['Location'] == 'Suburb'
              else np.random.beta(4, 6)),
        axis=1
    )
    
    # Distance from city center (km, correlated with location)
    df['distance_city_center_km'] = df.apply(
        lambda row: np.random.uniform(0, 5) if row['Location'] == 'City Center'
        else (np.random.uniform(5, 20) if row['Location'] == 'Suburb'
              else np.random.uniform(15, 50)),
        axis=1
    )
    
    # Rename Property_Type to property_type and Location to location
    df = df.rename(columns={
        'Property_Type': 'property_type',
        'Location': 'location'
    })
    
    # Reorder columns to match required schema
    column_order = [
        'area_sqft', 'bedrooms', 'bathrooms', 'floor', 'total_floors',
        'property_age', 'location', 'city', 'property_type', 'facing',
        'furnishing', 'parking', 'amenities_score', 'distance_city_center_km',
        'price_inr'
    ]
    
    # Keep Property_ID for reference but don't include in final model
    df_final = df[column_order].copy()
    
    # Save enriched dataset
    df_final.to_csv(output_path, index=False)
    print(f"[OK] Enriched dataset saved to {output_path}")
    print(f"[INFO] Shape: {df_final.shape}")
    print(f"[INFO] Columns: {list(df_final.columns)}")
    
    return df_final

if __name__ == "__main__":
    import os
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / 'data' / 'house_prices.csv'
    output_path = project_root / 'data' / 'house_prices_enriched.csv'
    enrich_dataset(str(input_path), str(output_path))

