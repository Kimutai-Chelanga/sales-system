import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sales_data(n_records=10000):
    """Generate synthetic sales data for Kenya market"""
    
    # Kenyan cities and regions
    cities = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
    products = ['Tea', 'Cooking Oil', 'Flour', 'Sugar', 'Rice', 'Milk']
    
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    
    data = []
    for i in range(n_records):
        date = start_date + timedelta(days=random.randint(0, 730))
        data.append({
            'transaction_id': f'TXN_{i:06d}',
            'date': date,
            'product': random.choice(products),
            'quantity': np.random.poisson(50) + 10,
            'unit_price': round(np.random.uniform(50, 500), 2),
            'city': random.choice(cities),
            'salesperson_id': f'SP_{random.randint(1, 20):03d}',
            'customer_id': f'CUST_{random.randint(1, 500):04d}',
            'day_of_week': date.strftime('%A'),
            'month': date.month,
            'is_holiday': random.random() < 0.05
        })
    
    df = pd.DataFrame(data)
    df['revenue'] = df['quantity'] * df['unit_price']
    return df

if __name__ == "__main__":
    # Generate the data
    print("Generating sales data...")
    df = generate_sales_data(10000)
    
    # Create data directory if it doesn't exist
    data_dir = "../../data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(data_dir, "sales_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    
    # Also save to parquet (more efficient)
    parquet_path = os.path.join(data_dir, "sales_data.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved to {parquet_path}")
    
    # Print summary
    print(f"\nDataset summary:")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total revenue: KES {df['revenue'].sum():,.2f}")
    print(f"\nFirst few rows:")
    print(df.head())