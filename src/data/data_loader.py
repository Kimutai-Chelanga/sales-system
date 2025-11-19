import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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