# 01 - Exploratory Data Analysis (EDA)
# This notebook demonstrates core data science skills for Solutech interview

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully")

# ============================================================================
# 1. LOAD AND INSPECT DATA
# ============================================================================

# Load the sales data
df = pd.read_csv('../data/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n{'='*60}")
print("DATASET OVERVIEW")
print(f"{'='*60}")
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Display first few rows
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)
df.head(10)

# ============================================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Check for missing values
missing = df.isnull().sum()
print("\nMissing Values:")
print(missing[missing > 0] if missing.sum() > 0 else "✓ No missing values")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Data types
print("\nData Types:")
print(df.dtypes)

# Statistical summary
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
df.describe()

# ============================================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("UNIVARIATE ANALYSIS")
print("="*60)

# Revenue distribution
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Revenue Distribution', 'Quantity Distribution',
                   'Unit Price Distribution', 'Revenue by Product')
)

# Revenue histogram
fig.add_trace(
    go.Histogram(x=df['revenue'], name='Revenue', nbinsx=50),
    row=1, col=1
)

# Quantity histogram
fig.add_trace(
    go.Histogram(x=df['quantity'], name='Quantity', nbinsx=50),
    row=1, col=2
)

# Unit price histogram
fig.add_trace(
    go.Histogram(x=df['unit_price'], name='Unit Price', nbinsx=50),
    row=2, col=1
)

# Revenue by product
product_revenue = df.groupby('product')['revenue'].sum().sort_values(ascending=True)
fig.add_trace(
    go.Bar(x=product_revenue.values, y=product_revenue.index, 
           orientation='h', name='Product Revenue'),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Distribution Analysis")
fig.show()

# ============================================================================
# 4. GEOGRAPHIC ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("GEOGRAPHIC ANALYSIS")
print("="*60)

# Sales by city
city_analysis = df.groupby('city').agg({
    'revenue': 'sum',
    'transaction_id': 'count',
    'quantity': 'sum'
}).round(2)
city_analysis.columns = ['Total Revenue', 'Transactions', 'Total Quantity']
city_analysis = city_analysis.sort_values('Total Revenue', ascending=False)

print("\nSales Performance by City:")
print(city_analysis)

# Visualization
fig = px.bar(city_analysis.reset_index(), 
             x='city', y='Total Revenue',
             title='Revenue by City',
             color='Total Revenue',
             color_continuous_scale='Viridis')
fig.show()

# ============================================================================
# 5. TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("TEMPORAL ANALYSIS")
print("="*60)

# Add time features
df['year'] = df['date'].dt.year
df['month_name'] = df['date'].dt.strftime('%B')
df['quarter'] = df['date'].dt.quarter

# Daily revenue trend
daily_revenue = df.groupby('date')['revenue'].sum().reset_index()

fig = px.line(daily_revenue, x='date', y='revenue',
              title='Daily Revenue Trend',
              labels={'revenue': 'Revenue (KES)', 'date': 'Date'})
fig.show()

# Monthly analysis
monthly = df.groupby(['year', 'month'])['revenue'].sum().reset_index()
monthly['year_month'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)

fig = px.bar(monthly, x='year_month', y='revenue',
             title='Monthly Revenue Trend',
             labels={'revenue': 'Revenue (KES)', 'year_month': 'Month'})
fig.show()

# Day of week analysis
dow_revenue = df.groupby('day_of_week')['revenue'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

print("\nAverage Revenue by Day of Week:")
print(dow_revenue.round(2))

fig = px.bar(x=dow_revenue.index, y=dow_revenue.values,
             title='Average Revenue by Day of Week',
             labels={'x': 'Day', 'y': 'Average Revenue (KES)'})
fig.show()

# ============================================================================
# 6. PRODUCT ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("PRODUCT ANALYSIS")
print("="*60)

product_analysis = df.groupby('product').agg({
    'revenue': ['sum', 'mean'],
    'quantity': 'sum',
    'transaction_id': 'count',
    'unit_price': 'mean'
}).round(2)

product_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Total Qty', 'Transactions', 'Avg Price']
product_analysis = product_analysis.sort_values('Total Revenue', ascending=False)

print("\nProduct Performance:")
print(product_analysis)

# Product mix visualization
fig = px.pie(df, names='product', values='revenue',
             title='Revenue Share by Product')
fig.show()

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Select numeric columns
numeric_cols = ['quantity', 'unit_price', 'revenue', 'month']
correlation = df[numeric_cols].corr()

print("\nCorrelation Matrix:")
print(correlation.round(3))

# Heatmap
fig = px.imshow(correlation, 
                text_auto='.2f',
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r')
fig.show()

# ============================================================================
# 8. KEY INSIGHTS & BUSINESS METRICS
# ============================================================================

print("\n" + "="*60)
print("KEY BUSINESS METRICS")
print("="*60)

total_revenue = df['revenue'].sum()
avg_transaction = df['revenue'].mean()
total_transactions = len(df)
avg_quantity = df['quantity'].mean()

print(f"\n📊 Business Performance Summary:")
print(f"   Total Revenue: KES {total_revenue:,.2f}")
print(f"   Total Transactions: {total_transactions:,}")
print(f"   Average Transaction Value: KES {avg_transaction:,.2f}")
print(f"   Average Quantity per Transaction: {avg_quantity:.1f} units")

# Top performing segments
print(f"\n🏆 Top Performers:")
print(f"   Best City: {city_analysis.index[0]} (KES {city_analysis.iloc[0]['Total Revenue']:,.2f})")
print(f"   Best Product: {product_analysis.index[0]} (KES {product_analysis.iloc[0]['Total Revenue']:,.2f})")

# Growth analysis
yearly_revenue = df.groupby('year')['revenue'].sum()
if len(yearly_revenue) > 1:
    growth_rate = ((yearly_revenue.iloc[-1] - yearly_revenue.iloc[0]) / yearly_revenue.iloc[0]) * 100
    print(f"\n📈 Year-over-Year Growth: {growth_rate:.1f}%")

print("\n" + "="*60)
print("✓ EDA COMPLETE - Ready for ML Modeling!")
print("="*60)