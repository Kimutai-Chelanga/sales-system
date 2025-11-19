# 04 - Advanced Visualizations with Plotly
# State-of-the-art interactive dashboards for business intelligence

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

df = pd.read_csv('../data/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M').astype(str)
df['year'] = df['date'].dt.year

print(f"Data loaded: {len(df):,} records")

# ============================================================================
# 2. EXECUTIVE DASHBOARD OVERVIEW
# ============================================================================

print("\n" + "="*60)
print("EXECUTIVE DASHBOARD")
print("="*60)

# Calculate KPIs
total_revenue = df['revenue'].sum()
total_transactions = len(df)
avg_transaction = df['revenue'].mean()
total_quantity = df['quantity'].sum()

# Create dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Revenue Trend', 'Top Products by Revenue',
                   'City Performance', 'Sales by Day of Week'),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# 1. Revenue trend
daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
fig.add_trace(
    go.Scatter(x=daily_revenue['date'], y=daily_revenue['revenue'],
              mode='lines', name='Daily Revenue',
              line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

# 2. Top products
product_revenue = df.groupby('product')['revenue'].sum().sort_values(ascending=True)
fig.add_trace(
    go.Bar(x=product_revenue.values, y=product_revenue.index,
          orientation='h', name='Product Revenue',
          marker=dict(color='#ff7f0e')),
    row=1, col=2
)

# 3. City performance
city_revenue = df.groupby('city')['revenue'].sum().sort_values(ascending=True)
fig.add_trace(
    go.Bar(x=city_revenue.values, y=city_revenue.index,
          orientation='h', name='City Revenue',
          marker=dict(color='#2ca02c')),
    row=2, col=1
)

# 4. Day of week
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_revenue = df.groupby('day_of_week')['revenue'].sum().reindex(dow_order)
fig.add_trace(
    go.Bar(x=dow_revenue.index, y=dow_revenue.values,
          name='DOW Revenue',
          marker=dict(color='#d62728')),
    row=2, col=2
)

fig.update_layout(
    height=800,
    showlegend=False,
    title_text=f"📊 Sales Performance Dashboard | Total Revenue: KES {total_revenue:,.0f}"
)
fig.show()

# ============================================================================
# 3. GEOGRAPHIC HEATMAP
# ============================================================================

print("\n" + "="*60)
print("GEOGRAPHIC ANALYSIS")
print("="*60)

# City performance matrix
city_product = df.groupby(['city', 'product'])['revenue'].sum().reset_index()
city_product_pivot = city_product.pivot(index='city', columns='product', values='revenue')

fig = px.imshow(city_product_pivot,
                labels=dict(x="Product", y="City", color="Revenue"),
                title="Revenue Heatmap: City vs Product",
                color_continuous_scale='Viridis',
                text_auto='.0f')
fig.update_layout(height=500)
fig.show()

# ============================================================================
# 4. TIME SERIES DECOMPOSITION
# ============================================================================

print("\n" + "="*60)
print("TIME SERIES ANALYSIS")
print("="*60)

# Monthly aggregation
monthly = df.groupby('year_month').agg({
    'revenue': 'sum',
    'quantity': 'sum',
    'transaction_id': 'count'
}).reset_index()
monthly.columns = ['year_month', 'revenue', 'quantity', 'transactions']

# Multi-line chart
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Monthly Revenue', 'Monthly Quantity Sold', 'Monthly Transactions'),
    vertical_spacing=0.1
)

fig.add_trace(
    go.Scatter(x=monthly['year_month'], y=monthly['revenue'],
              mode='lines+markers', name='Revenue',
              line=dict(color='#1f77b4', width=3)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=monthly['year_month'], y=monthly['quantity'],
              mode='lines+markers', name='Quantity',
              line=dict(color='#ff7f0e', width=3)),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=monthly['year_month'], y=monthly['transactions'],
              mode='lines+markers', name='Transactions',
              line=dict(color='#2ca02c', width=3)),
    row=3, col=1
)

fig.update_layout(height=900, showlegend=False,
                 title_text="📈 Time Series Analysis")
fig.show()

# ============================================================================
# 5. SALESPERSON PERFORMANCE
# ============================================================================

print("\n" + "="*60)
print("SALESPERSON PERFORMANCE")
print("="*60)

salesperson_stats = df.groupby('salesperson_id').agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': 'sum'
}).round(2)

salesperson_stats.columns = ['Total Revenue', 'Avg Revenue', 'Transactions', 'Total Quantity']
salesperson_stats = salesperson_stats.sort_values('Total Revenue', ascending=False).head(10)

# Create performance chart
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 10 Salespeople by Revenue', 'Transactions vs Avg Revenue')
)

# Bar chart
fig.add_trace(
    go.Bar(x=salesperson_stats.index, y=salesperson_stats['Total Revenue'],
          name='Revenue', marker=dict(color='#1f77b4')),
    row=1, col=1
)

# Scatter plot
fig.add_trace(
    go.Scatter(x=salesperson_stats['Transactions'], 
              y=salesperson_stats['Avg Revenue'],
              mode='markers+text',
              text=salesperson_stats.index,
              textposition='top center',
              marker=dict(size=15, color='#ff7f0e'),
              name='Performance'),
    row=1, col=2
)

fig.update_layout(height=500, title_text="👥 Salesperson Performance Analysis")
fig.show()

# ============================================================================
# 6. PRODUCT PORTFOLIO ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("PRODUCT PORTFOLIO ANALYSIS")
print("="*60)

# Calculate product metrics
product_metrics = df.groupby('product').agg({
    'revenue': 'sum',
    'quantity': 'sum',
    'transaction_id': 'count',
    'unit_price': 'mean'
}).round(2)

product_metrics.columns = ['Total Revenue', 'Total Quantity', 'Transactions', 'Avg Price']
product_metrics['Revenue Share'] = (product_metrics['Total Revenue'] / product_metrics['Total Revenue'].sum() * 100).round(1)

# Bubble chart
fig = px.scatter(product_metrics.reset_index(),
                x='Transactions', y='Avg Price',
                size='Total Revenue', color='product',
                hover_data=['Total Quantity', 'Revenue Share'],
                title='Product Portfolio: Transaction Volume vs Price',
                labels={'Transactions': 'Number of Transactions',
                       'Avg Price': 'Average Unit Price (KES)'})
fig.update_layout(height=600)
fig.show()

# ============================================================================
# 7. COHORT ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("CUSTOMER COHORT ANALYSIS")
print("="*60)

# Get first purchase date for each customer
customer_cohort = df.groupby('customer_id')['date'].min().reset_index()
customer_cohort.columns = ['customer_id', 'first_purchase']
customer_cohort['cohort'] = customer_cohort['first_purchase'].dt.to_period('M')

# Merge back
df_cohort = df.merge(customer_cohort[['customer_id', 'cohort']], on='customer_id')
df_cohort['order_period'] = df_cohort['date'].dt.to_period('M')
df_cohort['period_number'] = (df_cohort['order_period'] - df_cohort['cohort']).apply(lambda x: x.n)

# Cohort size
cohort_data = df_cohort.groupby(['cohort', 'period_number'])['customer_id'].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index='cohort', columns='period_number', values='customer_id')

# Retention rates
cohort_sizes = cohort_pivot[0]
retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100

fig = px.imshow(retention,
               labels=dict(x="Months Since First Purchase", y="Cohort", color="Retention %"),
               title="Customer Retention Cohort Analysis",
               color_continuous_scale='RdYlGn',
               text_auto='.1f')
fig.update_layout(height=600)
fig.show()

# ============================================================================
# 8. REVENUE DISTRIBUTION & OUTLIERS
# ============================================================================

print("\n" + "="*60)
print("REVENUE DISTRIBUTION & OUTLIERS")
print("="*60)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Revenue Distribution', 'Box Plot by Product')
)

# Histogram
fig.add_trace(
    go.Histogram(x=df['revenue'], nbinsx=100, name='Revenue'),
    row=1, col=1
)

# Box plot by product
for product in df['product'].unique():
    product_data = df[df['product'] == product]['revenue']
    fig.add_trace(
        go.Box(y=product_data, name=product),
        row=1, col=2
    )

fig.update_layout(height=500, showlegend=False,
                 title_text="📊 Revenue Distribution Analysis")
fig.show()

# Identify outliers
Q1 = df['revenue'].quantile(0.25)
Q3 = df['revenue'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['revenue'] < Q1 - 1.5*IQR) | (df['revenue'] > Q3 + 1.5*IQR)]

print(f"\nOutlier Detection:")
print(f"   Total outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
print(f"   Outlier revenue range: KES {outliers['revenue'].min():,.2f} - {outliers['revenue'].max():,.2f}")

# ============================================================================
# 9. FUNNEL ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("SALES FUNNEL")
print("="*60)

# Create revenue brackets
df['revenue_bracket'] = pd.cut(df['revenue'], 
                               bins=[0, 5000, 10000, 20000, 50000, np.inf],
                               labels=['< 5K', '5K-10K', '10K-20K', '20K-50K', '> 50K'])

funnel_data = df['revenue_bracket'].value_counts().sort_index()

fig = go.Figure(go.Funnel(
    y=funnel_data.index.astype(str),
    x=funnel_data.values,
    textinfo="value+percent initial",
    marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
))

fig.update_layout(
    title="Sales Revenue Funnel (Transaction Value Brackets)",
    height=600
)
fig.show()

# ============================================================================
# 10. SUMMARY METRICS CARDS
# ============================================================================

print("\n" + "="*60)
print("KEY PERFORMANCE INDICATORS")
print("="*60)

# Calculate metrics
total_revenue = df['revenue'].sum()
avg_transaction = df['revenue'].mean()
total_transactions = len(df)
total_customers = df['customer_id'].nunique()
total_salespeople = df['salesperson_id'].nunique()
best_product = df.groupby('product')['revenue'].sum().idxmax()
best_city = df.groupby('city')['revenue'].sum().idxmax()

print(f"""
📊 BUSINESS PERFORMANCE SUMMARY
{'-'*50}
💰 Total Revenue:        KES {total_revenue:,.2f}
📈 Avg Transaction:      KES {avg_transaction:,.2f}
🛒 Total Transactions:   {total_transactions:,}
👥 Unique Customers:     {total_customers:,}
🤝 Active Salespeople:   {total_salespeople}
🏆 Top Product:          {best_product}
🌍 Top City:             {best_city}
{'-'*50}
""")

print("\n" + "="*60)
print("✓ VISUALIZATION SUITE COMPLETE!")
print("="*60)