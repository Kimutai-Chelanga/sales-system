# 🚀 Getting Started - Sales Forecasting Project
# Complete guide for Solutech Data Scientist interview preparation

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     SALES FORECASTING & ML PROJECT FOR SOLUTECH             ║
║                                                              ║
║  This project demonstrates:                                  ║
║  ✓ Data Analysis & EDA                                       ║
║  ✓ Machine Learning (scikit-learn, XGBoost, LightGBM)      ║
║  ✓ MLOps with MLflow                                        ║
║  ✓ Advanced Visualizations (Plotly)                         ║
║  ✓ Production-ready code                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# STEP 1: VERIFY ENVIRONMENT SETUP
# ============================================================================

print("\n" + "="*60)
print("STEP 1: VERIFYING ENVIRONMENT")
print("="*60)

import sys
import os

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check if we're in the right directory
current_dir = os.getcwd()
print(f"✓ Current directory: {current_dir}")

# Check critical libraries
required_libraries = [
    'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm',
    'mlflow', 'plotly', 'seaborn', 'matplotlib', 'scipy'
]

missing_libraries = []
for lib in required_libraries:
    try:
        __import__(lib)
        print(f"✓ {lib} installed")
    except ImportError:
        print(f"✗ {lib} NOT FOUND")
        missing_libraries.append(lib)

if missing_libraries:
    print(f"\n⚠️  Missing libraries: {', '.join(missing_libraries)}")
    print("Run: pip install -r requirements.txt")
else:
    print("\n✓ All required libraries installed!")

# ============================================================================
# STEP 2: PROJECT STRUCTURE
# ============================================================================

print("\n" + "="*60)
print("STEP 2: PROJECT STRUCTURE")
print("="*60)

print("""
sales-system/
│
├── data/                    # Generated datasets
│   ├── sales_data.csv
│   └── sales_data.parquet
│
├── notebooks/              # Jupyter notebooks (YOU ARE HERE!)
│   ├── 00_getting_started.ipynb        ← Current notebook
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_machine_learning_models.ipynb
│   ├── 03_mlops_mlflow_tracking.ipynb
│   └── 04_advanced_visualizations.ipynb
│
├── src/                    # Source code
│   ├── data/
│   │   └── data_loader.py  # Data generation
│   ├── models/             # Model training
│   └── api/                # FastAPI deployment
│
├── models/                 # Saved models
├── mlops/                  # MLflow tracking
└── dashboards/             # Plotly dashboards
""")

# ============================================================================
# STEP 3: GENERATE DATA
# ============================================================================

print("\n" + "="*60)
print("STEP 3: GENERATE SAMPLE DATA")
print("="*60)

# Check if data exists
data_path = '../data/sales_data.csv'
if os.path.exists(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"✓ Data already exists: {len(df):,} records")
    print(f"  Location: {data_path}")
else:
    print("⚠️  Data not found. Generating now...")
    # Run data generation
    sys.path.append('../src/data')
    from data_loader import generate_sales_data
    
    df = generate_sales_data(10000)
    os.makedirs('../data', exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"✓ Generated {len(df):,} records")
    print(f"✓ Saved to: {data_path}")

# Quick data preview
import pandas as pd
df = pd.read_csv(data_path)

print("\nData Preview:")
print(df.head())

print(f"\nDataset Summary:")
print(f"  Records: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Products: {df['product'].nunique()}")
print(f"  Cities: {df['city'].nunique()}")
print(f"  Total Revenue: KES {df['revenue'].sum():,.2f}")

# ============================================================================
# STEP 4: QUICK DATA VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("STEP 4: QUICK VISUALIZATION")
print("="*60)

try:
    import plotly.express as px
    
    # Revenue by product
    product_revenue = df.groupby('product')['revenue'].sum().sort_values(ascending=True)
    
    fig = px.bar(
        x=product_revenue.values,
        y=product_revenue.index,
        orientation='h',
        title='Total Revenue by Product',
        labels={'x': 'Revenue (KES)', 'y': 'Product'},
        color=product_revenue.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500)
    fig.show()
    
    print("✓ Visualization created!")
    
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")

# ============================================================================
# STEP 5: NAVIGATION GUIDE
# ============================================================================

print("\n" + "="*60)
print("STEP 5: NEXT STEPS")
print("="*60)

print("""
🎯 RECOMMENDED WORKFLOW:

1️⃣  Exploratory Data Analysis (EDA)
    → Open: 01_exploratory_data_analysis.ipynb
    → Learn: Data quality, distributions, patterns, trends
    → Skills: pandas, plotly, statistical analysis

2️⃣  Machine Learning Models
    → Open: 02_machine_learning_models.ipynb
    → Learn: Feature engineering, model training, evaluation
    → Skills: scikit-learn, XGBoost, LightGBM

3️⃣  MLOps & Experiment Tracking
    → Open: 03_mlops_mlflow_tracking.ipynb
    → Learn: MLflow, model versioning, deployment
    → Skills: MLOps, model registry, production workflows

4️⃣  Advanced Visualizations
    → Open: 04_advanced_visualizations.ipynb
    → Learn: Interactive dashboards, business intelligence
    → Skills: Plotly, data storytelling

💡 INTERVIEW TIPS FOR SOLUTECH:
   
   ✓ Focus on field sales management context
   ✓ Emphasize business impact of models
   ✓ Discuss scalability and production deployment
   ✓ Show understanding of Kenyan market (Ketepa, Pwani Oil)
   ✓ Highlight MLOps and CI/CD experience
   ✓ Demonstrate data-driven decision making

🔗 SOLUTECH PRODUCTS TO UNDERSTAND:
   • Field Sales Management Tool
   • Retail Execution Platform
   • Logistics & Order Management
   • Data-driven insights for FMCG companies

📚 KEY TECHNOLOGIES TO MENTION:
   ✓ Python (pandas, scikit-learn, PyTorch/TensorFlow)
   ✓ Google Cloud Platform (BigQuery, AI Platform)
   ✓ Business Intelligence (Power BI, Looker, Tableau)
   ✓ MLOps (MLflow, model monitoring, CI/CD)
   ✓ GenAI & Computer Vision (if applicable)
""")

# ============================================================================
# STEP 6: SYSTEM CHECK
# ============================================================================

print("\n" + "="*60)
print("SYSTEM CHECK COMPLETE")
print("="*60)

import pandas as pd
import numpy as np

# Final verification
checks = {
    "✓ Data loaded": os.path.exists(data_path),
    "✓ Environment ready": len(missing_libraries) == 0,
    "✓ Notebooks available": os.path.exists('01_exploratory_data_analysis.ipynb'),
}

for check, status in checks.items():
    if status:
        print(f"✓ {check}")
    else:
        print(f"✗ {check}")

print("\n" + "="*60)
print("🚀 READY TO START!")
print("="*60)
print("\nYou're all set! Proceed to notebook 01 for EDA.")
print("Good luck with your Solutech interview! 💪")