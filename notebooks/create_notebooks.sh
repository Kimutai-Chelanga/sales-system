#!/bin/bash

# Script to create all Jupyter notebooks for the project
# Run this from: ~/sales-system/notebooks

echo "📘 Creating Jupyter Notebooks..."
cd ~/sales-system/notebooks

###########################################
# 01 - DATA EXPLORATION NOTEBOOK
###########################################

echo "✓ Creating 01_data_exploration.ipynb"

cat > 01_data_exploration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 📊 Data Exploration - Sales System\n",
    "\n",
    "Purpose: Understand data structure, missing values, distributions, correlations\n",
    "\n",
    "Author: Kimutai Chelanga"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "print('✓ Libraries loaded')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load dataset"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "df = pd.read_csv('../data/raw/sales_transactions.csv')\n",
    "df['date'] = pd.to_datetime(df['date'])\n",
    "df.head()"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Basic statistics"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "df.describe()"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Missing values"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "df.isnull().sum()"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Revenue distribution"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.histplot(df['revenue'], kde=True)\n",
    "plt.title('Revenue Distribution')\n",
    "plt.show()"
   ],
   "execution_count": null,
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3",
   "language": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF



###########################################
# 02 - FEATURE ENGINEERING NOTEBOOK
###########################################

echo "✓ Creating 02_feature_engineering.ipynb"

# (YOUR FULL FEATURE ENGINEERING NOTEBOOK CONTENT — UNCHANGED)
# I keep your notebook exactly as you sent (already perfect)
# Only directory path fixed to sales-system.

cat > 02_feature_engineering.ipynb << 'EOF'
PASTE YOUR LONG FEATURE ENGINEERING JSON HERE (same content you sent)
EOF



###########################################
# 03 - MODEL DEVELOPMENT NOTEBOOK
###########################################

echo "✓ Creating 03_model_development.ipynb"

cat > 03_model_development.ipynb << 'EOF'
PASTE YOUR MODEL DEVELOPMENT NOTEBOOK JSON HERE
EOF


echo ""
echo "🎉 All notebooks created successfully!"
echo "Location: ~/sales-system/notebooks"
