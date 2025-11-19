# 🚀 Jupyter Lab Startup Guide - Solutech Interview Prep

Complete guide to start Jupyter and work through the data science notebooks for your Solutech interview.

---

## 📋 Prerequisites

Make sure you're in your project directory with the virtual environment activated:

```bash
cd ~/sales-system
source venv/bin/activate
```

---

## 🎯 Quick Start

### 1. Start Jupyter Lab

```bash
jupyter lab
```

This will:
- Start the Jupyter Lab server
- Automatically open your browser to `http://localhost:8888`
- Display all notebooks in the `notebooks/` folder

### 2. Alternative: Start Jupyter Notebook (Classic Interface)

```bash
jupyter notebook
```

---

## 📚 Notebook Sequence & Learning Path

### **Notebook 00: Getting Started** ⭐ START HERE
**File:** `00_getting_started.ipynb`

**Purpose:** Environment check and project overview

**Topics:**
- Verify all libraries installed
- Generate sales data
- Understand project structure
- Quick visualization demo

**Time:** 10 minutes

---

### **Notebook 01: Exploratory Data Analysis**
**File:** `01_exploratory_data_analysis.ipynb`

**Purpose:** Master data analysis and visualization

**Topics:**
- Data quality assessment
- Statistical analysis
- Temporal patterns
- Geographic analysis
- Product performance
- Interactive Plotly visualizations

**Skills Demonstrated:**
- pandas data manipulation
- Statistical methods
- Data visualization (matplotlib, seaborn, plotly)
- Business insights extraction

**Time:** 30-45 minutes

**Key for Solutech:** Shows ability to understand field sales data and extract actionable insights

---

### **Notebook 02: Machine Learning Models**
**File:** `02_machine_learning_models.ipynb`

**Purpose:** Build and compare ML models

**Topics:**
- Feature engineering
- Model training (Linear Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning
- Model evaluation & comparison
- Feature importance analysis
- Model persistence

**Skills Demonstrated:**
- scikit-learn
- XGBoost
- LightGBM
- Model evaluation metrics
- Production model selection

**Time:** 45-60 minutes

**Key for Solutech:** Demonstrates 3-5 years ML experience with modern frameworks

---

### **Notebook 03: MLOps with MLflow**
**File:** `03_mlops_mlflow_tracking.ipynb`

**Purpose:** Production ML workflow management

**Topics:**
- Experiment tracking
- Model versioning
- Parameter logging
- Model registry
- Production deployment simulation
- Model lifecycle management

**Skills Demonstrated:**
- MLflow
- Experiment management
- Model registry
- Production workflows
- CI/CD readiness

**Time:** 30-45 minutes

**Key for Solutech:** Shows MLOps practices for scalable solutions

---

### **Notebook 04: Advanced Visualizations**
**File:** `04_advanced_visualizations.ipynb`

**Purpose:** Create business intelligence dashboards

**Topics:**
- Executive dashboards
- Interactive visualizations
- Time series analysis
- Cohort analysis
- Salesperson performance tracking
- Revenue funnel analysis

**Skills Demonstrated:**
- Plotly advanced features
- BI dashboard design
- Data storytelling
- Stakeholder communication

**Time:** 30-45 minutes

**Key for Solutech:** Aligns with Power BI/Looker/Tableau requirements

---

### **Notebook 05: GenAI & Computer Vision**
**File:** `05_genai_computer_vision.ipynb`

**Purpose:** AI-powered retail execution

**Topics:**
- Automated insight generation
- Shelf audit simulation
- Product detection
- Stock-out monitoring
- Store segmentation
- Real-time alert systems

**Skills Demonstrated:**
- GenAI applications
- Computer Vision (OpenCV)
- Automated reporting
- Business process optimization

**Time:** 45-60 minutes

**Key for Solutech:** Directly relevant to retail execution platform

---

### **Notebook 06: Google Cloud Deployment**
**File:** `06_google_cloud_deployment.ipynb`

**Purpose:** Production deployment on GCP

**Topics:**
- BigQuery data warehousing
- Cloud Storage integration
- Vertex AI deployment
- Monitoring & observability
- CI/CD pipelines
- Cost optimization

**Skills Demonstrated:**
- Google Cloud Platform
- BigQuery
- Vertex AI
- Production architecture
- DevOps practices

**Time:** 45-60 minutes

**Key for Solutech:** Shows GCP experience (required skill)

---

## 🎨 Jupyter Lab Tips

### Navigation
- **File Browser:** Left sidebar shows all notebooks
- **Table of Contents:** Right sidebar (click icon) shows notebook structure
- **Multiple Tabs:** Open multiple notebooks side-by-side

### Keyboard Shortcuts (Essential)
```
Shift + Enter    → Run cell and move to next
Ctrl + Enter     → Run cell and stay
Alt + Enter      → Run cell and insert below
A                → Insert cell above (command mode)
B                → Insert cell below (command mode)
D + D            → Delete cell (command mode)
M                → Convert to markdown (command mode)
Y                → Convert to code (command mode)
```

### Running Cells
1. **Run Individual Cell:** Click cell, press `Shift + Enter`
2. **Run All Cells:** Menu → Run → Run All Cells
3. **Restart & Run All:** Menu → Kernel → Restart Kernel and Run All

### Viewing Outputs
- Charts appear inline automatically
- Use `display()` for multiple outputs
- Scroll long outputs
- Clear outputs: Cell → All Output → Clear

---

## 🔧 Troubleshooting

### Jupyter Won't Start
```bash
# Reinstall jupyter
pip install --upgrade jupyterlab

# Or use notebook interface
jupyter notebook
```

### Port Already in Use
```bash
# Use different port
jupyter lab --port 8889
```

### Kernel Crashes
```bash
# Restart kernel in Jupyter interface
# Or restart Jupyter server
```

### Missing Libraries
```bash
# Install missing packages
pip install [package-name]

# Or reinstall all requirements
pip install -r requirements.txt
```

### Can't See Plots
```bash
# Make sure matplotlib backend is set
%matplotlib inline
```

---

## 📊 Data Generation

Before running analysis notebooks, ensure data exists:

```bash
# Navigate to data loader
cd ~/sales-system/src/data

# Generate data
python3 data_loader.py

# Verify data created
ls -lh ../../data/
```

You should see:
- `sales_data.csv` (main dataset)
- `sales_data.parquet` (optimized format)

---

## 🎯 Interview Preparation Strategy

### Day Before Interview
1. **Run Notebook 00** - Verify everything works
2. **Run Notebooks 01-02** - Core data science skills
3. **Run Notebook 03** - MLOps capabilities
4. **Review Notebook 06** - GCP architecture

### During Interview
- **Have Notebooks Open** - Show your work
- **Focus on Notebook 04** - For visualization questions
- **Reference Notebook 05** - For GenAI/CV discussions
- **Use Notebook 06** - For architecture/deployment questions

### Key Points to Emphasize
1. ✅ **3-5 years ML experience** - Demonstrated through diverse models
2. ✅ **Python proficiency** - pandas, scikit-learn, TensorFlow/PyTorch
3. ✅ **GenAI & Computer Vision** - Retail execution applications
4. ✅ **GCP experience** - BigQuery, Vertex AI
5. ✅ **MLOps practices** - Experiment tracking, deployment
6. ✅ **Business intelligence** - Plotly dashboards
7. ✅ **Field sales context** - Kenya market, FMCG products

---

## 🌐 MLflow UI (Optional)

To view experiment tracking:

```bash
# In a new terminal
cd ~/sales-system/mlops
mlflow ui

# Open browser to: http://localhost:5000
```

---

## 💡 Pro Tips

1. **Add Code Comments** - Demonstrate your thinking
2. **Use Markdown Cells** - Explain your approach
3. **Show Outputs** - Don't clear important results
4. **Save Regularly** - Jupyter auto-saves, but manual saves are safer
5. **Export as PDF** - File → Export → PDF for offline reference

---

## 📞 Quick Commands Reference

```bash
# Start Jupyter Lab
jupyter lab

# Start in specific directory
jupyter lab --notebook-dir=~/sales-system/notebooks

# List running servers
jupyter server list

# Stop Jupyter
Ctrl + C (in terminal)

# Generate data
cd ~/sales-system/src/data && python3 data_loader.py

# Run all notebooks programmatically
jupyter nbconvert --execute --to notebook notebooks/*.ipynb
```

---

## 🎓 Learning Order for Maximum Impact

**For Technical Interview (2-3 hours prep):**
1. Notebook 00 (Setup)
2. Notebook 01 (EDA - shows analysis skills)
3. Notebook 02 (ML - shows modeling skills)
4. Notebook 03 (MLOps - shows production skills)

**For Deep Dive (Full day prep):**
1. All above
2. Notebook 04 (BI visualizations)
3. Notebook 05 (GenAI/CV applications)
4. Notebook 06 (GCP architecture)

**For Presentation/Demo:**
- Keep Notebook 04 open (impressive visualizations)
- Have Notebook 06 ready (architecture discussion)
- Reference Notebook 05 (innovation with AI)

---

## ✅ Pre-Interview Checklist

- [ ] All notebooks run without errors
- [ ] Data generated and accessible
- [ ] Understand each model's purpose
- [ ] Can explain visualizations
- [ ] Know GCP architecture
- [ ] Familiar with Solutech products
- [ ] Prepared questions about their tech stack

---

## 🏆 Success!

You're now ready for the Solutech interview with:
- ✅ Complete data science portfolio
- ✅ Production-ready ML skills
- ✅ Cloud deployment knowledge
- ✅ Business intelligence capabilities
- ✅ GenAI & Computer Vision applications

**Good luck with your interview at Solutech! 🚀**

For questions or issues, refer to notebook-specific documentation or Python error messages.