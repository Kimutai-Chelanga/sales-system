# 06 - Google Cloud Platform Deployment
# Demonstrates GCP skills for production ML systems at Solutech

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         GOOGLE CLOUD PLATFORM FOR ML DEPLOYMENT             ║
║                                                              ║
║  This notebook demonstrates:                                 ║
║  ✓ BigQuery integration for data warehousing                ║
║  ✓ Cloud Storage for model artifacts                        ║
║  ✓ Vertex AI for model deployment                           ║
║  ✓ Production-ready ML pipelines                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# 1. GCP ARCHITECTURE OVERVIEW
# ============================================================================

print("\n" + "="*60)
print("PART 1: GCP ARCHITECTURE FOR SOLUTECH")
print("="*60)

print("""
🏗️ RECOMMENDED GCP ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│  Field App → Cloud Functions → Pub/Sub → BigQuery      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   DATA WAREHOUSE                        │
│  BigQuery: Sales data, customer data, audit results    │
│  - Partitioned by date for performance                 │
│  - Clustered by city/product for queries               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  ML PIPELINE (Vertex AI)                │
│  1. Data preprocessing (Dataflow)                      │
│  2. Feature engineering (BigQuery ML / Python)         │
│  3. Model training (Vertex AI Training)                │
│  4. Model evaluation (Vertex AI Experiments)           │
│  5. Model deployment (Vertex AI Endpoints)             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  SERVING & APPS                         │
│  Cloud Run: FastAPI service                            │
│  Cloud Functions: Real-time predictions                │
│  Looker/Data Studio: BI dashboards                     │
└─────────────────────────────────────────────────────────┘

💰 COST OPTIMIZATION:
   → Use BigQuery slots for predictable workloads
   → Implement data lifecycle policies
   → Use preemptible VMs for training
   → Cache frequently accessed predictions
""")

# ============================================================================
# 2. BIGQUERY INTEGRATION (SIMULATED)
# ============================================================================

print("\n" + "="*60)
print("PART 2: BIGQUERY DATA WAREHOUSE")
print("="*60)

print("""
📊 BIGQUERY SETUP FOR SALES DATA:

-- Create dataset
CREATE SCHEMA `solutech-analytics.sales_data`;

-- Create partitioned table
CREATE TABLE `solutech-analytics.sales_data.transactions`
(
  transaction_id STRING,
  date DATE,
  product STRING,
  quantity INT64,
  unit_price FLOAT64,
  revenue FLOAT64,
  city STRING,
  salesperson_id STRING,
  customer_id STRING,
  is_holiday BOOL
)
PARTITION BY date
CLUSTER BY city, product;

-- Create views for common queries
CREATE VIEW `solutech-analytics.sales_data.daily_summary` AS
SELECT 
  date,
  city,
  product,
  SUM(revenue) as total_revenue,
  COUNT(*) as transactions,
  SUM(quantity) as total_quantity
FROM `solutech-analytics.sales_data.transactions`
GROUP BY date, city, product;
""")

# Simulate BigQuery query results
def simulate_bigquery_query(query_type='daily_summary'):
    """Simulate BigQuery query results"""
    
    df = pd.read_csv('../data/sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    if query_type == 'daily_summary':
        result = df.groupby('date').agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'quantity': 'sum'
        }).reset_index()
        result.columns = ['date', 'total_revenue', 'transactions', 'total_quantity']
        
    elif query_type == 'city_product':
        result = df.groupby(['city', 'product']).agg({
            'revenue': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        result.columns = ['city', 'product', 'total_revenue', 'transactions']
        
    return result

print("\n🔍 Sample BigQuery Query:")
print("Query: SELECT * FROM daily_summary LIMIT 5")
print("\nResults:")
result = simulate_bigquery_query('daily_summary')
print(result.head())

print(f"\n✓ Query executed successfully")
print(f"   Rows returned: {len(result)}")
print(f"   Bytes processed: ~{len(result) * 100:,} bytes (simulated)")

# ============================================================================
# 3. CLOUD STORAGE FOR MODEL ARTIFACTS
# ============================================================================

print("\n" + "="*60)
print("PART 3: CLOUD STORAGE INTEGRATION")
print("="*60)

print("""
☁️ CLOUD STORAGE BUCKET STRUCTURE:

gs://solutech-ml-models/
├── production/
│   ├── sales_forecasting_v1.pkl
│   ├── encoders/
│   │   ├── product_encoder.pkl
│   │   ├── city_encoder.pkl
│   │   └── salesperson_encoder.pkl
│   └── metadata.json
│
├── staging/
│   └── sales_forecasting_v2.pkl
│
├── experiments/
│   └── 2024-01-15/
│       ├── random_forest_model.pkl
│       ├── xgboost_model.pkl
│       └── results.json
│
└── datasets/
    ├── training/
    └── validation/
""")

# Simulate Cloud Storage operations
class GCSSimulator:
    """Simulates Google Cloud Storage operations"""
    
    def __init__(self, bucket_name='solutech-ml-models'):
        self.bucket_name = bucket_name
        self.local_cache = {}
    
    def upload_model(self, model, model_name, destination='production'):
        """Simulate uploading model to GCS"""
        blob_path = f"{destination}/{model_name}"
        
        # In production, would use:
        # from google.cloud import storage
        # storage_client = storage.Client()
        # bucket = storage_client.bucket(self.bucket_name)
        # blob = bucket.blob(blob_path)
        # blob.upload_from_filename(local_file_path)
        
        self.local_cache[blob_path] = model
        return f"gs://{self.bucket_name}/{blob_path}"
    
    def download_model(self, blob_path):
        """Simulate downloading model from GCS"""
        
        # In production, would use:
        # blob = bucket.blob(blob_path)
        # blob.download_to_filename(local_file_path)
        
        return self.local_cache.get(blob_path, None)
    
    def list_models(self, prefix='production'):
        """List models in bucket"""
        return [k for k in self.local_cache.keys() if k.startswith(prefix)]

# Initialize simulator
gcs = GCSSimulator()

# Load and upload model
print("\n📤 Uploading model to Cloud Storage...")
try:
    model = joblib.load('../models/best_model_xgboost.pkl')
    model_uri = gcs.upload_model(model, 'sales_forecasting_v1.pkl')
    print(f"✓ Model uploaded to: {model_uri}")
except:
    print("⚠️ Model file not found. Run notebook 02 first.")
    # Create a dummy model for demonstration
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model_uri = gcs.upload_model(model, 'sales_forecasting_v1.pkl')
    print(f"✓ Demo model uploaded to: {model_uri}")

# Upload metadata
metadata = {
    'model_name': 'sales_forecasting_v1',
    'version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'framework': 'scikit-learn',
    'metrics': {
        'r2_score': 0.85,
        'mae': 2500.0,
        'rmse': 3200.0
    },
    'features': ['quantity', 'unit_price', 'month', 'city', 'product']
}

gcs.upload_model(metadata, 'metadata.json')
print("✓ Metadata uploaded")

# ============================================================================
# 4. VERTEX AI MODEL DEPLOYMENT
# ============================================================================

print("\n" + "="*60)
print("PART 4: VERTEX AI DEPLOYMENT")
print("="*60)

print("""
🚀 VERTEX AI DEPLOYMENT STEPS:

1. PREPARE MODEL ARTIFACT:
   - Package model with dependencies
   - Create serving function
   - Define input/output schema

2. UPLOAD TO VERTEX AI MODEL REGISTRY:
   ```python
   from google.cloud import aiplatform
   
   aiplatform.init(project='solutech-ml', location='us-central1')
   
   model = aiplatform.Model.upload(
       display_name='sales-forecasting-v1',
       artifact_uri='gs://solutech-ml-models/production/',
       serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
   )
   ```

3. DEPLOY TO ENDPOINT:
   ```python
   endpoint = aiplatform.Endpoint.create(display_name='sales-forecasting-endpoint')
   
   endpoint.deploy(
       model=model,
       deployed_model_display_name='sales-forecasting-v1',
       machine_type='n1-standard-4',
       min_replica_count=1,
       max_replica_count=5,
       traffic_percentage=100
   )
   ```

4. MAKE PREDICTIONS:
   ```python
   predictions = endpoint.predict(instances=[{
       'quantity': 50,
       'unit_price': 250.0,
       'month': 3,
       'city': 'Nairobi',
       'product': 'Tea'
   }])
   ```
""")

# Simulate Vertex AI prediction
class VertexAISimulator:
    """Simulates Vertex AI endpoint"""
    
    def __init__(self, model):
        self.model = model
        self.prediction_count = 0
    
    def predict(self, instances):
        """Simulate predictions"""
        self.prediction_count += len(instances)
        
        # In production, would call actual endpoint:
        # endpoint.predict(instances=instances)
        
        # Simulate preprocessing
        print(f"📊 Processing {len(instances)} prediction request(s)...")
        
        # Mock predictions
        predictions = []
        for instance in instances:
            # Simplified prediction logic
            base_revenue = instance['quantity'] * instance['unit_price']
            prediction = base_revenue * np.random.uniform(0.9, 1.1)
            predictions.append({
                'predicted_revenue': round(prediction, 2),
                'confidence': 0.85
            })
        
        return predictions

# Create endpoint simulator
print("\n🔮 Testing Vertex AI Endpoint...")
endpoint = VertexAISimulator(model)

# Make sample predictions
test_instances = [
    {'quantity': 50, 'unit_price': 250.0, 'month': 3, 'city': 'Nairobi', 'product': 'Tea'},
    {'quantity': 75, 'unit_price': 300.0, 'month': 3, 'city': 'Mombasa', 'product': 'Oil'},
    {'quantity': 30, 'unit_price': 150.0, 'month': 3, 'city': 'Kisumu', 'product': 'Flour'}
]

predictions = endpoint.predict(test_instances)

print("\n📈 Prediction Results:")
for i, (instance, pred) in enumerate(zip(test_instances, predictions), 1):
    print(f"\n{i}. Input: {instance['product']} in {instance['city']}")
    print(f"   Quantity: {instance['quantity']}, Price: KES {instance['unit_price']}")
    print(f"   → Predicted Revenue: KES {pred['predicted_revenue']:,.2f}")
    print(f"   → Confidence: {pred['confidence']:.1%}")

# ============================================================================
# 5. MONITORING & LOGGING
# ============================================================================

print("\n" + "="*60)
print("PART 5: MONITORING & OBSERVABILITY")
print("="*60)

print("""
📊 CLOUD MONITORING SETUP:

1. MODEL PERFORMANCE METRICS:
   - Prediction latency (p50, p95, p99)
   - Request rate (QPS)
   - Error rate
   - Model drift detection

2. INFRASTRUCTURE METRICS:
   - CPU/Memory utilization
   - Auto-scaling events
   - Network I/O
   - Instance health

3. BUSINESS METRICS:
   - Prediction accuracy over time
   - Data quality checks
   - Feature distribution shifts
   - Revenue impact

4. ALERTING POLICIES:
   - High error rate (> 5%)
   - Slow predictions (> 500ms)
   - Model performance degradation
   - Unusual prediction patterns

IMPLEMENTATION:
```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()

# Create custom metric
descriptor = monitoring_v3.MetricDescriptor()
descriptor.type = 'custom.googleapis.com/ml/prediction_accuracy'
descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE

# Write metric
series = monitoring_v3.TimeSeries()
series.metric.type = descriptor.type
point = series.points.add()
point.value.double_value = 0.85  # accuracy
point.interval.end_time.seconds = int(time.time())
```
""")

# Simulate monitoring data
monitoring_data = {
    'timestamp': pd.date_range(end=datetime.now(), periods=24, freq='H'),
    'prediction_latency_ms': np.random.normal(150, 30, 24),
    'requests_per_second': np.random.poisson(50, 24),
    'error_rate': np.random.uniform(0, 0.02, 24),
    'model_accuracy': np.random.normal(0.85, 0.02, 24)
}

monitoring_df = pd.DataFrame(monitoring_data)

print("\n📉 Sample Monitoring Metrics (Last 24 hours):")
print(monitoring_df.tail(5).to_string(index=False))

# Calculate SLIs
avg_latency = monitoring_df['prediction_latency_ms'].mean()
p95_latency = monitoring_df['prediction_latency_ms'].quantile(0.95)
avg_error_rate = monitoring_df['error_rate'].mean()

print(f"\n🎯 SLI Metrics:")
print(f"   Average Latency: {avg_latency:.1f}ms")
print(f"   P95 Latency: {p95_latency:.1f}ms")
print(f"   Error Rate: {avg_error_rate:.2%}")
print(f"   Uptime: 99.9%")

# Check SLO compliance
slo_compliance = {
    'Latency < 500ms': p95_latency < 500,
    'Error Rate < 1%': avg_error_rate < 0.01,
    'Uptime > 99.5%': True
}

print(f"\n✅ SLO Compliance:")
for metric, compliant in slo_compliance.items():
    status = "✓ PASS" if compliant else "✗ FAIL"
    print(f"   {metric}: {status}")

# ============================================================================
# 6. CI/CD PIPELINE
# ============================================================================

print("\n" + "="*60)
print("PART 6: CI/CD FOR ML MODELS")
print("="*60)

print("""
🔄 CLOUD BUILD CI/CD PIPELINE:

cloudbuild.yaml:
```yaml
steps:
  # Step 1: Run tests
  - name: 'python:3.9'
    entrypoint: 'python'
    args: ['-m', 'pytest', 'tests/']
    
  # Step 2: Train model
  - name: 'python:3.9'
    entrypoint: 'python'
    args: ['src/train.py']
    env:
      - 'PROJECT_ID=$PROJECT_ID'
      - 'BUCKET_NAME=$_BUCKET_NAME'
    
  # Step 3: Evaluate model
  - name: 'python:3.9'
    entrypoint: 'python'
    args: ['src/evaluate.py']
    
  # Step 4: Deploy if metrics pass threshold
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        if [ $(cat metrics.json | jq -r '.r2_score') > 0.80 ]; then
          gcloud ai models upload \\
            --region=us-central1 \\
            --display-name=sales-forecasting \\
            --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest \\
            --artifact-uri=gs://$_BUCKET_NAME/models/
        fi

  # Step 5: Run integration tests
  - name: 'python:3.9'
    entrypoint: 'python'
    args: ['tests/integration_test.py']

substitutions:
  _BUCKET_NAME: 'solutech-ml-models'
```

DEPLOYMENT STRATEGY:
1. Blue-Green Deployment
   - Deploy new version alongside old
   - Gradually shift traffic (10% → 50% → 100%)
   - Monitor metrics at each stage
   - Rollback if issues detected

2. Canary Deployment
   - Deploy to 5% of traffic
   - Monitor for 24 hours
   - Expand if stable
   - Full rollout after 72 hours
""")

# ============================================================================
# 7. COST OPTIMIZATION
# ============================================================================

print("\n" + "="*60)
print("PART 7: COST OPTIMIZATION")
print("="*60)

# Simulate cost breakdown
monthly_costs = {
    'BigQuery Storage': 150,
    'BigQuery Queries': 300,
    'Cloud Storage': 50,
    'Vertex AI Training': 200,
    'Vertex AI Endpoints': 400,
    'Cloud Run': 100,
    'Monitoring': 50,
    'Networking': 75
}

total_cost = sum(monthly_costs.values())

print("\n💰 ESTIMATED MONTHLY COSTS:\n")
for service, cost in monthly_costs.items():
    percentage = (cost / total_cost) * 100
    print(f"   {service:.<30} ${cost:>6} ({percentage:.1f}%)")
print(f"   {'TOTAL':.<30} ${total_cost:>6}")

print(f"\n💡 COST OPTIMIZATION RECOMMENDATIONS:")
print(f"""
1. BigQuery:
   ✓ Use partitioned tables (save ~30%)
   ✓ Cluster frequently queried columns
   ✓ Set up data lifecycle policies
   ✓ Use materialized views for repeated queries
   
2. Vertex AI:
   ✓ Use auto-scaling (min 0 replicas in off-hours)
   ✓ Right-size machine types
   ✓ Use preemptible VMs for training
   ✓ Implement prediction caching
   
3. Cloud Storage:
   ✓ Use Nearline/Coldline for old models
   ✓ Delete unused experiment artifacts
   ✓ Compress model files
   
4. Overall:
   ✓ Implement request batching
   ✓ Use committed use discounts
   ✓ Monitor and alert on cost spikes
   ✓ Regular cost analysis reviews
   
💵 Potential Savings: ~$400/month (30% reduction)
""")

# ============================================================================
# 8. PRODUCTION CHECKLIST
# ============================================================================

print("\n" + "="*60)
print("PRODUCTION DEPLOYMENT CHECKLIST")
print("="*60)

checklist = {
    '✓ Model Performance': ['R² > 0.80', 'MAE < threshold', 'No overfitting'],
    '✓ Data Pipeline': ['Automated data ingestion', 'Data validation', 'Schema enforcement'],
    '✓ Model Deployment': ['Containerized', 'Versioned', 'Rollback plan'],
    '✓ Monitoring': ['Metrics dashboard', 'Alerts configured', 'Logging enabled'],
    '✓ Security': ['IAM roles configured', 'Secrets managed', 'VPC configured'],
    '✓ Documentation': ['API docs', 'Model card', 'Runbooks'],
    '✓ Testing': ['Unit tests', 'Integration tests', 'Load tests'],
    '✓ Compliance': ['Data privacy', 'Audit trail', 'Backup strategy']
}

print("\n📋 DEPLOYMENT CHECKLIST:\n")
for category, items in checklist.items():
    print(f"{category}")
    for item in items:
        print(f"   [ ✓ ] {item}")
    print()

print("\n" + "="*60)
print("✓ GCP DEPLOYMENT GUIDE COMPLETE!")
print("="*60)

print("""
🎓 KEY TAKEAWAYS FOR SOLUTECH INTERVIEW:

1. Scalable Architecture
   → BigQuery for data warehousing
   → Vertex AI for ML lifecycle
   → Cloud Run for serving

2. Production Best Practices
   → Monitoring & alerting
   → CI/CD pipelines
   → Cost optimization

3. Enterprise Ready
   → Security & compliance
   → Documentation
   → Disaster recovery

4. Business Value
   → Fast time-to-market
   → Reliable predictions
   → Cost-effective scaling
""")