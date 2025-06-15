# WorkMate ML Model - Quick Terminal Deployment Guide

## 🚀 Get Started in 3 Minutes

### Prerequisites
- Python 3.8+ installed
- PowerShell/Command Prompt access

### Step 1: Set Up Environment
```powershell
# Navigate to WorkMate directory
cd "d:\Git\WorkMateApp"

# Create virtual environment (if not exists)
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Deployment
```powershell
# Check if everything is working
python verify_deployment.py
```

### Step 3: Quick Predictions

#### Option A: Interactive Mode
```powershell
# Interactive household assessment
python quick_predict.py --interactive
```

#### Option B: Quick Example
```powershell
# See example prediction
python quick_predict.py --example
```

#### Option C: Batch Processing
```powershell
# Process CSV file
python quick_predict.py --batch your_data.csv
```

## 🔧 Quick Commands Reference

### Model Operations
```powershell
# Retrain model
cd ml_pipeline && python workmate_sklearn_model.py

# Check model info
python -c "import joblib; import json; meta = json.load(open('models/workmate_vulnerability_model_metadata.json')); print(f'Accuracy: {meta[\"test_accuracy\"]:.1%}, Trained: {meta[\"training_date\"]}')"

# Start API server
cd deployment && python flask_api.py
```

### Data Operations
```powershell
# Check dataset
python -c "import pandas as pd; df = pd.read_csv('Datasets/DataScientist_01_Assessment.csv'); print(f'Dataset: {len(df)} records, {len(df.columns)} columns')"

# View model features
python -c "from ml_pipeline.workmate_sklearn_model import WorkMateVulnerabilityPredictor; p = WorkMateVulnerabilityPredictor(); p.load_model(); print('Required features:', p.feature_names)"
```

## 📊 Understanding Results

### Vulnerability Levels
- **High**: Families earning < $0.50/day (urgent support needed)
- **Moderate-High**: $0.50-$1.00/day (significant concerns)
- **Moderate**: $1.00-$2.00/day (some support helpful)
- **Low**: > $2.00/day (relatively stable)

### Confidence Levels
- **>80%**: Very confident prediction
- **60-80%**: Good confidence
- **40-60%**: Moderate confidence
- **<40%**: Low confidence (manual review recommended)

## 🛠️ Troubleshooting

### Common Issues
```powershell
# If model file not found
ls models/  # Check if model exists
cd ml_pipeline && python workmate_sklearn_model.py  # Retrain if needed

# If dependencies missing
pip install scikit-learn pandas numpy joblib flask

# If dataset not found
ls Datasets/  # Verify dataset exists

# If API port busy
# Change port in deployment/flask_api.py or kill process using port 5000
```

### Performance Tips
- Model loads in ~2 seconds
- Single prediction: ~50ms
- Batch predictions: ~10ms per household
- API response: ~100ms average

## 📱 Production Deployment
For production use:
1. Set up proper web server (Gunicorn/uWSGI)
2. Add authentication and rate limiting
3. Set up monitoring and logging
4. Use environment variables for configuration
5. Consider containerization with Docker

---
**Need Help?** Check the full `DEPLOYMENT_GUIDE.md` for detailed instructions or run the test script: `python deployment/test_deployment.py`
