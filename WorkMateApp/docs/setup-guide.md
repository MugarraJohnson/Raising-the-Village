# WorkMate App - Setup Guide

## Quick Setup

### Prerequisites
- Python 3.8+ installed
- Android Studio (for mobile development)
- Git for version control

### Installation Steps

1. **Clone or Download Project**
   ```bash
   # If using Git
   git clone <repository-url>
   cd "WorkMate App"
   
   # Or download and extract ZIP file
   ```

2. **Run Automated Deployment**
   ```bash
   # Full deployment (recommended for first setup)
   python deploy.py
   
   # Or step by step
   python deploy.py --skip-android  # Skip Android prep
   python deploy.py --validate-only # Just validate setup
   ```

3. **Manual Setup (Alternative)**

   **Python Environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

   **Train ML Model:**
   ```bash
   python ml_pipeline/model_training.py
   ```

   **Setup Backend:**
   ```bash
   cd backend_api
   pip install -r requirements.txt
   python app.py
   ```

### Verification

After setup, verify everything works:

1. **Check ML Model**
   ```bash
   python ml_pipeline/model_evaluation.py
   ```

2. **Test Backend API**
   ```bash
   # Start backend server
   cd backend_api
   python app.py
   
   # In another terminal, test API
   curl http://localhost:5000/api/health
   ```

3. **Android App Setup**
   - Open Android Studio
   - Import `android_app` folder
   - Sync Gradle dependencies
   - Build and run on device/emulator

## Troubleshooting

### Common Issues

**1. Python Dependencies**
```bash
# If installation fails, try upgrading pip
python -m pip install --upgrade pip

# Install specific versions
pip install tensorflow==2.13.0
```

**2. TensorFlow Lite Conversion**
```bash
# If model conversion fails
pip install tensorflow==2.13.0 --upgrade
python ml_pipeline/model_training.py
```

**3. Android Build Issues**
```bash
# Sync Gradle in Android Studio
./gradlew clean
./gradlew build
```

**4. Backend Database Issues**
```bash
# Reset database
rm workmate_backend.db
python -c "from app import db_manager; db_manager.init_database()"
```

### Getting Help

1. Check the main [README.md](README.md) for detailed documentation
2. Review error logs in the console output
3. Ensure all prerequisites are installed
4. Try the automated deployment script: `python deploy.py`

## Development Mode

For development, you can run components separately:

**Backend Development:**
```bash
cd backend_api
export FLASK_ENV=development
python app.py
```

**ML Model Development:**
```bash
# Train model with different parameters
python ml_pipeline/model_training.py

# Evaluate model performance
python ml_pipeline/model_evaluation.py
```

**Android Development:**
- Use Android Studio for UI development
- Use device debugging for testing TensorFlow Lite integration
- Monitor logs for ML inference performance

## Production Deployment

For production deployment:

1. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env with production settings
   ```

2. **Build Production Model:**
   ```bash
   python ml_pipeline/model_training.py
   ```

3. **Deploy Backend:**
   ```bash
   # Use production WSGI server
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 backend_api.app:app
   ```

4. **Build Android APK:**
   ```bash
   cd android_app
   ./gradlew assembleRelease
   ```

Your WorkMate App should now be ready for use! 🚀
