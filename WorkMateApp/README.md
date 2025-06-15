# WorkMate App - Machine Learning Integration for Household Vulnerability Prediction

## Overview

WorkMate App is a comprehensive mobile application designed for field officers working in last-mile communities to predict household vulnerability levels using Annual Household Survey (AHS) data. The app integrates machine learning capabilities with offline functionality, ensuring reliable operation in low-bandwidth environments.

## Features

### Core Functionality
- **Household Data Collection**: Intuitive forms for collecting comprehensive household information
- **AI-Powered Predictions**: Neural network model for vulnerability level prediction (High, Moderate, Low)
- **Offline Operation**: Full functionality without internet connectivity
- **Background Synchronization**: Automatic data sync when connectivity is restored
- **Real-time Recommendations**: Actionable insights based on prediction results

### Technical Capabilities
- **TensorFlow Lite Integration**: On-device model inference
- **Room Database**: Local data persistence with encryption
- **WorkManager**: Reliable background task execution
- **Retrofit API**: Robust server communication
- **Jetpack Compose UI**: Modern, responsive user interface

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Backend API   │    │  ML Pipeline    │
│                 │    │                 │    │                 │
│ • TensorFlow    │◄──►│ • Flask Server  │◄──►│ • Model Training│
│   Lite Model    │    │ • SQLite DB     │    │ • TF Conversion │
│ • Room Database │    │ • JWT Auth      │    │ • Evaluation    │
│ • WorkManager   │    │ • Data Sync     │    │ • Versioning    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Field Officer Input**: Collects household data via mobile form
2. **Local Processing**: TensorFlow Lite model processes data offline
3. **Prediction Generation**: AI model outputs vulnerability level and confidence
4. **Local Storage**: Data saved in encrypted Room database
5. **Background Sync**: When online, data syncs to backend server
6. **Model Updates**: Backend provides updated models for download

## Project Structure

```
WorkMate App/
├── ml_pipeline/                    # Machine Learning components
│   ├── model_training.py          # Neural network training and TFLite conversion
│   ├── model_evaluation.py        # Model validation and performance metrics
│   └── data/                      # Training data and models
├── android_app/                   # Android mobile application
│   ├── src/main/java/com/workmate/app/
│   │   ├── HouseholdViewModel.kt   # Main ViewModel with ML integration
│   │   ├── HouseholdRepository.kt  # Data management and sync
│   │   └── SyncWorker.kt          # Background synchronization
│   ├── build.gradle               # Android build configuration
│   └── AndroidManifest.xml        # App permissions and components
├── backend_api/                   # Flask backend server
│   ├── app.py                     # Main API server
│   └── requirements.txt           # Backend dependencies
├── docs/                          # Documentation
└── requirements.txt               # Python ML dependencies
```

## Quick Start

### Prerequisites
- Python 3.8+
- Android Studio (for mobile app development)
- TensorFlow 2.13+
- Flask 2.3+

### Setup Instructions

#### 1. ML Pipeline Setup
```bash
# Navigate to project directory
cd "WorkMate App"

# Install Python dependencies
pip install -r requirements.txt

# Train the model
python ml_pipeline/model_training.py

# Evaluate model performance
python ml_pipeline/model_evaluation.py
```

#### 2. Backend API Setup
```bash
# Navigate to backend directory
cd backend_api

# Install backend dependencies
pip install -r requirements.txt

# Initialize database
flask init-db

# Create sample data (optional)
flask create-sample-data

# Run the server
python app.py
```

#### 3. Android App Setup
```bash
# Open Android Studio
# Import the android_app directory as a project
# Sync Gradle dependencies
# Copy the generated .tflite model to assets/
# Build and run the app
```

## Model Details

### Neural Network Architecture
- **Input Layer**: 20 features (numerical and categorical)
- **Hidden Layers**: 
  - Dense(128) + ReLU + Dropout(0.3)
  - Dense(64) + ReLU + Dropout(0.3)
  - Dense(32) + ReLU
- **Output Layer**: Dense(3) + Softmax (High, Moderate, Low)

### Features Used
- **Numerical**: Household size, income, age
- **Categorical**: Education level, progress status, region
- **Binary**: Program participation, water/electricity/healthcare access

### Model Performance
- **Accuracy**: ~75-85% (varies by dataset)
- **Model Size**: ~2MB (TensorFlow Lite)
- **Inference Time**: <100ms on mobile devices

## API Documentation

### Authentication
All API endpoints require JWT token authentication:
```http
Authorization: Bearer <token>
```

### Key Endpoints

#### Sync Household Data
```http
POST /api/households
Content-Type: application/json

{
  "householdData": {
    "householdId": "HH_123",
    "householdSize": 5,
    "income": 15000,
    "education": "Primary",
    ...
  },
  "prediction": {
    "level": "MODERATE",
    "confidence": 0.82,
    "recommendations": ["Schedule regular monitoring"]
  }
}
```

#### Check Model Updates
```http
GET /api/model/version?currentVersion=1.0.0
```

#### Get Analytics
```http
GET /api/analytics?days=30
```

## Field Officer User Flow

### Step-by-Step Process
1. **App Launch**: Field officer opens WorkMate app
2. **Household Selection**: Choose existing or create new household entry
3. **Data Input**: Complete household information form
4. **AI Processing**: App processes data using local TensorFlow Lite model
5. **Results Display**: View vulnerability prediction and recommendations
6. **Data Storage**: Information saved locally (encrypted)
7. **Sync Management**: 
   - **Online**: Data syncs immediately to server
   - **Offline**: Queued for background sync when connected

### Offline Capabilities
- Full form completion and submission
- AI model inference without internet
- Local data storage with encryption
- Background sync queue management
- Automatic retry logic for failed syncs

## Security Features

### Data Protection
- **Local Encryption**: Room database with SQLCipher
- **HTTPS Communication**: All API calls use TLS
- **JWT Authentication**: Secure token-based auth
- **Data Anonymization**: PII protection in analytics

### Privacy Compliance
- Minimal data collection
- Local-first data processing
- User consent management
- Data retention policies

## Performance Optimization

### Mobile App
- **Model Quantization**: Reduced model size for mobile deployment
- **Lazy Loading**: Efficient memory management
- **Background Processing**: Non-blocking UI operations
- **Caching Strategy**: Smart data caching for offline use

### Backend
- **Database Indexing**: Optimized query performance
- **Connection Pooling**: Efficient database connections
- **Rate Limiting**: API protection against abuse
- **Batch Operations**: Efficient bulk data processing

## Monitoring and Analytics

### Application Metrics
- Prediction accuracy tracking
- Sync success rates
- Model performance monitoring
- User engagement analytics

### Operational Metrics
- API response times
- Database performance
- Error rates and patterns
- Resource utilization

## Deployment

### Mobile App Distribution
- **Production Build**: Optimized APK generation
- **Play Store**: Standard Android distribution
- **Enterprise**: Internal distribution options
- **Updates**: Over-the-air update mechanism

### Backend Deployment
- **Docker Support**: Containerized deployment
- **Cloud Platforms**: AWS, GCP, Azure compatible
- **Load Balancing**: Horizontal scaling support
- **Database**: SQLite for development, PostgreSQL for production

## Development Guidelines

### Code Standards
- **Kotlin**: Android development language
- **MVVM Pattern**: Clean architecture implementation
- **Dependency Injection**: Hilt for Android DI
- **Testing**: Unit and integration test coverage

### Contributing
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request with documentation

## Troubleshooting

### Common Issues

#### Model Loading Errors
```kotlin
// Check TensorFlow Lite model path
val modelPath = "vulnerability_model.tflite"
// Ensure model exists in assets folder
```

#### Sync Failures
```kotlin
// Check network connectivity
if (networkConnectivityHelper.isConnected.value) {
    // Retry sync operation
    syncManager.triggerImmediateSync()
}
```

#### Database Issues
```sql
-- Check database integrity
PRAGMA integrity_check;
-- Rebuild if necessary
VACUUM;
```

## Support and Resources

### Documentation
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [Android Jetpack Compose](https://developer.android.com/jetpack/compose)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Community
- GitHub Issues for bug reports
- Discussions for feature requests
- Wiki for additional documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 1.0.0 (2025-06-13)
- Initial release
- Core ML pipeline implementation
- Android app with offline capabilities
- Backend API with sync functionality
- Documentation and deployment guides

---

**WorkMate App** - Empowering field officers with AI-driven insights for better community support.
