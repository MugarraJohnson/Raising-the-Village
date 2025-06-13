# WorkMate App - Copilot Instructions

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview

This is the WorkMate App project - a comprehensive mobile application for household vulnerability prediction using machine learning. The project integrates AI/ML capabilities with mobile development and backend services.

## Key Technologies

- **Machine Learning**: TensorFlow, TensorFlow Lite, scikit-learn, pandas
- **Mobile Development**: Android, Kotlin, Jetpack Compose, Room Database
- **Backend**: Flask, SQLite, JWT authentication, RESTful APIs
- **Data Processing**: NumPy, pandas, scikit-learn preprocessing
- **Deployment**: TensorFlow Lite mobile optimization, Flask server deployment

## Project Structure Guidelines

- `ml_pipeline/` - Contains all machine learning code, model training, and evaluation
- `android_app/` - Android mobile application with Kotlin/Compose UI
- `backend_api/` - Flask backend server for data synchronization
- `docs/` - Project documentation and guides

## Development Best Practices

### Machine Learning Code
- Use TensorFlow 2.x with Keras API for model development
- Implement proper data preprocessing pipelines with scikit-learn
- Convert models to TensorFlow Lite format for mobile deployment
- Include comprehensive model evaluation and validation
- Save model artifacts (preprocessors, encoders, metadata) for reproducibility

### Android Development
- Follow MVVM architecture pattern with ViewModel and Repository
- Use Jetpack Compose for modern UI development
- Implement Room database for local data persistence
- Use WorkManager for background synchronization tasks
- Follow Android security best practices for data protection

### Backend Development
- Implement RESTful APIs with proper HTTP status codes
- Use JWT for authentication and authorization
- Include comprehensive error handling and logging
- Implement proper database schema design
- Follow Flask best practices for scalable web applications

## Code Quality Standards

- Write comprehensive unit tests for ML models and business logic
- Include proper error handling and logging throughout the application
- Follow consistent coding standards (PEP 8 for Python, Kotlin style guide)
- Document complex algorithms and business logic
- Use type hints in Python and proper type safety in Kotlin

## AI/ML Specific Guidelines

- When working with the neural network model, consider mobile optimization constraints
- Implement proper feature preprocessing that matches training pipeline
- Handle edge cases in model inference (invalid inputs, confidence thresholds)
- Provide fallback mechanisms for model loading failures
- Consider model versioning and update mechanisms

## Mobile-Specific Considerations

- Optimize for offline functionality - the app must work without internet
- Implement efficient background sync with proper retry mechanisms
- Consider battery optimization for background tasks
- Handle different screen sizes and orientations
- Implement proper permission handling for device features

## Data Privacy and Security

- Implement proper data encryption for local storage
- Follow privacy best practices for sensitive household data
- Ensure secure API communication with HTTPS
- Implement proper authentication and authorization
- Consider data minimization and retention policies

## Testing Strategy

- Unit tests for ML model components and preprocessing
- Integration tests for API endpoints and database operations
- UI tests for critical user flows in the mobile app
- End-to-end tests for data sync and offline scenarios
- Performance tests for model inference speed and API response times

## Common Patterns to Follow

### ML Pipeline
```python
# Use this pattern for model training
class VulnerabilityPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        
    def train_model(self, X, y):
        # Training logic here
        pass
        
    def convert_to_tflite(self):
        # TensorFlow Lite conversion
        pass
```

### Android Repository Pattern
```kotlin
@Singleton
class HouseholdRepository @Inject constructor(
    private val localDatabase: WorkMateDatabase,
    private val apiService: WorkMateApiService
) {
    suspend fun saveHouseholdLocally(data: HouseholdData) {
        // Local save logic
    }
    
    suspend fun syncToServer(data: HouseholdData) {
        // Server sync logic
    }
}
```

### API Error Handling
```python
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"API error: {error}")
    return jsonify({'message': 'Internal server error'}), 500
```

## Performance Considerations

- Model inference should complete within 100ms on mobile devices
- API responses should be under 2 seconds for normal operations
- Local database operations should be non-blocking
- Background sync should not impact user experience
- Consider data compression for large sync operations

## Deployment Notes

- TensorFlow Lite models should be under 5MB for mobile deployment
- Backend should support horizontal scaling for production
- Database migrations should be backwards compatible
- Consider CDN for model distribution and updates
- Implement proper monitoring and alerting for production systems

When generating code for this project, always consider these guidelines and maintain consistency with the existing codebase architecture and patterns.
