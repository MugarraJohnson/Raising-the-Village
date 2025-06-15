# WorkMate App - API Documentation

## Overview

The WorkMate Backend API provides endpoints for household data synchronization, model updates, and analytics. All endpoints use JSON for data exchange and require JWT authentication.

## Base URL
```
Development: http://localhost:5000/api
Production: https://your-domain.com/api
```

## Authentication

### Register Device
Register a new device/field officer to obtain an authentication token.

**Endpoint:** `POST /auth/register`

**Request Body:**
```json
{
  "device_id": "unique-device-identifier"
}
```

**Response:**
```json
{
  "message": "Device registered successfully",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "device_id": "unique-device-identifier"
}
```

### Using Authentication Token
Include the token in the Authorization header for all API requests:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## Household Data Endpoints

### Sync Single Household
Synchronize a single household record with prediction data.

**Endpoint:** `POST /households`

**Request Body:**
```json
{
  "householdData": {
    "householdId": "HH_1234567890",
    "householdSize": 5,
    "income": 15000,
    "age": 35,
    "education": "Primary",
    "progressStatus": "Struggling",
    "region": "South",
    "programParticipation": "Yes",
    "waterAccess": "No",
    "electricityAccess": "Yes",
    "healthcareAccess": "No"
  },
  "prediction": {
    "level": "MODERATE",
    "confidence": 0.82,
    "recommendations": [
      "Schedule regular monitoring visits",
      "Enroll in skill development programs",
      "Provide information about available support services"
    ]
  },
  "timestamp": 1697123456789,
  "deviceId": "DEVICE_001"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Household data synced successfully",
  "householdId": "HH_1234567890"
}
```

### Batch Sync Households
Synchronize multiple household records in a single request.

**Endpoint:** `POST /households/batch`

**Request Body:**
```json
[
  {
    "householdData": { /* household data object */ },
    "prediction": { /* prediction object */ },
    "timestamp": 1697123456789,
    "deviceId": "DEVICE_001"
  },
  {
    "householdData": { /* household data object */ },
    "prediction": { /* prediction object */ },
    "timestamp": 1697123456790,
    "deviceId": "DEVICE_001"
  }
]
```

**Response:**
```json
{
  "syncedCount": 2,
  "failedCount": 0,
  "message": "Batch sync completed: 2 success, 0 failed"
}
```

### Get Households
Retrieve household data with optional filtering.

**Endpoint:** `GET /households`

**Query Parameters:**
- `region` (optional): Filter by region
- `limit` (optional): Maximum number of records (default: 100)

**Example:** `GET /households?region=South&limit=50`

**Response:**
```json
{
  "households": [
    {
      "id": "HH_1234567890",
      "household_size": 5,
      "income": 15000,
      "vulnerability_level": "MODERATE",
      "confidence": 0.82,
      "region": "South",
      "timestamp": 1697123456789,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

## Model Management Endpoints

### Check Model Updates
Check if a newer version of the ML model is available.

**Endpoint:** `GET /model/version`

**Query Parameters:**
- `currentVersion`: Current model version installed on device

**Example:** `GET /model/version?currentVersion=1.0.0`

**Response:**
```json
{
  "hasUpdate": true,
  "modelVersion": "1.1.0",
  "downloadUrl": "/api/model/download?version=1.1.0",
  "modelSize": 2048576
}
```

### Download Model Update
Download the updated ML model file.

**Endpoint:** `GET /model/download`

**Query Parameters:**
- `version`: Model version to download

**Example:** `GET /model/download?version=1.1.0`

**Response:** Binary file (TensorFlow Lite model)

## Analytics Endpoints

### Get Analytics Data
Retrieve analytics data for dashboard and reporting.

**Endpoint:** `GET /analytics`

**Query Parameters:**
- `days` (optional): Number of days to include in analytics (default: 30)

**Example:** `GET /analytics?days=7`

**Response:**
```json
{
  "total_households": 150,
  "vulnerability_distribution": {
    "HIGH": 45,
    "MODERATE": 60,
    "LOW": 45
  },
  "regional_distribution": {
    "North": 30,
    "South": 40,
    "East": 35,
    "West": 25,
    "Central": 20
  },
  "period_days": 7
}
```

## Utility Endpoints

### Health Check
Check API server health and status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

## Error Responses

### Authentication Errors
```json
{
  "message": "Token is missing"
}
```
**Status Code:** 401 Unauthorized

```json
{
  "message": "Token has expired"
}
```
**Status Code:** 401 Unauthorized

### Validation Errors
```json
{
  "message": "Invalid request data",
  "errors": {
    "householdSize": "Value must be a positive number",
    "income": "Value is required"
  }
}
```
**Status Code:** 400 Bad Request

### Server Errors
```json
{
  "message": "Internal server error"
}
```
**Status Code:** 500 Internal Server Error

## Data Models

### HouseholdData
```json
{
  "householdId": "string",
  "householdSize": "number",
  "income": "number",
  "age": "number",
  "education": "string (None|Primary|Secondary|Higher)",
  "progressStatus": "string (Severely Struggling|Struggling|At Risk|On Track)",
  "region": "string",
  "programParticipation": "string (Yes|No)",
  "waterAccess": "string (Yes|No)",
  "electricityAccess": "string (Yes|No)",
  "healthcareAccess": "string (Yes|No)",
  "timestamp": "number (Unix timestamp in milliseconds)"
}
```

### VulnerabilityPrediction
```json
{
  "level": "string (HIGH|MODERATE|LOW)",
  "confidence": "number (0.0 to 1.0)",
  "recommendations": ["string array"]
}
```

## Rate Limiting

API requests are rate limited to prevent abuse:
- **Authentication endpoints**: 10 requests per minute per IP
- **Data sync endpoints**: 100 requests per minute per device
- **Model download**: 5 requests per hour per device
- **Analytics endpoints**: 20 requests per minute per device

## SDK and Code Examples

### Python Example
```python
import requests
import json

# Base configuration
API_BASE_URL = "http://localhost:5000/api"
TOKEN = "your-jwt-token-here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Sync household data
household_data = {
    "householdData": {
        "householdId": "HH_123",
        "householdSize": 4,
        "income": 12000,
        # ... other fields
    },
    "prediction": {
        "level": "HIGH",
        "confidence": 0.89,
        "recommendations": ["Immediate referral to social services"]
    },
    "timestamp": int(time.time() * 1000),
    "deviceId": "DEVICE_001"
}

response = requests.post(
    f"{API_BASE_URL}/households",
    headers=headers,
    json=household_data
)

if response.status_code == 200:
    print("Sync successful!")
else:
    print(f"Sync failed: {response.json()}")
```

### Android/Kotlin Example
```kotlin
// Using Retrofit
interface WorkMateApiService {
    @POST("households")
    suspend fun syncHousehold(
        @Header("Authorization") token: String,
        @Body request: HouseholdApiRequest
    ): Response<HouseholdApiResponse>
}

// Usage
val apiService = /* initialize Retrofit service */
val token = "Bearer your-jwt-token-here"
val request = HouseholdApiRequest(/* ... */)

try {
    val response = apiService.syncHousehold(token, request)
    if (response.isSuccessful) {
        println("Sync successful!")
    } else {
        println("Sync failed: ${response.errorBody()}")
    }
} catch (e: Exception) {
    println("Network error: ${e.message}")
}
```

## Testing

### Using cURL
```bash
# Register device
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"device_id": "TEST_DEVICE"}'

# Health check
curl -X GET http://localhost:5000/api/health

# Sync household (replace TOKEN with actual token)
curl -X POST http://localhost:5000/api/households \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d @household_data.json
```

### Using Postman
1. Import the API collection (if available)
2. Set up environment variables for base URL and token
3. Test each endpoint with sample data

## Deployment Considerations

### Production Setup
1. **Environment Variables:**
   - Set `FLASK_ENV=production`
   - Use secure `SECRET_KEY`
   - Configure proper database URL

2. **Security:**
   - Use HTTPS in production
   - Implement proper CORS policies
   - Set up rate limiting
   - Use secure headers

3. **Monitoring:**
   - Set up logging and monitoring
   - Track API performance metrics
   - Monitor error rates and response times

4. **Scaling:**
   - Use production WSGI server (Gunicorn)
   - Implement database connection pooling
   - Consider load balancing for high traffic

This API documentation provides comprehensive information for integrating with the WorkMate backend services. For additional support, refer to the main project documentation or contact the development team.
