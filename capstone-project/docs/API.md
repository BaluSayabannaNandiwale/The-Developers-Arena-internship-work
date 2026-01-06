# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production, implement API keys or OAuth2.

## Endpoints

### 1. Root

**GET** `/`

Returns basic API information.

**Response:**
```json
{
  "message": "Real Estate Price Predictor API",
  "version": "1.0.0",
  "status": "operational"
}
```

### 2. Health Check

**GET** `/api/v1/health`

Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "preprocessor_status": "loaded",
  "timestamp": "2024-01-01T12:00:00"
}
```

### 3. Single Prediction

**POST** `/api/v1/predict`

Predicts price for a single property.

**Request Body:**
```json
{
  "area_sqft": 1200,
  "bedrooms": 2,
  "bathrooms": 2,
  "floor": 3,
  "total_floors": 10,
  "property_age": 5,
  "location": "City Center",
  "city": "Bangalore",
  "property_type": "Apartment",
  "facing": "East",
  "furnishing": "Semi-Furnished",
  "parking": 1,
  "amenities_score": 0.75,
  "distance_city_center_km": 12.4
}
```

**Response:**
```json
{
  "prediction_id": "uuid",
  "timestamp": "2024-01-01T12:00:00",
  "predicted_price": 5000000.00,
  "currency": "INR",
  "confidence_interval": {
    "lower_bound": 4500000.00,
    "upper_bound": 5500000.00
  },
  "model_version": "1.0.0",
  "metadata": {
    "area_sqft": 1200,
    "location": "City Center",
    "property_type": "Apartment",
    "city": "Bangalore"
  }
}
```

### 4. Batch Prediction

**POST** `/api/v1/batch`

Predicts prices for multiple properties.

**Request Body:**
```json
{
  "properties": [
    {
      "area_sqft": 1200,
      ...
    },
    {
      "area_sqft": 1500,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction_id": "uuid",
      "timestamp": "2024-01-01T12:00:00",
      "predicted_price": 5000000.00,
      ...
    }
  ],
  "total_count": 2
}
```

### 5. Metrics

**GET** `/api/v1/metrics`

Returns Prometheus metrics.

**Response:** Plain text Prometheus format

### 6. API Documentation

**GET** `/docs`

Interactive API documentation (Swagger UI).

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Area must be positive"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "area_sqft"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded. Please ensure model files exist."
}
```

