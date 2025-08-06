# Build the Docker image
docker build -t no-show-predictor .

# Run the container
docker run -p 8000:8000 no-show-predictor

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Age": 34,
        "Gender": 1,
        "Scholarship": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Hypertension": 1,
        "SMS_received": 1,
        "WaitingDays": 3
      }'
