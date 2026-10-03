# Iris ML Prediction API

## Project Description

This project demonstrates an end-to-end machine learning deployment workflow using a trained machine learning model and FastAPI.

The model is a Random Forest classifier trained on the Iris dataset.

## What the Model Predicts

The model predicts the class of an Iris flower using four features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The prediction classes are:

* `0` - Setosa
* `1` - Versicolor
* `2` - Virginica

## API Endpoints

### GET /health

Checks whether the API is running and whether the trained model has been loaded successfully.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Predicts the Iris flower class.

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": 0
}
```

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Model File

The trained Random Forest model is saved as `model.pkl` using Joblib.
