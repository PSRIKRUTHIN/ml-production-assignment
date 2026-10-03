from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained model
try:
    model = joblib.load("model.pkl")
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris ML Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }

            .container {
                background: white;
                width: 420px;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
                margin-bottom: 10px;
            }

            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 20px;
            }

            .status {
                text-align: center;
                margin-bottom: 20px;
                color: green;
                font-weight: bold;
            }

            label {
                display: block;
                margin-top: 12px;
                font-weight: bold;
            }

            input {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 15px;
            }

            button {
                width: 100%;
                margin-top: 20px;
                padding: 12px;
                border: none;
                border-radius: 6px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            #result {
                margin-top: 20px;
                padding: 15px;
                background: #f0fdf4;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                display: none;
            }

            .api-link {
                text-align: center;
                margin-top: 20px;
            }

            .api-link a {
                color: #2563eb;
                text-decoration: none;
            }
        </style>
    </head>

    <body>

    <div class="container">

        <h1>Iris Flower Predictor</h1>

        <div class="subtitle">
            Random Forest Machine Learning API
        </div>

        <div class="status">
            ● ML Model Loaded
        </div>

        <label>Sepal Length</label>
        <input id="sepal_length" type="number" step="0.1" value="5.1">

        <label>Sepal Width</label>
        <input id="sepal_width" type="number" step="0.1" value="3.5">

        <label>Petal Length</label>
        <input id="petal_length" type="number" step="0.1" value="1.4">

        <label>Petal Width</label>
        <input id="petal_width" type="number" step="0.1" value="0.2">

        <button onclick="predict()">Predict Iris Class</button>

        <div id="result"></div>

        <div class="api-link">
            <a href="/docs" target="_blank">Open FastAPI Documentation</a>
        </div>

    </div>

    <script>
        async function predict() {

            const data = {
                sepal_length: parseFloat(
                    document.getElementById("sepal_length").value
                ),
                sepal_width: parseFloat(
                    document.getElementById("sepal_width").value
                ),
                petal_length: parseFloat(
                    document.getElementById("petal_length").value
                ),
                petal_width: parseFloat(
                    document.getElementById("petal_width").value
                )
            };

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                const resultBox = document.getElementById("result");

                const classes = {
                    0: "Setosa",
                    1: "Versicolor",
                    2: "Virginica"
                };

                resultBox.style.display = "block";

                resultBox.innerHTML =
                    "Prediction: Class " +
                    result.prediction +
                    " (" +
                    classes[result.prediction] +
                    ")";

            } catch (error) {
                const resultBox = document.getElementById("result");
                resultBox.style.display = "block";
                resultBox.innerHTML = "Prediction failed. Please try again.";
            }
        }
    </script>

    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


@app.post("/predict")
def predict(data: IrisInput):

    if model is None:
        return {
            "error": "Model not loaded"
        }

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction)
    }