## Demo Video

[Click here to watch the demo](https://youtu.be/uG3sCMOg27k?si=wURz61jpAD9HmUVe)


# lab3_Dhanachitra_Dhanapal
# 🐧 Penguin Species Predictor API

This is a FastAPI-based machine learning API that predicts the species of a penguin using an XGBoost model trained on the Seaborn Penguins dataset.

##  Project Structure

├── app/
│ ├── main.py # FastAPI app for inference
│ └── data/
│ ├── model.json # Trained XGBoost model
│ └── label_encoder_classes.json # Encoded class labels
├── train.py # Script to preprocess and train the model
├── pyproject.toml # Project dependencies
└── README.md # Project documentation



##  Features Used for Prediction

- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g
- sex (categorical: "male", "female")
- island (categorical: "Biscoe", "Dream", "Torgersen")

##  Getting Started

### 1. Set up your environment

```bash
uv venv
.venv\Scripts\activate
uv pip install fastapi uvicorn xgboost pandas scikit-learn seaborn
uvicorn app.main:app --reload 


2. Train the Model

python train.py
This saves the model and label encoder files in app/data/.

3. Run the API

uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
Visit: http://127.0.0.1:8000/docs

API Endpoints
GET /
Check if the server is running.

Response:

{ "status": "Penguin Predictor API is running." }
POST /predict
Make a prediction based on penguin features.

Request Body:
{
  "bill_length_mm": 45.2,
  "bill_depth_mm": 15.4,
  "flipper_length_mm": 220,
  "body_mass_g": 5000,
  "sex": "male",
  "island": "Biscoe"
}
Response:

{ "prediction": "Gentoo" }

Be sure to match the exact enum values for sex and island.

The model and label encoder must be trained before using the API.

Author
Dhanachitra Dhanapal : 100953671
Durham College – AIDI 2004
