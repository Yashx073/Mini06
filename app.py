import joblib
from flask import Flask, request, jsonify
import numpy as np

# Load the trained model and scaler
model = joblib.load("student_model.pkl")  # Your trained Logistic Regression model
scaler = joblib.load("scaler.pkl")  # Load the scaler (ensure you saved it)

# Updated feature names
feature_names = [
    "CGPA", "Major Projects", "Workshops/Certifications", "Mini Projects", "Skills", 
    "Communication Skill Rating", "12th Percentage", "10th Percentage", 
    "backlogs", "Internship_Yes", "Hackathon_Yes"
]

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from Flutter (JSON format)
        data = request.json["input"]  # Example: {"input": [8.5, 2, 3, 1, 5, 4, 85, 90, 0, 1, 1]}
        input_array = np.array(data).reshape(1, -1)

        # Scale input using the same scaler from training
        input_scaled = scaler.transform(input_array)

        # Make a prediction
        prediction = model.predict(input_scaled)
        
        # Make a prediction
        prediction = model.predict(input_scaled)[0]
        result = "Placed" if prediction == 1 else "Not Placed"
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
