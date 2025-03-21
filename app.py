from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("random_forest.pkl")  # Ensure this file exists

@app.route("/", methods=["GET"])
def home():
    return "Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])
        
        if not features:
            return jsonify({"error": "Missing 'features' key in request"}), 400
        
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)

