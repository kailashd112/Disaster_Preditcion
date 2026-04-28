from flask import Flask, render_template, request, jsonify
import os
import pickle
import math

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DISASTERS = {
    "flood": {
        "title": "Flood",
        "icon": "🌊",
        "model": "flood_model.pkl",
        "features": ["rainfall", "river_level", "temperature", "humidity", "soil_moisture"],
        "units": ["mm", "m", "°C", "%", "%"],
        "tip": "High rainfall, high river level, and saturated soil increase flood risk."
    },
    "earthquake": {
        "title": "Earthquake",
        "icon": "🌍",
        "model": "earthquake_model.pkl",
        "features": ["magnitude", "depth_km", "latitude", "longitude", "risk_score"],
        "units": ["Mw", "km", "lat", "lon", "0-1"],
        "tip": "Higher magnitude and risk score usually indicate higher earthquake danger."
    },
    "fire": {
        "title": "Wild Fire",
        "icon": "🔥",
        "model": "fire_model.pkl",
        "features": ["temperature", "humidity", "wind_speed", "rainfall", "drought_index"],
        "units": ["°C", "%", "km/h", "mm", "0-10"],
        "tip": "Hot, dry, windy conditions increase fire risk."
    },
    "cyclone": {
        "title": "Cyclone",
        "icon": "🌪️",
        "model": "cyclone_model.pkl",
        "features": ["sea_temp", "wind_speed", "pressure", "humidity"],
        "units": ["°C", "km/h", "hPa", "%"],
        "tip": "Warm sea, fast wind, and lower pressure increase cyclone risk."
    },
    "drought": {
        "title": "Drought",
        "icon": "🌵",
        "model": "drought_model.pkl",
        "features": ["temperature", "rainfall", "soil_moisture", "evaporation"],
        "units": ["°C", "mm", "%", "mm/day"],
        "tip": "High temperature, low rainfall, low soil moisture, and high evaporation increase drought risk."
    }
}

def load_model(disaster):
    path = os.path.join(MODEL_DIR, DISASTERS[disaster]["model"])
    with open(path, "rb") as file:
        return pickle.load(file)

MODELS = {name: load_model(name) for name in DISASTERS}

def sigmoid(x):
    return 1 / (1 + math.exp(-max(min(x, 60), -60)))

def predict_probability(config, values):
    score = config.get("bias", 0)
    for weight, value in zip(config["weights"], values):
        score += weight * value
    return sigmoid(score)

def level_from_probability(prob):
    if prob >= 0.70:
        return "High Risk", "danger", "Disaster may occur. Take safety precautions."
    if prob >= 0.40:
        return "Moderate Risk", "warning", "Conditions are risky. Monitor carefully."
    return "Low Risk", "safe", "Disaster is unlikely based on current input."

@app.route("/")
def home():
    return render_template("index.html", disasters=DISASTERS)

@app.route("/api/predict/<disaster>", methods=["POST"])
def predict(disaster):
    if disaster not in DISASTERS:
        return jsonify({"error": "Invalid disaster type"}), 404

    payload = request.get_json(silent=True) or request.form
    features = DISASTERS[disaster]["features"]

    try:
        values = [float(payload.get(feature)) for feature in features]
    except (TypeError, ValueError):
        return jsonify({"error": "Please enter valid numeric values for all fields."}), 400

    prob = predict_probability(MODELS[disaster], values)
    label, level_class, message = level_from_probability(prob)

    return jsonify({
        "disaster": DISASTERS[disaster]["title"],
        "probability": round(prob * 100, 2),
        "label": label,
        "level_class": level_class,
        "message": message,
        "features": features,
        "values": values
    })

# Backward-compatible routes for your old frontend
@app.route("/predict_flood", methods=["POST"])
def predict_flood():
    return old_style_prediction("flood")

@app.route("/predict_earthquake", methods=["POST"])
def predict_earthquake():
    return old_style_prediction("earthquake")

@app.route("/predict_fire", methods=["POST"])
def predict_fire():
    return old_style_prediction("fire")

@app.route("/predict_cyclone", methods=["POST"])
def predict_cyclone():
    return old_style_prediction("cyclone")

@app.route("/predict_drought", methods=["POST"])
def predict_drought():
    return old_style_prediction("drought")

def old_style_prediction(disaster):
    features = DISASTERS[disaster]["features"]
    values = [float(x) for x in request.form.values()]
    prob = predict_probability(MODELS[disaster], values)
    label, _, _ = level_from_probability(prob)
    return f"{DISASTERS[disaster]['title']} Probability: {prob*100:.2f}% — {label}"

if __name__ == "__main__":
    app.run(debug=True)
