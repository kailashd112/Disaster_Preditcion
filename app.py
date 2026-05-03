from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DISASTERS = {
    "flood": {
        "title": "Flood",
        "icon": "🌊",
        "model": "flood_dataset_model.pkl",
        "features": ["rainfall", "river_level", "temperature", "humidity", "soil_moisture"],
        "units": ["mm", "m", "°C", "%", "%"],
        "tip": "High rainfall, high river level, and saturated soil increase flood risk."
    },
    "earthquake": {
        "title": "Earthquake",
        "icon": "🌍",
        "model": "earthquake_dataset_model.pkl",
        "features": ["magnitude", "depth_km", "latitude", "longitude"],
        "units": ["Mw", "km", "lat", "lon"],
        "tip": "Higher magnitude and shallow depth can increase earthquake risk."
    },
    "fire": {
        "title": "Wild Fire",
        "icon": "🔥",
        "model": "fire_dataset_model.pkl",
        "features": ["temperature", "humidity", "wind_speed", "rainfall", "drought_index"],
        "units": ["°C", "%", "km/h", "mm", "0-10"],
        "tip": "Hot, dry, windy conditions increase fire risk."
    },
    "cyclone": {
        "title": "Cyclone",
        "icon": "🌪️",
        "model": "cyclone_dataset_model(1).pkl",
        "features": ["sea_temp", "wind_speed", "pressure", "humidity"],
        "units": ["°C", "km/h", "hPa", "%"],
        "tip": "Warm sea, fast wind, and low pressure increase cyclone risk."
    },
    "drought": {
        "title": "Drought",
        "icon": "🌵",
        "model": "drought_dataset_model.pkl",
        "features": ["temperature", "rainfall", "soil_moisture", "evaporation"],
        "units": ["°C", "mm", "%", "mm/day"],
        "tip": "High temperature, low rainfall, low soil moisture, and high evaporation increase drought risk."
    }
}


def load_model(disaster):
    model_path = os.path.join(MODEL_DIR, DISASTERS[disaster]["model"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as file:
        return pickle.load(file)


MODELS = {}

for disaster in DISASTERS:
    MODELS[disaster] = load_model(disaster)


def level_from_probability(prob):
    if prob >= 0.70:
        return "High Risk", "danger", "Disaster may occur. Take safety precautions."
    elif prob >= 0.40:
        return "Moderate Risk", "warning", "Conditions are risky. Monitor carefully."
    else:
        return "Low Risk", "safe", "Disaster is unlikely based on current input."


def make_prediction(disaster, values):
    model = MODELS[disaster]
    features = DISASTERS[disaster]["features"]

    input_df = pd.DataFrame([values], columns=features)

    # Classification models
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(model.predict(input_df)[0])
        label, level_class, message = level_from_probability(prob)

        return {
            "prediction": prediction,
            "probability": round(prob * 100, 2),
            "label": label,
            "level_class": level_class,
            "message": message
        }

    # Regression model for earthquake
    risk_score = float(model.predict(input_df)[0])

    if risk_score >= 0.70:
        label = "High Risk"
        level_class = "danger"
        message = "High earthquake risk detected. Take safety precautions."
    elif risk_score >= 0.40:
        label = "Moderate Risk"
        level_class = "warning"
        message = "Moderate earthquake risk. Monitor carefully."
    else:
        label = "Low Risk"
        level_class = "safe"
        message = "Low earthquake risk based on current input."

    return {
        "prediction": round(risk_score, 3),
        "probability": round(risk_score * 100, 2),
        "label": label,
        "level_class": level_class,
        "message": message
    }


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

    result = make_prediction(disaster, values)

    return jsonify({
        "disaster": DISASTERS[disaster]["title"],
        "features": features,
        "values": values,
        **result
    })


def old_style_prediction(disaster):
    features = DISASTERS[disaster]["features"]

    try:
        values = [float(request.form.get(feature)) for feature in features]
    except (TypeError, ValueError):
        return "Please enter valid numeric values for all fields."

    result = make_prediction(disaster, values)

    return (
        f"{DISASTERS[disaster]['title']} Prediction: "
        f"{result['probability']}% — {result['label']}"
    )


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


if __name__ == "__main__":
    app.run(debug=True)
