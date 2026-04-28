"""Optional helper to regenerate model .pkl files used by app.py.
Run: python train_models.py
"""
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

models = {
    "flood_model.pkl": {"bias": -7.2, "weights": [0.020, 0.70, 0.020, 0.010, 0.030]},
    "earthquake_model.pkl": {"bias": -6.4, "weights": [0.95, -0.012, 0.000, 0.000, 1.60]},
    "fire_model.pkl": {"bias": -7.0, "weights": [0.11, -0.045, 0.075, -0.055, 0.31]},
    "cyclone_model.pkl": {"bias": -9.2, "weights": [0.18, 0.040, -0.004, 0.012]},
    "drought_model.pkl": {"bias": -6.9, "weights": [0.14, -0.055, -0.060, 0.075]},
}

for filename, config in models.items():
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(config, f)

print("Model files generated successfully.")
