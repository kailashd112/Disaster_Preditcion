# AI Powered Disaster Prediction System

A GitHub-ready Flask project for predicting disaster risk for Flood, Earthquake, Wild Fire, Cyclone, and Drought.

## Features

- All-in-one premium dashboard
- Flask backend API
- Model `.pkl` files included
- Dataset CSV files included
- Risk probability and label
- Charts for prediction inputs
- Render deployment-ready

## Run in VS Code

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Deploy on Render

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

## Folder Structure

```text
app.py
requirements.txt
Procfile
train_models.py
templates/index.html
models/*.pkl
datasets/*.csv
```

## Disclaimer

This project is for education and portfolio use only. It is not an official emergency alert system.
