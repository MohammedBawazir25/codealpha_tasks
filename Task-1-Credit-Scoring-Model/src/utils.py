"""
utils.py
---------
Common utility functions for the Credit Scoring Project.
"""

from pathlib import Path
import json
from datetime import datetime

import joblib
import pandas as pd


# ==========================================================
# DIRECTORY UTILITIES
# ==========================================================

def create_directory(path):
    """
    Create directory if it doesn't exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# ==========================================================
# MODEL UTILITIES
# ==========================================================

def save_model(model, filepath):
    """
    Save trained model.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)

    print(f"Model saved to:\n{filepath}")


def load_model(filepath):
    """
    Load trained model.
    """
    filepath = Path(filepath)

    model = joblib.load(filepath)

    print(f"Model loaded from:\n{filepath}")

    return model


# ==========================================================
# SCALER UTILITIES
# ==========================================================

def save_scaler(scaler, filepath):
    """
    Save scaler.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, filepath)

    print(f"Scaler saved to:\n{filepath}")


def load_scaler(filepath):
    """
    Load scaler.
    """
    filepath = Path(filepath)

    scaler = joblib.load(filepath)

    print(f"Scaler loaded from:\n{filepath}")

    return scaler


# ==========================================================
# DATAFRAME UTILITIES
# ==========================================================

def save_dataframe(df, filepath):
    """
    Save DataFrame as CSV.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)

    print(f"CSV saved to:\n{filepath}")


def load_dataframe(filepath):
    """
    Load CSV into DataFrame.
    """
    return pd.read_csv(filepath)


# ==========================================================
# JSON UTILITIES
# ==========================================================

def save_json(data, filepath):
    """
    Save dictionary as JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


def load_json(filepath):
    """
    Load JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


# ==========================================================
# REPORT UTILITIES
# ==========================================================

def save_model_comparison(results, filepath):
    """
    Save model comparison DataFrame.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(filepath, index=False)

    print(f"Model comparison saved to:\n{filepath}")


# ==========================================================
# TIME UTILITIES
# ==========================================================

def current_timestamp():
    """
    Return current timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================
# DISPLAY UTILITIES
# ==========================================================

def print_heading(text):
    """
    Print formatted heading.
    """
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_success(message):
    """
    Print success message.
    """
    print(f"[SUCCESS] {message}")


def print_error(message):
    """
    Print error message.
    """
    print(f"[ERROR] {message}")


# ==========================================================
# DATA INFORMATION
# ==========================================================

def dataset_info(df):
    """
    Display dataset information.
    """
    print_heading("Dataset Information")

    print(f"Shape : {df.shape}")

    print("\nColumns:")

    for column in df.columns:
        print(f"• {column}")

    print("\nMissing Values")

    print(df.isnull().sum())


# ==========================================================
# EXPORT RESULTS
# ==========================================================

def export_metrics(metrics, filepath):
    """
    Save metrics dictionary to CSV.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])

    df.to_csv(filepath, index=False)

    print(f"Metrics exported to:\n{filepath}")