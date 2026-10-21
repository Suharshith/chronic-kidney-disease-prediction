"""
Unit Tests for CKD Prediction Tabular Model Pipeline
Validates model artifact integrity, column ordering, and prediction behavior.
"""
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABULAR_MODEL_PATH = os.path.join(BASE_DIR, "tabular_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

def test_model_artifacts_exist():
    """Verify essential tabular model files exist."""
    assert os.path.exists(TABULAR_MODEL_PATH), f"Missing tabular model at {TABULAR_MODEL_PATH}"
    assert os.path.exists(COLUMNS_PATH), f"Missing columns specification at {COLUMNS_PATH}"

def test_load_feature_columns():
    """Verify that model_columns.pkl contains valid feature names."""
    with open(COLUMNS_PATH, "rb") as f:
        columns = pickle.load(f)
    assert isinstance(columns, (list, np.ndarray, pd.Index))
    assert len(columns) == 24
    expected_sample = {"age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo"}
    assert expected_sample.issubset(set(columns))

def test_tabular_model_inference():
    """Test end-to-end inference on a synthetic patient record."""
    model = joblib.load(TABULAR_MODEL_PATH)
    with open(COLUMNS_PATH, "rb") as f:
        columns = list(pickle.load(f))
    
    # Construct a sample patient dataframe with 24 features
    sample_data = {col: [0.0] for col in columns}
    sample_data["age"] = [55.0]
    sample_data["bp"] = [80.0]
    sample_data["sg"] = [1.020]
    sample_data["al"] = [0.0]
    sample_data["bgr"] = [120.0]
    sample_data["bu"] = [36.0]
    sample_data["sc"] = [1.2]
    sample_data["hemo"] = [15.4]
    
    df = pd.DataFrame(sample_data)
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)
    
    assert len(prediction) == 1
    assert prediction[0] in ["ckd", "notckd"]
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)
    print(f"Verified Prediction: {prediction[0]}, Probabilities: {probabilities[0]}")
