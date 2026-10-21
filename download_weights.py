"""
Weights & Dataset Setup Helper for CKD Prediction Application
Guides local verification and download instructions for the 228.5 MB CNN Model.
"""
import os
import sys

CNN_MODEL_NAME = "cnn_model.h5"
EXPECTED_SIZE_BYTES = 228509248  # ~228.5 MB

def verify_artifacts():
    print("=" * 60)
    print("CKD Prediction Model Artifact Verification")
    print("=" * 60)
    
    files = {
        "tabular_model.joblib": "Scikit-learn Random Forest Model (Tabular)",
        "model_columns.pkl": "Feature Column Order Specification",
        CNN_MODEL_NAME: "Keras 4-Class Renal CT Image Classifier"
    }
    
    for filename, desc in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"[OK] {filename:<22} ({size:,} bytes) - {desc}")
        else:
            if filename == CNN_MODEL_NAME:
                print(f"[MISSING] {filename:<22} - {desc}")
                print("   Note: cnn_model.h5 exceeds GitHub\'s 100 MB limit.")
                print("   If you have trained weights locally, copy them to this directory.")
                print("   The tabular Random Forest + SHAP model runs independently without it.")
            else:
                print(f"[MISSING] {filename:<22} - {desc}")

if __name__ == "__main__":
    verify_artifacts()
