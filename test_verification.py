"""
End-to-End Verification Test Script for PLOS Rehabilitation Hospital Project.
"""

import os
import json
from src.prediction import predictor

def run_tests():
    print("==================================================")
    print(" TESTING PLOS APPLICATION & PREDICTION ENGINE")
    print("==================================================")
    
    # Test 1: Artifact Existence Check
    assert os.path.exists("models/plos_model.pkl"), "Model file missing!"
    assert os.path.exists("models/preprocessor.pkl"), "Preprocessor file missing!"
    assert os.path.exists("models/model_metrics.json"), "Metrics file missing!"
    print("[OK] Model artifacts check passed.")
    
    # Test 2: Sample Patient Test with STRING inputs as submitted from HTML forms
    form_string_input = {
        "age": "65",
        "gender": "Male",
        "diagnosis": "Stroke",
        "rehab_type": "Neurological",
        "admission_type": "Emergency",
        "previous_hospitalization": "Yes",
        "previous_surgery": "No",
        "severity": "High",
        "mobility_score": "35",
        "comorbidities": "3",
        "diabetes": "Yes",
        "hypertension": "Yes",
        "insurance_type": "Private",
        "functional_score": "40",
        "pain_level": "7",
        "therapy_frequency": "5",
        "previous_admissions": "2"
    }
    
    # Validate input
    errors = predictor.validate_input(form_string_input)
    assert len(errors) == 0, f"Validation failed: {errors}"
    print("[OK] String input validation passed.")
    
    # Run prediction
    result = predictor.predict(form_string_input)
    print("\n--- SAMPLE PATIENT PREDICTION RESULT (FORM STRING TEST) ---")
    print(f" -> Estimated Length of Stay: {result['predicted_days']} Days")
    print(f" -> Stay Category: {result['stay_category']}")
    print(f" -> Risk Level: {result['risk_level']}")
    print(f" -> Factors: {result['factors']}")
    print("----------------------------------------------------------")
    
    assert result["predicted_days"] > 0
    assert result["stay_category"] in ["Short Stay", "Medium Stay", "Long Stay"]
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    print("[OK] Prediction engine with string form inputs test passed successfully.")
    print("==================================================\n")

if __name__ == "__main__":
    run_tests()
