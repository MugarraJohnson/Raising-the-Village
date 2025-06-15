#!/usr/bin/env python3
"""
WorkMate Model Test Script - Verify Deployment Status

This script verifies that the WorkMate model is properly deployed and ready for use.
"""

import sys
from pathlib import Path

# Add ml_pipeline to path
sys.path.append(str(Path(__file__).parent / 'ml_pipeline'))

from workmate_sklearn_model import WorkMateVulnerabilityPredictor
import json

def main():
    print("=== WorkMate Model Deployment Verification ===\n")
    
    # Check if model files exist
    model_path = Path("models/workmate_vulnerability_model.joblib")
    metadata_path = Path("models/workmate_vulnerability_model_metadata.json")
    
    print("1. Checking model files...")
    if model_path.exists():
        print(f"   ✓ Model file found: {model_path}")
    else:
        print(f"   ❌ Model file missing: {model_path}")
        return False
    
    if metadata_path.exists():
        print(f"   ✓ Metadata file found: {metadata_path}")
    else:
        print(f"   ❌ Metadata file missing: {metadata_path}")
        return False
    
    # Load and display metadata
    print("\n2. Model Information:")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"   • Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"   • Training Date: {metadata.get('training_date', 'Unknown')}")
    print(f"   • Test Accuracy: {metadata.get('test_accuracy', 0):.1%}")
    print(f"   • Features: {metadata.get('n_features', 0)}")
    print(f"   • Training Samples: {metadata.get('n_samples', 0)}")
    
    # Test model loading
    print("\n3. Testing model loading...")
    try:
        predictor = WorkMateVulnerabilityPredictor()
        predictor.load_model(str(model_path))
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Test prediction
    print("\n4. Testing prediction...")
    test_household = {
        'HouseholdSize': 5,
        'TimeToOPD': 30,
        'TimeToWater': 10,
        'AgricultureLand': 1.0,
        'HHIncome/Day': 1.5,
        'Consumption/Day': 1.2,
        'hhh_sex': 1,
        'hhh_read_write': 1,
        'Material_walls': 1,
        'radios_owned': 1,
        'phones_owned': 1,
        'daily_meals': 3,
        'latrine_constructed': 1,
        'tippy_tap_available': 1,
        'kitchen_house': 1,
        'bathroom_constructed': 1,
        'swept_compound': 1
    }
    
    try:
        result = predictor.predict_vulnerability(test_household)
        print(f"   ✓ Prediction successful: {result['vulnerability_level']} ({result['confidence']:.1%})")
        print(f"   • Predicted Level: {result['vulnerability_level']}")
        print(f"   • Confidence: {result['confidence']:.1%}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False
    
    print("\n=== DEPLOYMENT STATUS: SUCCESS ✓ ===")
    print("🎉 WorkMate model is ready for deployment!")
    print("\nQuick Usage Commands:")
    print("• python quick_predict.py --example")
    print("• python quick_predict.py --interactive")
    print("• cd deployment && python flask_api.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
