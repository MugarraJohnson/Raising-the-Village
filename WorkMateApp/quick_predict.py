#!/usr/bin/env python3
"""
WorkMate Quick Prediction Tool - Terminal Interface

Run this script directly from the terminal to make quick vulnerability predictions.
Usage: python quick_predict.py [--interactive] [--batch input.csv]
"""

import sys
import json
import argparse
from pathlib import Path

# Add ml_pipeline to path
sys.path.append(str(Path(__file__).parent / 'ml_pipeline'))

from workmate_sklearn_model import WorkMateVulnerabilityPredictor
import pandas as pd


def load_model():
    """Load the trained model."""
    print("🤖 Loading WorkMate AI model...")
    predictor = WorkMateVulnerabilityPredictor()
    try:
        # Try different model paths
        model_path = Path(__file__).parent / 'models' / 'workmate_vulnerability_model.joblib'
        predictor.load_model(str(model_path))
        print("✓ Model loaded successfully!")
        return predictor
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first:")
        print("   cd ml_pipeline && python workmate_sklearn_model.py")
        sys.exit(1)


def interactive_mode(predictor):
    """Interactive prediction mode."""
    print("\n=== WorkMate Interactive Prediction ===")
    print("Enter household information (press Enter for default values):")
    
    # Default values based on typical household
    defaults = {
        'HouseholdSize': 5,
        'TimeToOPD': 60,
        'TimeToWater': 20,
        'AgricultureLand': 1.0,
        'HHIncome/Day': 1.0,
        'Consumption/Day': 0.8,
        'hhh_sex': 1,
        'hhh_read_write': 1,
        'Material_walls': 1,
        'radios_owned': 1,
        'phones_owned': 1,
        'daily_meals': 2,
        'latrine_constructed': 1,
        'tippy_tap_available': 0,
        'kitchen_house': 1,
        'bathroom_constructed': 0,
        'swept_compound': 1
    }
    
    household = {}
    
    for feature, default in defaults.items():
        while True:
            try:
                value = input(f"{feature} (default: {default}): ").strip()
                if not value:
                    household[feature] = default
                    break
                household[feature] = float(value) if '.' in value else int(value)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Make prediction
    print("\n🔍 Analyzing household...")
    result = predictor.predict_vulnerability(household)
    
    # Display results
    print(f"\n📊 VULNERABILITY ASSESSMENT RESULTS")
    print(f"{'='*50}")
    print(f"🎯 Vulnerability Level: {result['vulnerability_level']}")
    print(f"📈 AI Confidence: {result['confidence']:.1%}")
    print(f"\n📋 Detailed Breakdown:")
    for level, prob in result['probabilities'].items():
        bar = '█' * int(prob * 20)
        print(f"   {level:12}: {prob:.1%} {bar}")
    
    # Interpretation
    level = result['vulnerability_level']
    if level == 'High':
        print(f"\n🚨 URGENT: This household needs immediate support!")
    elif level == 'Moderate-High':
        print(f"\n⚠️  ATTENTION: This household has significant vulnerabilities.")
    elif level == 'Moderate':
        print(f"\n💙 SUPPORT: This household could benefit from assistance.")
    else:
        print(f"\n✅ STABLE: This household appears relatively stable.")


def batch_mode(predictor, input_file, output_file=None):
    """Batch prediction mode from CSV file."""
    print(f"\n📁 Processing batch predictions from {input_file}...")
    
    try:
        data = pd.read_csv(input_file)
        print(f"✓ Loaded {len(data)} households from {input_file}")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        sys.exit(1)
    
    # Make predictions
    predictions = []
    confidences = []
    
    for idx, row in data.iterrows():
        try:
            result = predictor.predict_vulnerability(row.to_dict())
            predictions.append(result['vulnerability_level'])
            confidences.append(result['confidence'])
        except Exception as e:
            print(f"⚠️  Error processing row {idx}: {e}")
            predictions.append('Error')
            confidences.append(0.0)
    
    # Add predictions to dataframe
    data['predicted_vulnerability'] = predictions
    data['prediction_confidence'] = confidences
    
    # Save results
    if output_file is None:
        output_file = input_file.replace('.csv', '_predictions.csv')
    
    data.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to: {output_file}")
    
    # Summary
    summary = pd.Series(predictions).value_counts()
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"{'='*30}")
    for level, count in summary.items():
        if level != 'Error':
            print(f"{level:12}: {count} households ({count/len(data)*100:.1f}%)")


def quick_example(predictor):
    """Show a quick example prediction."""
    print("\n🧪 Quick Example - Typical Rural Household:")
    
    example_household = {
        'HouseholdSize': 6,
        'TimeToOPD': 45,
        'TimeToWater': 15,
        'AgricultureLand': 1.2,
        'HHIncome/Day': 0.75,
        'Consumption/Day': 0.65,
        'hhh_sex': 1,
        'hhh_read_write': 1,
        'Material_walls': 1,
        'radios_owned': 1,
        'phones_owned': 0,
        'daily_meals': 2,
        'latrine_constructed': 1,
        'tippy_tap_available': 0,
        'kitchen_house': 1,
        'bathroom_constructed': 0,
        'swept_compound': 1
    }
    
    result = predictor.predict_vulnerability(example_household)
    
    print(f"📋 Household Profile:")
    print(f"   • Family Size: {example_household['HouseholdSize']} members")
    print(f"   • Daily Income: ${example_household['HHIncome/Day']:.2f}")
    print(f"   • Water Access: {example_household['TimeToWater']} minutes")
    print(f"   • Has Latrine: {'Yes' if example_household['latrine_constructed'] else 'No'}")
    
    print(f"\n🎯 AI Assessment: {result['vulnerability_level']} ({result['confidence']:.1%} confidence)")


def main():
    parser = argparse.ArgumentParser(description='WorkMate Quick Prediction Tool')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--batch', '-b', type=str, 
                       help='Process CSV file in batch mode')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output file for batch mode (optional)')
    parser.add_argument('--example', '-e', action='store_true',
                       help='Show a quick example prediction')
    
    args = parser.parse_args()
    
    # Load model
    predictor = load_model()
    
    if args.interactive:
        interactive_mode(predictor)
    elif args.batch:
        batch_mode(predictor, args.batch, args.output)
    elif args.example:
        quick_example(predictor)
    else:
        print("\n🚀 WorkMate Quick Prediction Tool")
        print("Choose an option:")
        print("  --interactive  (-i) : Interactive household assessment")
        print("  --batch file   (-b) : Process CSV file")
        print("  --example      (-e) : Show example prediction")
        print("\nExamples:")
        print("  python quick_predict.py --interactive")
        print("  python quick_predict.py --batch households.csv")
        print("  python quick_predict.py --example")


if __name__ == "__main__":
    main()
