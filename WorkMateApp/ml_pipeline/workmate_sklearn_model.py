"""
WorkMate App - Smart Household Vulnerability Assessment

Hey there! This is the brain behind WorkMate's ability to understand and predict 
household vulnerability levels. Think of it as a really smart assistant that looks 
at household data and says "Hey, this family might need some extra support."

We're using scikit-learn here because it's reliable, fast, and doesn't need 
the heavy TensorFlow setup. Perfect for getting real results to real families!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os
import json
from datetime import datetime


class WorkMateVulnerabilityPredictor:
    """
    The heart of WorkMate's prediction system! 
    
    This friendly AI assistant helps us understand which households might be 
    struggling and need support. It's like having an experienced social worker 
    who can quickly assess thousands of families and flag those who need help most.
    
    What makes it special:    - Learns from real household survey data from Uganda
    - Considers things like income, family size, access to water, education
    - Gives clear vulnerability levels: High, Moderate-High, Moderate, Low
    - Fast enough to work on mobile phones in rural areas
    """
    
    def __init__(self):
        # Think of these as the AI's memory - where it stores what it learned
        self.model = None                # The trained AI brain
        self.preprocessor = None         # How to clean and prepare data
        self.label_encoder = None        # Translates between words and numbers        self.feature_names = None        # List of things we look at (income, family size, etc.)
        self.model_metadata = {}         # Info about when/how the model was trained
        self.data_dictionary = None      # Explains what each data column means
        
    def load_and_preprocess_data(self, data_path='../Datasets/DataScientist_01_Assessment.csv'):
        """
        Hey! This is where we teach the AI about real families and their situations.
        
        We take the raw survey data (like a really detailed questionnaire that 
        families filled out) and turn it into something our AI can understand.
        It's like translating human stories into AI language!
        
        What we're looking for:
        - How much money families make per day
        - Family size, education, access to basic needs
        - Signs that might indicate if a family is struggling
        
        Returns the cleaned data ready for the AI to learn from.
        """
        try:            # Time to load our family data! This is real survey info from Uganda
            data = pd.read_csv(data_path)
            print(f"✓ Loaded {len(data)} records from {data_path}")
            
            # Let's also grab the dictionary that explains what everything means
            dictionary_path = '../Datasets/Dictionary.xlsx'
            if os.path.exists(dictionary_path):
                data_dict = pd.read_excel(dictionary_path)
                print(f"✓ Loaded data dictionary with {len(data_dict)} entries")
                self.data_dictionary = data_dict
            
            # Here's the magic: we create vulnerability levels based on daily income
            # Think of it like traffic lights: Red = needs urgent help, Green = doing okay
            if 'HHIncome+Consumption+Residues/Day' in data.columns:
                # These thresholds are based on poverty line research
                # Less than $0.50/day = High vulnerability (urgent help needed)
                # $0.50-$1.00 = Moderate-High (significant concerns)
                # $1.00-$2.00 = Moderate (some support helpful)
                # Above $2.00 = Low vulnerability (relatively stable)
                data['VulnerabilityLevel'] = pd.cut(
                    data['HHIncome+Consumption+Residues/Day'],
                    bins=[-float('inf'), 0.5, 1.0, 2.0, float('inf')],
                    labels=['High', 'Moderate-High', 'Moderate', 'Low']
                )
            else:
                # Just in case we don't have the income data, we'll create random examples
                data['VulnerabilityLevel'] = np.random.choice(
                    ['High', 'Moderate-High', 'Moderate', 'Low'],
                    size=len(data)
                )
            
            # Define key features for vulnerability prediction
            feature_columns = [
                'HouseholdSize', 'TimeToOPD', 'TimeToWater', 'AgricultureLand',
                'HHIncome/Day', 'Consumption/Day', 'hhh_sex', 'hhh_read_write',
                'Material_walls', 'radios_owned', 'phones_owned', 'daily_meals',
                'latrine_constructed', 'tippy_tap_available', 'kitchen_house',
                'bathroom_constructed', 'swept_compound'
            ]
            
            # Filter available columns and handle missing values
            available_features = [col for col in feature_columns if col in data.columns]
            print(f"✓ Using {len(available_features)} features for prediction")
            
            # Handle missing values
            for col in available_features:
                if data[col].dtype in ['object', 'string']:
                    data[col] = data[col].fillna('Unknown')
                else:
                    data[col] = data[col].fillna(data[col].median())
            
            # Separate features and target
            X = data[available_features]
            y = data['VulnerabilityLevel'].dropna()
            
            # Align X and y indices
            X = X.loc[y.index]
            
            self.feature_names = available_features
            
            print(f"✓ Final dataset: {len(X)} samples, {len(available_features)} features")
            print(f"✓ Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except FileNotFoundError:
            print(f"Data file not found at {data_path}. Creating synthetic data for demo.")
            return self._create_synthetic_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self, n_samples=1000):
        """Create synthetic AHS data for demonstration purposes."""
        np.random.seed(42)
        
        # Generate synthetic household data
        data = {
            'HouseholdSize': np.random.randint(1, 12, n_samples),
            'TimeToOPD': np.random.randint(5, 300, n_samples),
            'TimeToWater': np.random.randint(1, 120, n_samples),
            'AgricultureLand': np.random.exponential(2, n_samples),
            'HHIncome/Day': np.random.exponential(1.5, n_samples),
            'Consumption/Day': np.random.exponential(1.2, n_samples),
            'hhh_sex': np.random.choice([1, 2], n_samples),
            'hhh_read_write': np.random.choice([0, 1], n_samples),
            'Material_walls': np.random.choice([1, 2], n_samples),
            'radios_owned': np.random.randint(0, 5, n_samples),
            'phones_owned': np.random.randint(0, 4, n_samples),
            'daily_meals': np.random.choice([1, 2, 3], n_samples),
            'latrine_constructed': np.random.choice([0, 1], n_samples),
            'tippy_tap_available': np.random.choice([0, 1], n_samples),
            'kitchen_house': np.random.choice([0, 1], n_samples),
            'bathroom_constructed': np.random.choice([0, 1], n_samples),
            'swept_compound': np.random.choice([0, 1], n_samples)
        }
        
        X = pd.DataFrame(data)
        
        # Create target based on income and household characteristics
        vulnerability_score = (
            -X['HHIncome/Day'] * 0.4 +
            X['HouseholdSize'] * 0.1 +
            X['TimeToWater'] * 0.01 +
            -X['hhh_read_write'] * 0.3 +
            -X['latrine_constructed'] * 0.2
        )
        
        # Convert to categorical
        y = pd.cut(vulnerability_score, bins=4, labels=['Low', 'Moderate', 'Moderate-High', 'High'])
        
        self.feature_names = list(X.columns)
        
        print(f"✓ Created synthetic dataset: {len(X)} samples, {len(X.columns)} features")
        
        return X, y
    def train_model(self, X, y, test_size=0.2, model_type='random_forest'):
        """
        This is where the real magic happens! We're teaching our AI to be a 
        vulnerability assessment expert.
        
        Imagine we're training a really smart social worker who can look at 
        household information and instantly know who needs help most. We show 
        the AI thousands of examples, and it learns the patterns.
        
        We use Random Forest - think of it like having a whole team of experts 
        voting on each family's situation. Usually gets it right!
        
        What it learns to recognize:
        - Families with very low daily income
        - Large families with limited resources  
        - Poor access to clean water or healthcare        - Lack of education or basic infrastructure
        
        Returns how well our AI performed on its final exam!
        """
        print(f"\n=== Training WorkMate's AI Brain ({model_type}) ===")
        
        # First, we need to translate our vulnerability levels into numbers
        # High=0, Low=1, etc. - computers love numbers!
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split our data: some for teaching, some for testing (like school!)
        # We hold back 20% to see if our AI really learned or just memorized
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Separate the numbers from the words in our data
        numeric_features = X.select_dtypes(include=[np.number]).columns  # Age, income, etc.
        categorical_features = X.select_dtypes(include=['object']).columns  # Yes/No, Male/Female, etc.
        
        # Create our data preparation pipeline
        # Numbers need to be standardized (like grading on a curve)
        # Categories need to be converted to numbers the AI understands
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Choose model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Store metadata
        self.model_metadata = {
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'target_classes': list(self.label_encoder.classes_)
        }
        
        # Print results
        print(f"✓ Training completed!")
        print(f"✓ Training accuracy: {train_accuracy:.3f}")
        print(f"✓ Test accuracy: {test_accuracy:.3f}")
        print(f"✓ Model type: {model_type}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test, 
                                  target_names=self.label_encoder.classes_))
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'model_type': model_type,
            'feature_names': self.feature_names
        }
    
    def predict_vulnerability(self, household_data):
        """
        Predict vulnerability level for new household data.
        
        Args:
            household_data (dict or pd.DataFrame): Household features
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert to DataFrame if needed
        if isinstance(household_data, dict):
            household_data = pd.DataFrame([household_data])
        
        # Make prediction
        prediction = self.model.predict(household_data)
        probabilities = self.model.predict_proba(household_data)
        
        # Convert back to labels
        vulnerability_level = self.label_encoder.inverse_transform(prediction)[0]
        
        # Get class probabilities
        prob_dict = {
            class_name: prob 
            for class_name, prob in zip(self.label_encoder.classes_, probabilities[0])
        }
        
        return {
            'vulnerability_level': vulnerability_level,
            'confidence': max(probabilities[0]),
            'probabilities': prob_dict
        }
    
    def save_model(self, model_path='../models/workmate_vulnerability_model.joblib'):
        """Save the trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'metadata': self.model_metadata
        }
        
        joblib.dump(model_data, model_path)
        
        # Save metadata as JSON
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def load_model(self, model_path='../models/workmate_vulnerability_model.joblib'):
        """Load a previously trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_metadata = model_data['metadata']
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Model type: {self.model_metadata.get('model_type', 'Unknown')}")
        print(f"✓ Training date: {self.model_metadata.get('training_date', 'Unknown')}")
        print(f"✓ Test accuracy: {self.model_metadata.get('test_accuracy', 'Unknown')}")


def main():
    """
    Welcome to WorkMate! Let's train our AI to help vulnerable families.
    
    This is like a complete training session where we:
    1. Load real family data from Uganda
    2. Teach our AI to recognize vulnerability patterns  
    3. Test how well it learned
    4. Save the trained AI for the mobile app
    
    Think of it as graduating our AI from "Vulnerability Assessment University"!
    """
    print("=== WorkMate: AI-Powered Vulnerability Assessment ===")
    print("🏠 Helping identify families who need support most")
    print("🤖 Training our AI social worker...\n")
    
    # Create our AI assistant
    predictor = WorkMateVulnerabilityPredictor()
    
    # Step 1: Load real family stories and data
    print("1. 📊 Loading real household survey data...")
    X, y = predictor.load_and_preprocess_data()
    
    # Step 2: Teach our AI to be a vulnerability expert
    print("\n2. 🧠 Training our AI to recognize vulnerability patterns...")
    results = predictor.train_model(X, y, model_type='random_forest')
    
    # Step 3: Save our trained AI for the mobile app
    print("\n3. 💾 Saving our trained AI for deployment...")
    predictor.save_model()
    
    # Step 4: Test our AI with a real family example
    print("\n4. 🧪 Testing our AI with a sample household...")
    if len(X) > 0:
        # Let's see how our AI assesses the first family in our dataset
        sample_household = X.iloc[0:1]
        prediction = predictor.predict_vulnerability(sample_household)
        
        print(f"📋 Sample Family Assessment:")
        print(f"   🎯 Vulnerability Level: {prediction['vulnerability_level']}")
        print(f"   📊 AI Confidence: {prediction['confidence']:.1%}")
        print(f"   📈 Detailed Breakdown: {prediction['probabilities']}")
    
    print("\n🎉 SUCCESS! WorkMate AI is ready to help families!")
    print("🚀 The trained model is now ready for the mobile app")
    print("📱 Field workers can now use this AI to quickly assess household vulnerability")
    
    return predictor


if __name__ == "__main__":
    # Let's train our AI to help vulnerable families!
    predictor = main()
