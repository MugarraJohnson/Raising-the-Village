"""
WorkMate App - Machine Learning Pipeline for Household Vulnerability Prediction (Scikit-learn Version)

This module handles the training and evaluation of machine learning models for
predicting household vulnerability levels using Annual Household Survey (AHS) data.
This version uses scikit-learn instead of TensorFlow for broader compatibility.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
import csv
from datetime import datetime


class VulnerabilityPredictorSK:
    """
    Scikit-learn based model for predicting household vulnerability levels.
    
    This class handles data preprocessing, model training, evaluation,
    and model persistence using scikit-learn algorithms.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metadata = {}
        self.data_dictionary = None
    
    def load_and_preprocess_data(self, data_path='../Datasets/DataScientist_01_Assessment.csv'):
        """
        Load and preprocess AHS data for training.
        
        Args:
            data_path (str): Path to the AHS dataset CSV file
            
        Returns:
            tuple: Preprocessed features and target variables
        """
        try:
            # Load AHS data from the Datasets folder
            data = pd.read_csv(data_path)
            print(f"Loaded {len(data)} records from {data_path}")
            
            # Also load the data dictionary for reference
            dictionary_path = '../Datasets/Dictionary.xlsx'
            if os.path.exists(dictionary_path):
                try:
                    data_dict = pd.read_excel(dictionary_path)
                    print(f"Loaded data dictionary with {len(data_dict)} entries")
                    self.data_dictionary = data_dict
                except Exception as e:
                    print(f"Could not load dictionary: {e}")
            
            # Create ProgressStatus feature based on income, consumption, and residues
            if 'HHIncome+Consumption+Residues/Day' in data.columns:
                data['ProgressStatus'] = pd.cut(
                    data['HHIncome+Consumption+Residues/Day'],
                    bins=[-float('inf'), 1.25, 1.77, 2.15, float('inf')],
                    labels=['Severely Struggling', 'Struggling', 'At Risk', 'On Track']
                )
            else:
                # Create synthetic ProgressStatus for demo
                data['ProgressStatus'] = np.random.choice(
                    ['Severely Struggling', 'Struggling', 'At Risk', 'On Track'],
                    size=len(data)
                )
            
            # Define features for the model
            feature_columns = [
                'HouseholdSize', 'HHIncome/Day', 'Consumption/Day', 
                'hhh_sex', 'hhh_read_write', 'Material_walls',
                'daily_meals', 'latrine_constructed', 'tippy_tap_available',
                'kitchen_house', 'bathroom_constructed'
            ]
            
            # Filter to available columns
            available_features = [col for col in feature_columns if col in data.columns]
            
            if len(available_features) < 5:
                print("Warning: Limited features available, using synthetic data")
                return self._create_synthetic_data()
            
            # Create target variable based on income per day
            if 'HHIncome+Consumption+Residues/Day' in data.columns:
                income_values = pd.to_numeric(data['HHIncome+Consumption+Residues/Day'], errors='coerce')
                data['VulnerabilityLevel'] = pd.cut(
                    income_values,
                    bins=[0, 1.25, 2.15, float('inf')],
                    labels=['High', 'Moderate', 'Low']
                )
            else:
                data['VulnerabilityLevel'] = np.random.choice(['High', 'Moderate', 'Low'], len(data))
            
            # Separate features and target
            X = data[available_features].copy()
            y = data['VulnerabilityLevel'].copy()
            
            # Clean the data
            X = X.fillna(0)  # Fill missing values
            y = y.dropna()   # Remove missing targets
            X = X.loc[y.index]  # Align features with valid targets
            
            self.feature_names = available_features
            print(f"Using {len(available_features)} features: {available_features}")
            
            return X, y
            
        except FileNotFoundError:
            print(f"Data file not found at {data_path}. Creating synthetic data for demo.")
            return self._create_synthetic_data()
        except Exception as e:
            print(f"Error loading data: {e}. Creating synthetic data for demo.")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self, n_samples=1000):
        """Create synthetic AHS data for demonstration purposes."""
        np.random.seed(42)
        
        # Generate synthetic household data
        data = {
            'HouseholdSize': np.random.randint(1, 12, n_samples),
            'Income': np.random.lognormal(10, 0.5, n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Education': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], n_samples),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'WaterAccess': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'ElectricityAccess': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'HealthcareAccess': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2])
        }
        
        X = pd.DataFrame(data)
        
        # Create target based on logical rules
        vulnerability_score = (
            (X['HouseholdSize'] > 6).astype(int) * 2 +
            (X['Income'] < X['Income'].median()).astype(int) * 3 +
            (X['WaterAccess'] == 'No').astype(int) * 2 +
            (X['ElectricityAccess'] == 'No').astype(int) * 1 +
            np.random.normal(0, 1, n_samples)
        )
        
        y = pd.cut(vulnerability_score, bins=3, labels=['Low', 'Moderate', 'High'])
        
        self.feature_names = list(X.columns)
        print(f"Created synthetic dataset with {n_samples} samples and {len(self.feature_names)} features")
        
        return X, y
    
    def preprocess_features(self, X, y):
        """
        Preprocess features for machine learning.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            
        Returns:
            tuple: Preprocessed features and targets
        """
        # Identify numerical and categorical columns
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create preprocessing pipeline
        preprocessor_steps = []
        
        if numerical_features:
            preprocessor_steps.append(('num', StandardScaler(), numerical_features))
        
        if categorical_features:
            preprocessor_steps.append(('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features))
        
        if preprocessor_steps:
            self.preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='passthrough'
            )
        else:
            # If no preprocessing needed, create identity transformer
            from sklearn.preprocessing import FunctionTransformer
            self.preprocessor = FunctionTransformer()
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit and transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y_encoded
    
    def train_model(self, X, y, test_size=0.2, model_type='random_forest'):
        """
        Train the vulnerability prediction model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            test_size (float): Proportion of data for testing
            model_type (str): Type of model to train
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Training {model_type} model for vulnerability prediction...")
        
        # Preprocess the data
        X_processed, y_encoded = self.preprocess_features(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        print(f"Training on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_test_pred, target_names=target_names)
        print("Classification Report:")
        print(report)
        
        # Store metadata
        self.model_metadata = {
            'model_type': model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_features': X_processed.shape[1],
            'n_samples': X_processed.shape[0],
            'feature_names': self.feature_names,
            'target_classes': list(target_names),
            'training_date': datetime.now().isoformat()
        }
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'model_metadata': self.model_metadata
        }
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.array: Predicted vulnerability levels
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Preprocess the input
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions
        y_pred_encoded = self.model.predict(X_processed)
        
        # Convert back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def save_model(self, model_dir='../models'):
        """
        Save the trained model and preprocessors.
        
        Args:
            model_dir (str): Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, os.path.join(model_dir, 'vulnerability_model.pkl'))
        joblib.dump(self.preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        
        # Save metadata
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='../models'):
        """
        Load a trained model and preprocessors.
        
        Args:
            model_dir (str): Directory containing the saved model
        """
        try:
            self.model = joblib.load(os.path.join(model_dir, 'vulnerability_model.pkl'))
            self.preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            
            # Load metadata
            with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
                self.model_metadata = json.load(f)
            
            print(f"Model loaded from {model_dir}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found in {model_dir}: {e}")
    
    def read_datasets_from_static(self, static_folder='static'):
        """
        Read all datasets from the static folder.
        
        Args:
            static_folder (str): Path to the static folder containing datasets
            
        Returns:
            dict: Dictionary containing loaded datasets
        """
        datasets = {}
        
        # Create static folder if it doesn't exist
        if not os.path.exists(static_folder):
            print(f"Creating {static_folder} folder...")
            os.makedirs(static_folder)
            print(f"Static folder created at: {os.path.abspath(static_folder)}")
            return datasets
        
        # Get list of files in static folder
        try:
            files = os.listdir(static_folder)
            print(f"Found {len(files)} files in {static_folder} folder:")
            
            for file in files:
                file_path = os.path.join(static_folder, file)
                print(f"  - {file}")
                
                # Read different file types
                if file.endswith('.csv'):
                    try:
                        datasets[file] = pd.read_csv(file_path)
                        print(f"    ✓ Loaded CSV with {len(datasets[file])} rows, {len(datasets[file].columns)} columns")
                    except Exception as e:
                        print(f"    ✗ Error loading CSV: {e}")
                        
                elif file.endswith(('.xlsx', '.xls')):
                    try:
                        datasets[file] = pd.read_excel(file_path)
                        print(f"    ✓ Loaded Excel with {len(datasets[file])} rows, {len(datasets[file].columns)} columns")
                    except Exception as e:
                        print(f"    ✗ Error loading Excel: {e}")
                        
                elif file.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            datasets[file] = json.load(f)
                        print(f"    ✓ Loaded JSON with {len(datasets[file])} entries")
                    except Exception as e:
                        print(f"    ✗ Error loading JSON: {e}")
                else:
                    print(f"    - Unsupported file type")
            
            return datasets
            
        except Exception as e:
            print(f"Error reading static folder: {e}")
            return datasets


def main():
    """Main function to demonstrate the WorkMate ML pipeline."""
    print("🏠 WorkMate ML Pipeline - Scikit-learn Version")
    print("="*50)
    
    # Initialize predictor
    predictor = VulnerabilityPredictorSK()
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    X, y = predictor.load_and_preprocess_data()
    
    # Train model
    print("\n2. Training vulnerability prediction model...")
    results = predictor.train_model(X, y, model_type='random_forest')
    
    # Save model
    print("\n3. Saving model...")
    predictor.save_model()
    
    # Test static folder reading
    print("\n4. Testing static folder integration...")
    static_datasets = predictor.read_datasets_from_static('static')
    
    print("\n" + "="*50)
    print("🎉 WorkMate ML Pipeline completed successfully!")
    print(f"Model accuracy: {results['test_accuracy']:.4f}")
    print(f"Features used: {len(predictor.feature_names)}")
    print(f"Static datasets: {len(static_datasets)}")


if __name__ == "__main__":
    main()
