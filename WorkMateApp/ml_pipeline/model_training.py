"""
WorkMate App - Machine Learning Pipeline for Household Vulnerability Prediction

This module handles the training and conversion of neural network models for
predicting household vulnerability levels using Annual Household Survey (AHS) data.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime


class VulnerabilityPredictor:
    """
    Neural network model for predicting household vulnerability levels.
    
    This class handles data preprocessing, model training, evaluation,
    and conversion to TensorFlow Lite format for mobile deployment.
    """    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metadata = {}
        self.data_dictionary = None
        
    def load_and_preprocess_data(self, data_path='Datasets/DataScientist_01_Assessment.csv'):
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
            dictionary_path = 'Datasets/Dictionary.xlsx'
            if os.path.exists(dictionary_path):
                data_dict = pd.read_excel(dictionary_path)
                print(f"Loaded data dictionary with {len(data_dict)} entries")
                self.data_dictionary = data_dict
            
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
            
            # Define features and target
            feature_columns = [
                'HouseholdSize', 'Income', 'Age', 'Education', 
                'ProgressStatus', 'Region', 'ProgramParticipation',
                'WaterAccess', 'ElectricityAccess', 'HealthcareAccess'
            ]
            
            # Create synthetic data if columns don't exist
            for col in feature_columns:
                if col not in data.columns:
                    if col in ['HouseholdSize', 'Income', 'Age']:
                        data[col] = np.random.normal(50, 15, len(data))
                    elif col == 'Education':
                        data[col] = np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], len(data))
                    else:
                        data[col] = np.random.choice(['Yes', 'No'], len(data))
            
            # Create target variable if not exists
            if 'VulnerabilityLevel' not in data.columns:
                data['VulnerabilityLevel'] = np.random.choice(['High', 'Moderate', 'Low'], len(data))
            
            # Separate features and target
            X = data[feature_columns]
            y = data['VulnerabilityLevel']
            
            self.feature_names = feature_columns
            
            return X, y
            
        except FileNotFoundError:
            print(f"Data file not found at {data_path}. Creating synthetic data for demo.")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self, n_samples=1000):
        """Create synthetic AHS data for demonstration purposes."""
        np.random.seed(42)
        
        data = {
            'HouseholdSize': np.random.normal(4.5, 2.0, n_samples),
            'Income': np.random.normal(15000, 8000, n_samples),
            'Age': np.random.normal(45, 15, n_samples),
            'Education': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], n_samples),
            'ProgressStatus': np.random.choice(['Severely Struggling', 'Struggling', 'At Risk', 'On Track'], n_samples),
            'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
            'ProgramParticipation': np.random.choice(['Yes', 'No'], n_samples),
            'WaterAccess': np.random.choice(['Yes', 'No'], n_samples),
            'ElectricityAccess': np.random.choice(['Yes', 'No'], n_samples),
            'HealthcareAccess': np.random.choice(['Yes', 'No'], n_samples)
        }
        
        # Create correlated vulnerability levels
        vulnerability_scores = (
            (data['Income'] < 10000).astype(int) * 2 +
            (data['ProgressStatus'] == 'Severely Struggling').astype(int) * 2 +
            (data['Education'] == 'None').astype(int) +
            (data['WaterAccess'] == 'No').astype(int) +
            (data['ElectricityAccess'] == 'No').astype(int) +
            (data['HealthcareAccess'] == 'No').astype(int)
        )
        
        vulnerability_levels = []
        for score in vulnerability_scores:
            if score >= 5:
                vulnerability_levels.append('High')
            elif score >= 3:
                vulnerability_levels.append('Moderate')
            else:
                vulnerability_levels.append('Low')
        
        X = pd.DataFrame(data)
        y = pd.Series(vulnerability_levels)
        
        self.feature_names = list(data.keys())
        
        return X, y
    
    def setup_preprocessing(self, X):
        """
        Set up preprocessing pipeline for features.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            
        Returns:
            ColumnTransformer: Configured preprocessing pipeline
        """
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def build_model(self, input_shape):
        """
        Build neural network model architecture.
        
        Args:
            input_shape (int): Number of input features
            
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: High, Moderate, Low
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        """
        Train the neural network model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target labels
            test_size (float): Proportion of data for testing
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            dict: Training history and evaluation metrics
        """
        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded)
          # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Preprocess features
        self.setup_preprocessing(X_train)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Build and train model
        self.build_model(X_train_processed.shape[1])
        
        # Train model
        history = self.model.fit(
            X_train_processed, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_processed, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test_processed, y_test, verbose=0)
        
        # Generate predictions for detailed evaluation
        y_pred = self.model.predict(X_test_processed)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Store metadata
        self.model_metadata = {
            'training_date': datetime.now().isoformat(),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'epochs': epochs,
            'batch_size': batch_size,
            'n_features': X_train_processed.shape[1],
            'feature_names': self.feature_names,
            'class_labels': self.label_encoder.classes_.tolist()
        }
        
        print(f"\nModel Training Complete!")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes, 
                                  target_names=self.label_encoder.classes_))
        
        return {
            'history': history.history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': classification_report(y_test_classes, y_pred_classes, 
                                                         target_names=self.label_encoder.classes_,
                                                         output_dict=True)
        }
    
    def convert_to_tflite(self, output_path='models/vulnerability_model.tflite'):
        """
        Convert trained model to TensorFlow Lite format.
        
        Args:
            output_path (str): Path to save the .tflite model
            
        Returns:
            str: Path to the saved .tflite model
        """
        if self.model is None:
            raise ValueError("Model must be trained before conversion")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable dynamic range quantization for smaller model size
        converter.representative_dataset = None
        tflite_model = converter.convert()
        
        # Save .tflite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return output_path
    
    def save_artifacts(self, model_dir='models'):
        """
        Save all model artifacts including preprocessor and metadata.
        
        Args:
            model_dir (str): Directory to save model artifacts
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        
        # Save metadata
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        # Save Keras model
        self.model.save(os.path.join(model_dir, 'keras_model.h5'))
        
        print(f"Model artifacts saved to: {model_dir}")
    
    def load_artifacts(self, model_dir='models'):
        """
        Load saved model artifacts.
        
        Args:
            model_dir (str): Directory containing model artifacts
        """
        # Load preprocessor
        self.preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
        
        # Load label encoder
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        
        # Load metadata
        with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
            self.model_metadata = json.load(f)
        
        # Load Keras model
        self.model = tf.keras.models.load_model(os.path.join(model_dir, 'keras_model.h5'))
        
        print(f"Model artifacts loaded from: {model_dir}")
    
    def predict_single(self, household_data):
        """
        Make prediction for a single household.
        
        Args:
            household_data (dict): Dictionary containing household features
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([household_data])
        
        # Preprocess
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        predictions = self.model.predict(X_processed)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'predicted_class': self.label_encoder.classes_[predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {
                cls: float(prob) for cls, prob in 
                zip(self.label_encoder.classes_, predictions[0])
            }
        }
    
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
    
    def explore_dataset_structure(self, dataset, dataset_name="Dataset"):
        """
        Explore and display the structure of a dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to explore
            dataset_name (str): Name of the dataset for display
        """
        print(f"\n=== {dataset_name} Structure ===")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")
        
        print(f"\nData types:")
        for col, dtype in dataset.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print(f"\nMissing values:")
        missing = dataset.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(dataset)*100:.1f}%)")
        
        print(f"\nFirst 5 rows:")
        print(dataset.head())
        
        if len(dataset.select_dtypes(include=[np.number]).columns) > 0:
            print(f"\nNumerical summary:")
            print(dataset.describe())
    
    def load_datasets_from_folder(self, folder_path='Datasets'):
        """
        Load all datasets from the specified folder (default: Datasets).
        
        Args:
            folder_path (str): Path to the folder containing datasets
            
        Returns:
            dict: Dictionary containing loaded datasets
        """
        datasets = {}
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            return datasets
        
        print(f"Loading datasets from {folder_path} folder:")
        
        try:
            files = os.listdir(folder_path)
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                print(f"\nProcessing: {file}")
                
                if file.endswith('.csv'):
                    datasets[file] = pd.read_csv(file_path)
                    self.explore_dataset_structure(datasets[file], file)
                    
                elif file.endswith(('.xlsx', '.xls')):
                    datasets[file] = pd.read_excel(file_path)
                    self.explore_dataset_structure(datasets[file], file)
            
            return datasets
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return datasets
    
def main():
    """Main function to demonstrate the ML pipeline."""
    print("WorkMate App - ML Pipeline for Household Vulnerability Prediction")
    print("=" * 70)
    
    # Initialize predictor
    predictor = VulnerabilityPredictor()
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X, y = predictor.load_and_preprocess_data()
    
    # Train model
    print("\n2. Training neural network model...")
    training_results = predictor.train_model(X, y, epochs=30)
    
    # Save artifacts
    print("\n3. Saving model artifacts...")
    predictor.save_artifacts()
    
    # Convert to TensorFlow Lite
    print("\n4. Converting to TensorFlow Lite...")
    tflite_path = predictor.convert_to_tflite()
    
    # Test single prediction
    print("\n5. Testing single prediction...")
    sample_household = {
        'HouseholdSize': 5,
        'Income': 8000,
        'Age': 35,
        'Education': 'Primary',
        'ProgressStatus': 'Struggling',
        'Region': 'South',
        'ProgramParticipation': 'Yes',
        'WaterAccess': 'No',
        'ElectricityAccess': 'Yes',
        'HealthcareAccess': 'No'
    }
    
    result = predictor.predict_single(sample_household)
    print(f"Sample Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
    
    # Read datasets from static folder
    print("\n6. Reading datasets from static folder...")
    static_datasets = predictor.read_datasets_from_static('static')
    
    # Load additional datasets from folder
    print("\n7. Loading additional datasets from folder...")
    additional_datasets = predictor.load_datasets_from_folder('Datasets')
    
    print("\n" + "=" * 70)
    print("ML Pipeline completed successfully!")
    print(f"TensorFlow Lite model ready for mobile deployment: {tflite_path}")


if __name__ == "__main__":
    main()
