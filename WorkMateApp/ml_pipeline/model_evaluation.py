"""
Model Evaluation and Validation Module

This module provides comprehensive evaluation tools for the household vulnerability
prediction model, including performance metrics, validation, and analysis.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and validation class.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize evaluator with model directory.
        
        Args:
            model_dir (str): Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.metadata = None
        self.evaluation_results = {}
        
    def load_model_artifacts(self):
        """Load all model artifacts for evaluation."""
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(
                os.path.join(self.model_dir, 'keras_model.h5')
            )
            
            # Load preprocessor
            self.preprocessor = joblib.load(
                os.path.join(self.model_dir, 'preprocessor.pkl')
            )
            
            # Load label encoder
            self.label_encoder = joblib.load(
                os.path.join(self.model_dir, 'label_encoder.pkl')
            )
            
            # Load metadata
            with open(os.path.join(self.model_dir, 'model_metadata.json'), 'r') as f:
                self.metadata = json.load(f)
            
            print("Model artifacts loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            return False
    
    def evaluate_tflite_model(self, X_test, y_test, tflite_path='models/vulnerability_model.tflite'):
        """
        Evaluate TensorFlow Lite model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            tflite_path (str): Path to TensorFlow Lite model
            
        Returns:
            dict: TFLite model evaluation results
        """
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            predictions = []
            
            # Make predictions for each sample
            for i in range(len(X_test)):
                # Set input tensor
                interpreter.set_tensor(
                    input_details[0]['index'], 
                    X_test[i:i+1].astype(np.float32)
                )
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                output = interpreter.get_tensor(output_details[0]['index'])
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predicted_classes)
            precision = precision_score(y_test, predicted_classes, average='weighted')
            recall = recall_score(y_test, predicted_classes, average='weighted')
            f1 = f1_score(y_test, predicted_classes, average='weighted')
            
            tflite_results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'predictions': predictions.tolist(),
                'predicted_classes': predicted_classes.tolist()
            }
            
            print(f"TensorFlow Lite Model Evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            return tflite_results
            
        except Exception as e:
            print(f"Error evaluating TFLite model: {e}")
            return None
    
    def compare_keras_vs_tflite(self, X_test, y_test):
        """
        Compare Keras and TensorFlow Lite model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Comparison results
        """
        if self.model is None:
            print("Keras model not loaded!")
            return None
        
        # Keras model predictions
        keras_predictions = self.model.predict(X_test)
        keras_classes = np.argmax(keras_predictions, axis=1)
        
        # TFLite model predictions
        tflite_results = self.evaluate_tflite_model(X_test, y_test)
        
        if tflite_results is None:
            return None
        
        tflite_classes = np.array(tflite_results['predicted_classes'])
        
        # Calculate agreement between models
        agreement = np.mean(keras_classes == tflite_classes)
        
        # Keras metrics
        keras_accuracy = accuracy_score(y_test, keras_classes)
        keras_precision = precision_score(y_test, keras_classes, average='weighted')
        keras_recall = recall_score(y_test, keras_classes, average='weighted')
        keras_f1 = f1_score(y_test, keras_classes, average='weighted')
        
        comparison_results = {
            'keras_metrics': {
                'accuracy': float(keras_accuracy),
                'precision': float(keras_precision),
                'recall': float(keras_recall),
                'f1_score': float(keras_f1)
            },
            'tflite_metrics': {
                'accuracy': tflite_results['accuracy'],
                'precision': tflite_results['precision'],
                'recall': tflite_results['recall'],
                'f1_score': tflite_results['f1_score']
            },
            'model_agreement': float(agreement),
            'differences': {
                'accuracy_diff': abs(keras_accuracy - tflite_results['accuracy']),
                'precision_diff': abs(keras_precision - tflite_results['precision']),
                'recall_diff': abs(keras_recall - tflite_results['recall']),
                'f1_diff': abs(keras_f1 - tflite_results['f1_score'])
            }
        }
        
        print(f"\nModel Comparison Results:")
        print(f"Model Agreement: {agreement:.4f}")
        print(f"Accuracy Difference: {comparison_results['differences']['accuracy_diff']:.4f}")
        print(f"Precision Difference: {comparison_results['differences']['precision_diff']:.4f}")
        
        return comparison_results
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            title (str): Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.label_encoder.classes_
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {plot_path}")
        plt.show()
    
    def plot_model_performance(self, history):
        """
        Plot training history.
        
        Args:
            history (dict): Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
        plt.show()
    
    def generate_evaluation_report(self, X_test, y_test, save_path=None):
        """
        Generate comprehensive evaluation report.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            save_path (str): Optional path to save report
            
        Returns:
            dict: Complete evaluation report
        """
        if self.model is None:
            print("Model not loaded! Cannot generate report.")
            return None
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, predicted_classes)
        precision = precision_score(y_test, predicted_classes, average='weighted')
        recall = recall_score(y_test, predicted_classes, average='weighted')
        f1 = f1_score(y_test, predicted_classes, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(
            y_test, predicted_classes, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Feature importance (approximate using permutation importance)
        feature_importance = self._calculate_feature_importance(X_test, y_test)
        
        # Model comparison
        comparison = self.compare_keras_vs_tflite(X_test, y_test)
        
        # Compile report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'model_metadata': self.metadata,
            'test_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class_metrics': class_report,
            'feature_importance': feature_importance,
            'model_comparison': comparison,
            'recommendations': self._generate_recommendations(accuracy, precision, recall, f1)
        }
        
        # Save report if path provided
        if save_path is None:
            save_path = os.path.join(self.model_dir, 'evaluation_report.json')
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {save_path}")
        return report
    
    def _calculate_feature_importance(self, X_test, y_test, n_iterations=10):
        """
        Calculate approximate feature importance using permutation importance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            n_iterations (int): Number of permutation iterations
            
        Returns:
            dict: Feature importance scores
        """
        baseline_score = self.model.evaluate(X_test, y_test, verbose=0)[1]
        
        importances = []
        feature_names = self.metadata.get('feature_names', [f'feature_{i}' for i in range(X_test.shape[1])])
        
        for i in range(X_test.shape[1]):
            scores = []
            for _ in range(n_iterations):
                X_permuted = X_test.copy()
                # Permute feature i
                np.random.shuffle(X_permuted[:, i])
                score = self.model.evaluate(X_permuted, y_test, verbose=0)[1]
                scores.append(baseline_score - score)
            
            importances.append(np.mean(scores))
        
        # Create feature importance dictionary
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(feature_names, importances)
        }
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def _generate_recommendations(self, accuracy, precision, recall, f1):
        """
        Generate recommendations based on model performance.
        
        Args:
            accuracy (float): Model accuracy
            precision (float): Model precision
            recall (float): Model recall
            f1 (float): Model F1-score
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        if accuracy < 0.7:
            recommendations.append("Consider increasing model complexity or collecting more training data")
        
        if precision < 0.7:
            recommendations.append("High false positive rate - consider adjusting classification threshold")
        
        if recall < 0.7:
            recommendations.append("High false negative rate - may need more balanced training data")
        
        if f1 < 0.7:
            recommendations.append("Overall model performance needs improvement - consider feature engineering")
        
        if len(recommendations) == 0:
            recommendations.append("Model performance is satisfactory for deployment")
        
        return recommendations


def main():
    """Main function for model evaluation."""
    print("WorkMate App - Model Evaluation and Validation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model artifacts
    if not evaluator.load_model_artifacts():
        print("Failed to load model artifacts. Please train the model first.")
        return
    
    # For demonstration, create test data
    # In practice, this would be your actual test set
    from model_training import VulnerabilityPredictor
    
    predictor = VulnerabilityPredictor()
    X, y = predictor._create_synthetic_data(n_samples=200)
    
    # Preprocess test data
    predictor.setup_preprocessing(X)
    X_processed = predictor.preprocessor.fit_transform(X)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nEvaluating model on {len(X_processed)} test samples...")
    
    # Generate comprehensive evaluation report
    report = evaluator.generate_evaluation_report(X_processed, y_encoded)
    
    # Plot confusion matrix
    predictions = evaluator.model.predict(X_processed)
    predicted_classes = np.argmax(predictions, axis=1)
    evaluator.plot_confusion_matrix(y_encoded, predicted_classes)
    
    print("\nEvaluation completed!")
    print(f"Test Accuracy: {report['test_metrics']['accuracy']:.4f}")
    print(f"Test F1-Score: {report['test_metrics']['f1_score']:.4f}")


if __name__ == "__main__":
    main()
