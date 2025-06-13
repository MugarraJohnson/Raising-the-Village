#!/usr/bin/env python3
"""
WorkMate App Deployment Script

This script handles the complete deployment workflow for the WorkMate App,
including model training, backend setup, and Android app preparation.
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path

class WorkMateDeployer:
    """WorkMate App deployment manager."""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.ml_pipeline_dir = self.project_root / "ml_pipeline"
        self.backend_dir = self.project_root / "backend_api"
        self.android_dir = self.project_root / "android_app"
        self.models_dir = self.project_root / "models"
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
    
    def run_command(self, command, cwd=None, check=True):
        """Execute a shell command with error handling."""
        print(f"Executing: {' '.join(command)}")
        try:
            result = subprocess.run(
                command, 
                cwd=cwd or self.project_root,
                check=check,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            if check:
                sys.exit(1)
            return e
    
    def setup_python_environment(self):
        """Set up Python virtual environment and install dependencies."""
        print("Setting up Python environment...")
        
        # Create virtual environment if it doesn't exist
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            self.run_command([sys.executable, "-m", "venv", "venv"])
        
        # Determine pip executable path
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip.exe"
            python_path = venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
        
        # Install main requirements
        if (self.project_root / "requirements.txt").exists():
            self.run_command([str(pip_path), "install", "-r", "requirements.txt"])
        
        # Install backend requirements
        if (self.backend_dir / "requirements.txt").exists():
            self.run_command([str(pip_path), "install", "-r", "backend_api/requirements.txt"])
        
        return str(python_path)
    
    def train_ml_model(self, python_path):
        """Train the machine learning model."""
        print("Training ML model...")
        
        training_script = self.ml_pipeline_dir / "model_training.py"
        if training_script.exists():
            self.run_command([python_path, str(training_script)])
        else:
            print("Warning: ML training script not found, skipping model training")
    
    def setup_backend(self, python_path):
        """Set up and configure the backend API."""
        print("Setting up backend API...")
        
        backend_script = self.backend_dir / "app.py"
        if backend_script.exists():
            # Initialize database
            env = os.environ.copy()
            env['FLASK_APP'] = str(backend_script)
            
            self.run_command([python_path, "-m", "flask", "init-db"], 
                           cwd=self.backend_dir, check=False)
            
            # Create sample data
            self.run_command([python_path, "-m", "flask", "create-sample-data"], 
                           cwd=self.backend_dir, check=False)
            
            print("Backend setup completed")
        else:
            print("Warning: Backend script not found, skipping backend setup")
    
    def prepare_android_assets(self):
        """Prepare Android app assets including TensorFlow Lite model."""
        print("Preparing Android assets...")
        
        # Create Android assets directory
        assets_dir = self.android_dir / "src" / "main" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy TensorFlow Lite model to assets
        tflite_model = self.models_dir / "vulnerability_model.tflite"
        if tflite_model.exists():
            shutil.copy2(tflite_model, assets_dir)
            print(f"Copied TensorFlow Lite model to Android assets")
        else:
            print("Warning: TensorFlow Lite model not found, Android app may not work properly")
        
        # Copy model metadata
        metadata_file = self.models_dir / "model_metadata.json"
        if metadata_file.exists():
            shutil.copy2(metadata_file, assets_dir)
            print("Copied model metadata to Android assets")
    
    def validate_deployment(self):
        """Validate that all components are properly set up."""
        print("Validating deployment...")
        
        issues = []
        
        # Check for required files
        required_files = [
            self.project_root / "requirements.txt",
            self.ml_pipeline_dir / "model_training.py",
            self.backend_dir / "app.py",
            self.android_dir / "build.gradle"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                issues.append(f"Missing required file: {file_path}")
        
        # Check for model files
        if not (self.models_dir / "vulnerability_model.tflite").exists():
            issues.append("TensorFlow Lite model not found - run model training first")
        
        # Check Android assets
        android_assets = self.android_dir / "src" / "main" / "assets"
        if not android_assets.exists():
            issues.append("Android assets directory not found")
        
        if issues:
            print("Deployment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("Deployment validation passed!")
            return True
    
    def generate_deployment_summary(self):
        """Generate a deployment summary report."""
        print("\n" + "="*60)
        print("WORKMATE APP DEPLOYMENT SUMMARY")
        print("="*60)
        
        # Check component status
        components = {
            "ML Pipeline": self.ml_pipeline_dir.exists(),
            "Backend API": self.backend_dir.exists(),
            "Android App": self.android_dir.exists(),
            "TF Lite Model": (self.models_dir / "vulnerability_model.tflite").exists(),
            "Documentation": (self.project_root / "README.md").exists()
        }
        
        for component, status in components.items():
            status_str = "✓ Ready" if status else "✗ Missing"
            print(f"{component:20s}: {status_str}")
        
        print("\nNext Steps:")
        
        if components["ML Pipeline"]:
            print("1. Train ML model: python ml_pipeline/model_training.py")
        
        if components["Backend API"]:
            print("2. Start backend: python backend_api/app.py")
        
        if components["Android App"]:
            print("3. Build Android app in Android Studio")
        
        print("4. Deploy backend to production server")
        print("5. Distribute Android APK to field officers")
        
        print("\nDeployment completed successfully! 🎉")
        print("="*60 + "\n")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="WorkMate App Deployment Script")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML model training")
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend setup")
    parser.add_argument("--skip-android", action="store_true", help="Skip Android preparation")
    parser.add_argument("--validate-only", action="store_true", help="Only validate deployment")
    
    args = parser.parse_args()
    
    # Get project root directory
    project_root = Path(__file__).parent
    deployer = WorkMateDeployer(project_root)
    
    try:
        if args.validate_only:
            deployer.validate_deployment()
            return
        
        print("Starting WorkMate App deployment...")
        
        # Set up Python environment
        python_path = deployer.setup_python_environment()
        
        # Train ML model
        if not args.skip_ml:
            deployer.train_ml_model(python_path)
        
        # Set up backend
        if not args.skip_backend:
            deployer.setup_backend(python_path)
        
        # Prepare Android assets
        if not args.skip_android:
            deployer.prepare_android_assets()
        
        # Validate deployment
        if deployer.validate_deployment():
            deployer.generate_deployment_summary()
        else:
            print("Deployment validation failed. Please check the issues above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
