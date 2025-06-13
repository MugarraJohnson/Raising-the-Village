"""
WorkMate App - Backend API Server

This Flask application provides the backend API for the WorkMate mobile app,
handling data synchronization, model updates, and household data management.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'workmate-secret-key-change-in-production')
CORS(app)

# Database configuration
DATABASE_PATH = 'workmate_backend.db'
MODEL_STORAGE_PATH = 'models'

class DatabaseManager:
    """Database operations manager for the WorkMate backend."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Households table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS households (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    household_size REAL,
                    income REAL,
                    age REAL,
                    education TEXT,
                    progress_status TEXT,
                    region TEXT,
                    program_participation TEXT,
                    water_access TEXT,
                    electricity_access TEXT,
                    healthcare_access TEXT,
                    vulnerability_level TEXT,
                    confidence REAL,
                    recommendations TEXT,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Field officers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS field_officers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    device_id TEXT,
                    region TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Model versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    description TEXT,
                    is_active BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Sync logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    sync_type TEXT NOT NULL,
                    records_count INTEGER,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def save_household(self, household_data: dict) -> bool:
        """Save household data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO households (
                        id, device_id, household_size, income, age, education,
                        progress_status, region, program_participation, water_access,
                        electricity_access, healthcare_access, vulnerability_level,
                        confidence, recommendations, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    household_data.get('id'),
                    household_data.get('device_id'),
                    household_data.get('household_size'),
                    household_data.get('income'),
                    household_data.get('age'),
                    household_data.get('education'),
                    household_data.get('progress_status'),
                    household_data.get('region'),
                    household_data.get('program_participation'),
                    household_data.get('water_access'),
                    household_data.get('electricity_access'),
                    household_data.get('healthcare_access'),
                    household_data.get('vulnerability_level'),
                    household_data.get('confidence'),
                    household_data.get('recommendations'),
                    household_data.get('timestamp')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving household data: {e}")
            return False
    
    def get_households_by_region(self, region: str, limit: int = 100) -> List[dict]:
        """Get households filtered by region."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM households 
                    WHERE region = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (region, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error fetching households: {e}")
            return []
    
    def get_analytics_data(self, days: int = 30) -> dict:
        """Get analytics data for the dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Date range
                start_date = datetime.now() - timedelta(days=days)
                start_timestamp = int(start_date.timestamp() * 1000)
                
                # Total households
                cursor.execute('SELECT COUNT(*) FROM households WHERE timestamp >= ?', (start_timestamp,))
                total_households = cursor.fetchone()[0]
                
                # Vulnerability distribution
                cursor.execute('''
                    SELECT vulnerability_level, COUNT(*) 
                    FROM households 
                    WHERE timestamp >= ? 
                    GROUP BY vulnerability_level
                ''', (start_timestamp,))
                vulnerability_dist = dict(cursor.fetchall())
                
                # Regional distribution
                cursor.execute('''
                    SELECT region, COUNT(*) 
                    FROM households 
                    WHERE timestamp >= ? 
                    GROUP BY region
                ''', (start_timestamp,))
                regional_dist = dict(cursor.fetchall())
                
                return {
                    'total_households': total_households,
                    'vulnerability_distribution': vulnerability_dist,
                    'regional_distribution': regional_dist,
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {}
    
    def log_sync_operation(self, device_id: str, sync_type: str, records_count: int, 
                          status: str, error_message: str = None):
        """Log synchronization operations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO sync_logs (device_id, sync_type, records_count, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (device_id, sync_type, records_count, status, error_message))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging sync operation: {e}")

# Initialize database manager
db_manager = DatabaseManager(DATABASE_PATH)

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_device_id = data['device_id']
            
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_device_id, *args, **kwargs)
    
    return decorated

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/auth/register', methods=['POST'])
def register_device():
    """Register a new device/field officer."""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({'message': 'Device ID is required'}), 400
        
        # Generate a simple token for device authentication
        token = jwt.encode({
            'device_id': device_id,
            'exp': datetime.utcnow() + timedelta(days=365)  # 1 year expiry
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        logger.info(f"Device registered: {device_id}")
        
        return jsonify({
            'message': 'Device registered successfully',
            'token': token,
            'device_id': device_id
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'message': 'Registration failed'}), 500

@app.route('/api/households', methods=['POST'])
@token_required
def sync_household(current_device_id):
    """Sync single household data."""
    try:
        data = request.get_json()
        
        # Extract household data
        household_data = data.get('householdData', {})
        prediction = data.get('prediction', {})
        
        # Prepare data for database
        db_data = {
            'id': household_data.get('householdId', f"HH_{int(datetime.now().timestamp())}"),
            'device_id': current_device_id,
            'household_size': household_data.get('householdSize'),
            'income': household_data.get('income'),
            'age': household_data.get('age'),
            'education': household_data.get('education'),
            'progress_status': household_data.get('progressStatus'),
            'region': household_data.get('region'),
            'program_participation': household_data.get('programParticipation'),
            'water_access': household_data.get('waterAccess'),
            'electricity_access': household_data.get('electricityAccess'),
            'healthcare_access': household_data.get('healthcareAccess'),
            'vulnerability_level': prediction.get('level'),
            'confidence': prediction.get('confidence'),
            'recommendations': '|'.join(prediction.get('recommendations', [])),
            'timestamp': data.get('timestamp', int(datetime.now().timestamp() * 1000))
        }
        
        success = db_manager.save_household(db_data)
        
        if success:
            db_manager.log_sync_operation(current_device_id, 'single', 1, 'success')
            return jsonify({
                'success': True,
                'message': 'Household data synced successfully',
                'householdId': db_data['id']
            })
        else:
            db_manager.log_sync_operation(current_device_id, 'single', 1, 'failed', 'Database save failed')
            return jsonify({
                'success': False,
                'message': 'Failed to save household data'
            }), 500
            
    except Exception as e:
        logger.error(f"Error syncing household: {e}")
        db_manager.log_sync_operation(current_device_id, 'single', 1, 'error', str(e))
        return jsonify({
            'success': False,
            'message': f'Sync failed: {str(e)}'
        }), 500

@app.route('/api/households/batch', methods=['POST'])
@token_required
def sync_households_batch(current_device_id):
    """Sync multiple households in batch."""
    try:
        data = request.get_json()
        households = data if isinstance(data, list) else []
        
        synced_count = 0
        failed_count = 0
        
        for household_request in households:
            try:
                # Process each household similar to single sync
                household_data = household_request.get('householdData', {})
                prediction = household_request.get('prediction', {})
                
                db_data = {
                    'id': household_data.get('householdId', f"HH_{int(datetime.now().timestamp())}_{synced_count}"),
                    'device_id': current_device_id,
                    'household_size': household_data.get('householdSize'),
                    'income': household_data.get('income'),
                    'age': household_data.get('age'),
                    'education': household_data.get('education'),
                    'progress_status': household_data.get('progressStatus'),
                    'region': household_data.get('region'),
                    'program_participation': household_data.get('programParticipation'),
                    'water_access': household_data.get('waterAccess'),
                    'electricity_access': household_data.get('electricityAccess'),
                    'healthcare_access': household_data.get('healthcareAccess'),
                    'vulnerability_level': prediction.get('level'),
                    'confidence': prediction.get('confidence'),
                    'recommendations': '|'.join(prediction.get('recommendations', [])),
                    'timestamp': household_request.get('timestamp', int(datetime.now().timestamp() * 1000))
                }
                
                if db_manager.save_household(db_data):
                    synced_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing household in batch: {e}")
                failed_count += 1
        
        db_manager.log_sync_operation(current_device_id, 'batch', synced_count, 'success')
        
        return jsonify({
            'syncedCount': synced_count,
            'failedCount': failed_count,
            'message': f'Batch sync completed: {synced_count} success, {failed_count} failed'
        })
        
    except Exception as e:
        logger.error(f"Error in batch sync: {e}")
        db_manager.log_sync_operation(current_device_id, 'batch', 0, 'error', str(e))
        return jsonify({
            'syncedCount': 0,
            'failedCount': len(data) if isinstance(data, list) else 1,
            'message': f'Batch sync failed: {str(e)}'
        }), 500

@app.route('/api/model/version', methods=['GET'])
@token_required
def check_model_update(current_device_id):
    """Check for model updates."""
    try:
        current_version = request.args.get('currentVersion', '1.0.0')
        
        # In a real implementation, you would check against your model versioning system
        # For demo purposes, we'll simulate a model update scenario
        
        latest_version = "1.1.0"  # This would come from your model registry
        has_update = current_version != latest_version
        
        response = {
            'hasUpdate': has_update,
            'modelVersion': latest_version,
            'downloadUrl': f'/api/model/download?version={latest_version}' if has_update else None,
            'modelSize': 2048576 if has_update else 0  # 2MB example
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error checking model update: {e}")
        return jsonify({
            'hasUpdate': False,
            'modelVersion': current_version,
            'downloadUrl': None,
            'modelSize': 0
        })

@app.route('/api/model/download', methods=['GET'])
@token_required
def download_model(current_device_id):
    """Download model update."""
    try:
        version = request.args.get('version', '1.0.0')
        
        # In a real implementation, you would serve the actual model file
        model_path = os.path.join(MODEL_STORAGE_PATH, f'vulnerability_model_{version}.tflite')
        
        if os.path.exists(model_path):
            return send_file(model_path, as_attachment=True)
        else:
            # For demo, return the base model
            base_model_path = os.path.join(MODEL_STORAGE_PATH, 'vulnerability_model.tflite')
            if os.path.exists(base_model_path):
                return send_file(base_model_path, as_attachment=True)
            else:
                return jsonify({'message': 'Model file not found'}), 404
                
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({'message': f'Download failed: {str(e)}'}), 500

@app.route('/api/analytics', methods=['GET'])
@token_required
def get_analytics(current_device_id):
    """Get analytics data for dashboard."""
    try:
        days = int(request.args.get('days', 30))
        analytics_data = db_manager.get_analytics_data(days)
        
        return jsonify(analytics_data)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'message': f'Analytics failed: {str(e)}'}), 500

@app.route('/api/households', methods=['GET'])
@token_required
def get_households(current_device_id):
    """Get households data with filtering."""
    try:
        region = request.args.get('region', '')
        limit = int(request.args.get('limit', 100))
        
        if region:
            households = db_manager.get_households_by_region(region, limit)
        else:
            # Get all households for this device (simplified)
            households = db_manager.get_households_by_region('', limit)
        
        return jsonify({
            'households': households,
            'count': len(households)
        })
        
    except Exception as e:
        logger.error(f"Error getting households: {e}")
        return jsonify({'message': f'Failed to get households: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

# CLI Commands for model management
@app.cli.command()
def init_db():
    """Initialize the database."""
    db_manager.init_database()
    print("Database initialized successfully!")

@app.cli.command()
def create_sample_data():
    """Create sample data for testing."""
    import random
    
    regions = ['North', 'South', 'East', 'West', 'Central']
    education_levels = ['None', 'Primary', 'Secondary', 'Higher']
    vulnerability_levels = ['HIGH', 'MODERATE', 'LOW']
    
    for i in range(50):
        household_data = {
            'id': f'SAMPLE_HH_{i}',
            'device_id': 'SAMPLE_DEVICE',
            'household_size': random.randint(1, 8),
            'income': random.randint(5000, 25000),
            'age': random.randint(25, 65),
            'education': random.choice(education_levels),
            'progress_status': random.choice(['Severely Struggling', 'Struggling', 'At Risk', 'On Track']),
            'region': random.choice(regions),
            'program_participation': random.choice(['Yes', 'No']),
            'water_access': random.choice(['Yes', 'No']),
            'electricity_access': random.choice(['Yes', 'No']),
            'healthcare_access': random.choice(['Yes', 'No']),
            'vulnerability_level': random.choice(vulnerability_levels),
            'confidence': random.uniform(0.6, 0.95),
            'recommendations': 'Sample recommendation 1|Sample recommendation 2',
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        db_manager.save_household(household_data)
    
    print("Sample data created successfully!")

if __name__ == '__main__':
    # Ensure model storage directory exists
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
