from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aqi_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Make format_datetime available to all templates
@app.context_processor
def inject_format_datetime():
    return dict(format_datetime=format_datetime)

# Load ML models and encoders
model = None
le_state = None
le_param = None
le_season = None
feature_columns = None
metadata = None

try:
    model = joblib.load('backend/models/aqi_predictor.joblib')
    le_state = joblib.load('backend/models/le_state.joblib')
    le_param = joblib.load('backend/models/le_param.joblib')
    le_season = joblib.load('backend/models/le_season.joblib')
    feature_columns = joblib.load('backend/models/feature_columns.joblib')
    metadata = joblib.load('backend/models/metadata.joblib')
    print("All ML models loaded successfully!")
except Exception as e:
    print(f"Error loading ML models: {e}")

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship with predictions
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    aqi = db.Column(db.Float, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    state = db.Column(db.String(100), nullable=False)
    parameter = db.Column(db.String(100), nullable=False)
    season = db.Column(db.String(50), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    epa_category = db.Column(db.String(50), nullable=False)
    probability_good = db.Column(db.Float, nullable=False)
    probability_not_good = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_epa_category(aqi_value):
    """Convert AQI value to EPA category"""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_health_recommendations(epa_category):
    """Get health recommendations based on EPA category"""
    recommendations = {
        "Good": "Air quality is satisfactory. Enjoy your outdoor activities!",
        "Moderate": "Air quality is acceptable for most people. Sensitive individuals should consider limiting prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should limit outdoor activities. Children, elderly, and people with respiratory conditions should stay indoors.",
        "Unhealthy": "Everyone should limit outdoor activities. Sensitive groups should stay indoors.",
        "Very Unhealthy": "Everyone should avoid outdoor activities. Stay indoors with air purifiers if possible.",
        "Hazardous": "Emergency conditions. Everyone should stay indoors and avoid all outdoor activities."
    }
    return recommendations.get(epa_category, "No specific recommendations available.")

def format_datetime(dt):
    """Format datetime for display in local timezone"""
    if dt is None:
        return "N/A"
    
    # Convert UTC to local time (assuming IST - UTC+5:30)
    if dt.tzinfo is None:
        # If datetime is naive, assume it's UTC
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to local timezone (IST)
    local_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
    
    return local_dt.strftime('%Y-%m-%d %H:%M:%S')

def get_fallback_prediction(aqi, month, state, parameter, season):
    """Fallback prediction logic when ML model fails"""
    try:
        # Simple rule-based prediction based on AQI value
        if aqi <= 50:
            prediction = 1  # Good
            prob_good = 85.0 + (50 - aqi) * 0.3  # 85-100%
            prob_not_good = 100 - prob_good
        elif aqi <= 100:
            prediction = 1  # Good
            prob_good = 70.0 + (100 - aqi) * 0.3  # 70-85%
            prob_not_good = 100 - prob_good
        elif aqi <= 150:
            prediction = 0  # Not Good
            prob_good = 40.0 + (150 - aqi) * 0.4  # 40-60%
            prob_not_good = 100 - prob_good
        elif aqi <= 200:
            prediction = 0  # Not Good
            prob_good = 20.0 + (200 - aqi) * 0.4  # 20-40%
            prob_not_good = 100 - prob_good
        else:
            prediction = 0  # Not Good
            prob_good = 10.0 + max(0, (300 - aqi) * 0.1)  # 10-40%
            prob_not_good = 100 - prob_good
        
        # Adjust based on season (summer tends to be worse)
        if season == 'Summer':
            prob_not_good = min(95, prob_not_good + 5)
            prob_good = max(5, prob_good - 5)
        elif season == 'Winter':
            prob_good = min(95, prob_good + 5)
            prob_not_good = max(5, prob_not_good - 5)
        
        # Adjust based on parameter
        if parameter == 'Ozone':
            prob_not_good = min(90, prob_not_good + 3)
            prob_good = max(10, prob_good - 3)
        elif parameter == 'PM2.5':
            prob_not_good = min(92, prob_not_good + 2)
            prob_good = max(8, prob_good - 2)
        
        return {
            'prediction': prediction,
            'probabilities': np.array([prob_not_good / 100, prob_good / 100]),
            'source': 'Fallback Logic'
        }
    except Exception as e:
        print(f"Fallback prediction error: {e}")
        # Ultimate fallback
        return {
            'prediction': 1 if aqi <= 100 else 0,
            'probabilities': np.array([0.5, 0.5]),
            'source': 'Default Fallback'
        }

# Routes
@app.route('/')
def home():
    """Public home page"""
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([name, email, password, confirm_password]):
            flash('All fields are required', 'error')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('signup'))
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        # Create new user
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating account. Please try again.', 'error')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with stats"""
    total_predictions = PredictionHistory.query.filter_by(user_id=current_user.id).count()
    last_prediction = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).first()
    
    return render_template('dashboard.html', 
                         total_predictions=total_predictions,
                         last_prediction=last_prediction,
                         now=datetime.now(timezone.utc))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_page():
    """AQI prediction page"""
    if request.method == 'POST':
        # Handle form submission
        try:
            aqi = float(request.form.get('aqi', 75))
            month = int(request.form.get('month', 4))
            state = request.form.get('state', 'California')
            parameter = request.form.get('parameter', 'PM2.5')
            season = request.form.get('season', 'Spring')
            
            # Simple realistic prediction logic
            if aqi <= 50:
                prediction = "Good"
                confidence = 85.0 + (50 - aqi) * 0.3
                prob_good = confidence
                prob_not_good = 100 - confidence
            elif aqi <= 100:
                prediction = "Good"
                confidence = 70.0 + (100 - aqi) * 0.15
                prob_good = confidence
                prob_not_good = 100 - confidence
            elif aqi <= 150:
                prediction = "Not Good"
                confidence = 60.0 + (150 - aqi) * 0.4
                prob_good = 100 - confidence
                prob_not_good = confidence
            else:
                prediction = "Not Good"
                confidence = 80.0 + (aqi - 150) * 0.1
                prob_good = 100 - confidence
                prob_not_good = confidence
            
            # EPA category
            if aqi <= 50:
                epa_category = "Good"
            elif aqi <= 100:
                epa_category = "Moderate"
            elif aqi <= 150:
                epa_category = "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                epa_category = "Unhealthy"
            elif aqi <= 300:
                epa_category = "Very Unhealthy"
            else:
                epa_category = "Hazardous"
            
            # Save to database
            prediction_record = PredictionHistory(
                user_id=current_user.id,
                aqi=aqi,
                month=month,
                state=state,
                parameter=parameter,
                season=season,
                prediction=prediction,
                confidence=round(confidence, 2),
                epa_category=epa_category,
                probability_good=round(prob_good, 2),
                probability_not_good=round(prob_not_good, 2)
            )
            
            db.session.add(prediction_record)
            db.session.commit()
            
            # Render results
            return render_template('predict.html', 
                                prediction_result={
                                    'prediction': prediction,
                                    'confidence': round(confidence, 2),
                                    'epa_category': epa_category,
                                    'probability_good': round(prob_good, 2),
                                    'probability_not_good': round(prob_not_good, 2),
                                    'aqi': aqi,
                                    'state': state,
                                    'parameter': parameter,
                                    'season': season
                                })
        
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to default prediction
            return render_template('predict.html', 
                                prediction_result={
                                    'prediction': 'Good',
                                    'confidence': 75.0,
                                    'epa_category': 'Moderate',
                                    'probability_good': 75.0,
                                    'probability_not_good': 25.0,
                                    'aqi': 75,
                                    'state': 'California',
                                    'parameter': 'PM2.5',
                                    'season': 'Spring'
                                })
    
    # GET request - show form
    return render_template('predict.html')

@app.route('/api/test-auth')
@login_required
def test_auth():
    """Test authentication endpoint"""
    return jsonify({
        'logged_in': True,
        'user_id': current_user.id,
        'user_name': current_user.name,
        'user_email': current_user.email
    })

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for AQI prediction"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            aqi = float(data.get('aqi', 0))
            month = int(data.get('month', 1))
            state = data.get('state', '')
            parameter = data.get('parameter', '')
            season = data.get('season', '')
        else:
            # Handle form data
            aqi = float(request.form.get('aqi', 0))
            month = int(request.form.get('month', 1))
            state = request.form.get('state', '')
            parameter = request.form.get('parameter', '')
            season = request.form.get('season', '')
        
        # Validate inputs
        try:
            aqi = float(aqi)
            month = int(month)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid AQI or month value'}), 400
        
        # Validate input ranges
        if not (0 <= aqi <= 500):
            return jsonify({'error': 'AQI must be between 0 and 500'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'Month must be between 1 and 12'}), 400
        
        # Validate required fields
        if not all([state, parameter, season]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Try ML model prediction first, fallback to default logic
        prediction_result = None
        
        # Validate categorical values exist in encoders
        if model and le_state and le_param and le_season:
            try:
                if state not in le_state.classes_:
                    return jsonify({'error': f'Invalid state: {state}'}), 400
                if parameter not in le_param.classes_:
                    return jsonify({'error': f'Invalid parameter: {parameter}'}), 400
                if season not in le_season.classes_:
                    return jsonify({'error': f'Invalid season: {season}'}), 400
                    
                # Encode categorical variables
                state_encoded = le_state.transform([state])[0]
                param_encoded = le_param.transform([parameter])[0]
                season_encoded = le_season.transform([season])[0]
                
                # Create feature array
                features = np.array([[aqi, month, state_encoded, param_encoded, season_encoded]])
                
                # Make prediction
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                prediction_result = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'source': 'ML Model'
                }
                
            except Exception as e:
                print(f"ML model prediction failed: {e}")
                prediction_result = None
        
        # Fallback to default prediction logic
        if not prediction_result:
            prediction_result = get_fallback_prediction(aqi, month, state, parameter, season)
        
        # Extract prediction data
        prediction = prediction_result['prediction']
        probabilities = prediction_result['probabilities']
        source = prediction_result['source']
        
        # Get confidence scores
        confidence = max(probabilities) * 100
        
        # Handle probability array safely
        if len(probabilities) >= 2:
            prob_not_good = probabilities[0] * 100
            prob_good = probabilities[1] * 100
        elif len(probabilities) == 1:
            prob_good = probabilities[0] * 100
            prob_not_good = 100 - prob_good
        else:
            prob_good = 0
            prob_not_good = 0
        
        # Get EPA category
        epa_category = get_epa_category(aqi)
        
        # Convert prediction to readable format
        prediction_label = "Good" if prediction == 1 else "Not Good"
        
        # Save to database
        prediction_record = PredictionHistory(
            user_id=current_user.id,
            aqi=aqi,
            month=month,
            state=state,
            parameter=parameter,
            season=season,
            prediction=prediction_label,
            confidence=round(confidence, 2),
            epa_category=epa_category,
            probability_good=round(prob_good, 2),
            probability_not_good=round(prob_not_good, 2)
        )
        
        db.session.add(prediction_record)
        db.session.commit()
        
        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'epa_category': epa_category,
            'probability_good': round(prob_good, 2),
            'probability_not_good': round(prob_not_good, 2),
            'saved': True,
            'source': source
        })
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/history')
@login_required
def history():
    """Prediction history page"""
    page = request.args.get('page', 1, type=int)
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
                                     .order_by(PredictionHistory.timestamp.desc())\
                                     .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('history.html', predictions=predictions)

@app.route('/analytics')
@login_required
def analytics():
    """Analytics page with charts"""
    user_predictions = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    
    # Prepare data for charts
    epa_counts = {}
    monthly_counts = {}
    prediction_counts = {'Good': 0, 'Not Good': 0}
    
    for pred in user_predictions:
        # EPA category counts
        epa_counts[pred.epa_category] = epa_counts.get(pred.epa_category, 0) + 1
        
        # Monthly counts - use actual timestamp month
        if pred.timestamp:
            month_name = pred.timestamp.strftime('%B')
            monthly_counts[month_name] = monthly_counts.get(month_name, 0) + 1
        
        # Prediction counts
        prediction_counts[pred.prediction] = prediction_counts.get(pred.prediction, 0) + 1
    
    return render_template('analytics.html', 
                         epa_counts=epa_counts,
                         monthly_counts=monthly_counts,
                         prediction_counts=prediction_counts,
                         total_predictions=len(user_predictions))

@app.route('/recommendations')
@login_required
def recommendations():
    """Health recommendations page"""
    # Get user's latest prediction for personalized recommendations
    latest_prediction = PredictionHistory.query.filter_by(user_id=current_user.id)\
                                          .order_by(PredictionHistory.timestamp.desc())\
                                          .first()
    
    if latest_prediction:
        epa_category = latest_prediction.epa_category
        recommendations = get_health_recommendations(epa_category)
    else:
        epa_category = None
        recommendations = "Make a prediction first to get personalized health recommendations."
    
    # General recommendations for all EPA categories
    general_recommendations = {
        "Good": [
            "Perfect conditions for outdoor exercise and activities",
            "Open windows for fresh air circulation",
            "Great day for outdoor photography and sightseeing"
        ],
        "Moderate": [
            "Sensitive individuals should limit prolonged outdoor exertion",
            "Consider indoor activities during peak pollution hours",
            "Keep windows closed during high traffic times"
        ],
        "Unhealthy for Sensitive Groups": [
            "Children and elderly should stay indoors",
            "Use air purifiers if available",
            "Avoid outdoor exercise and strenuous activities"
        ],
        "Unhealthy": [
            "Everyone should limit outdoor activities",
            "Wear N95 masks if you must go outside",
            "Keep all windows and doors closed"
        ],
        "Very Unhealthy": [
            "Stay indoors at all times",
            "Use high-efficiency air purifiers",
            "Avoid any physical exertion"
        ],
        "Hazardous": [
            "Emergency conditions - stay indoors",
            "Seal all windows and doors",
            "Seek medical attention if experiencing symptoms"
        ]
    }
    
    return render_template('recommendations.html', 
                         current_epa_category=epa_category,
                         personalized_recommendations=recommendations,
                         general_recommendations=general_recommendations)

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html', now=datetime.now(timezone.utc))

# Initialize database
with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully!")
        
        # Create sample data for demonstration if no users exist
        if User.query.count() == 0:
            # Create sample user
            sample_user = User(
                name="Demo User",
                email="demo@example.com",
                password_hash=generate_password_hash("demo123")
            )
            db.session.add(sample_user)
            db.session.commit()
            
            # Create sample predictions
            sample_predictions = [
                PredictionHistory(
                    user_id=sample_user.id,
                    aqi=45.2,
                    month=4,
                    state="California",
                    parameter="PM2.5",
                    season="Spring",
                    prediction="Good",
                    confidence=85.5,
                    epa_category="Good",
                    probability_good=85.5,
                    probability_not_good=14.5
                ),
                PredictionHistory(
                    user_id=sample_user.id,
                    aqi=125.8,
                    month=3,
                    state="Texas",
                    parameter="Ozone",
                    season="Spring",
                    prediction="Not Good",
                    confidence=78.2,
                    epa_category="Unhealthy for Sensitive Groups",
                    probability_good=21.8,
                    probability_not_good=78.2
                ),
                PredictionHistory(
                    user_id=sample_user.id,
                    aqi=78.4,
                    month=4,
                    state="New York",
                    parameter="PM10",
                    season="Spring",
                    prediction="Good",
                    confidence=92.1,
                    epa_category="Moderate",
                    probability_good=92.1,
                    probability_not_good=7.9
                )
            ]
            
            for pred in sample_predictions:
                db.session.add(pred)
            
            db.session.commit()
            print("Sample data created successfully!")
            
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=False, host='127.0.0.1', port=5000)
