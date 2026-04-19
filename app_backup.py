from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load all ML models and encoders
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
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Application cannot start without ML models. Exiting...")
    import sys
    sys.exit(1)

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

@app.route('/')
def home():
    """Render the main page"""
    # Get unique values for dropdowns
    states = list(le_state.classes_)
    parameters = list(le_param.classes_)
    seasons = list(le_season.classes_)
    
    # Get model accuracy from metadata if available
    model_accuracy = metadata.get('accuracy', 'N/A') if metadata else 'N/A'
    
    return render_template('index.html', 
                         states=states, 
                         parameters=parameters, 
                         seasons=seasons,
                         model_accuracy=model_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract and validate inputs
        try:
            aqi = float(data.get('aqi', 0))
            month = int(data.get('month', 1))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid AQI or month value'}), 400
            
        state = data.get('state', '')
        parameter = data.get('parameter', '')
        season = data.get('season', '')
        
        # Validate input ranges
        if not (0 <= aqi <= 500):
            return jsonify({'error': 'AQI must be between 0 and 500'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'Month must be between 1 and 12'}), 400
        
        # Validate required fields
        if not all([state, parameter, season]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Validate categorical values exist in encoders
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
        except ValueError as e:
            return jsonify({'error': f'Encoding error: {str(e)}'}), 400
        
        # Create feature array in the correct order
        features = np.array([[aqi, month, state_encoded, param_encoded, season_encoded]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
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
        
        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'epa_category': epa_category,
            'probability_good': round(prob_good, 2),
            'probability_not_good': round(prob_not_good, 2)
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/get-model-info')
def get_model_info():
    """Get model information"""
    try:
        info = {
            'accuracy': metadata.get('accuracy', 'N/A') if metadata else 'N/A',
            'features': list(feature_columns) if feature_columns else [],
            'model_type': 'Random Forest'
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=False, host='127.0.0.1', port=5000)
