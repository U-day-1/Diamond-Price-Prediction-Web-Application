from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diamond_price_predictor_model.pkl')

# Mapping for color and clarity
color_mapping = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
clarity_mapping = {"IF": 1, "VVS1": 2, "VVS2": 3, "VS1": 4, "VS2": 5, "SI1": 6, "SI2": 7, "I1": 8}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        carat = float(request.form['carat'])
        color = request.form['color']
        clarity = request.form['clarity']

        color_num = color_mapping.get(color, 0)
        clarity_num = clarity_mapping.get(clarity, 0)

        prediction = model.predict([[carat, color_num, clarity_num]])

        # Return the prediction result as JSON
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
