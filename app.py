from flask import Flask, render_template, request, jsonify
from model import predict_heart_disease, generate_visualization

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        # Predict the result
        prediction = predict_heart_disease(features)
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        treemap_html = generate_visualization(features)
        
        # Return the result and visualization as JSON
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": f"Error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
