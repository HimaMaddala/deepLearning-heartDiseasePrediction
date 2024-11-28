from tensorflow.keras.models import load_model
import numpy as np
import plotly.express as px
import plotly.io as pio

# Load the saved model
model = load_model('model_saved.h5')

def predict_heart_disease(features):
    # Validate and reshape input features
    int_features = np.array(features[:9], dtype=np.int64) 
    oldpeak_feature = np.array([features[9]], dtype=np.float64)
    int_features_last = np.array(features[10:], dtype=np.int64)
    validated_features = np.concatenate((int_features, oldpeak_feature, int_features_last)).reshape(1, -1)
    
    # Get the model prediction
    prediction = model.predict(validated_features, verbose=0).flatten()
    return int(prediction[0] > 0.5)

def create_treemap(risk_factors):
    # Filter out non-risky factors
    filtered_factors = {k: v for k, v in risk_factors.items() if v > 0}
    
    # Prepare data for the TreeMap
    labels = list(filtered_factors.keys())
    values = list(filtered_factors.values())
    data = {"Feature": labels, "Impact": values}

    # Create the TreeMap
    fig = px.treemap(
        data,
        path=["Feature"],
        values="Impact",
        title="Major Contributing Factors to Heart Disease",
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig

def generate_visualization(features):
    # Define thresholds for risk factors
    risk_factors = {
        "High Cholesterol": features[4] if features[4] > 200 else 0,
        "High Blood Pressure": features[3] if features[3] > 130 else 0,
        "Low Max Heart Rate": features[7] if features[7] < 100 else 0,
        "High Fasting Blood Sugar": features[5] if features[5] == 1 else 0,
    }
    
    # Create TreeMap
    fig = create_treemap(risk_factors)
    
    # Save as HTML
    treemap_html = pio.to_html(fig, full_html=False)
    return treemap_html
