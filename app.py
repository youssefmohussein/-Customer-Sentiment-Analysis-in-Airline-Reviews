import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import scipy.sparse as sp

app = Flask(__name__)
CORS(app)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'airline_sentiment_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

model = None
label_encoder = None
tfidf_vectorizer = None

def load_models():
    global model, label_encoder, tfidf_vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models. Make sure pkl files are in the '{MODEL_DIR}' directory. Error: {e}")

load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not label_encoder or not tfidf_vectorizer:
        return jsonify({"error": "Models are not loaded on the server. Ensure pkl files are in 'models/' directory."}), 500

    try:
        data = request.json
        review_text = data.get('review', '')
        
        # 1. Transform review text using tfidf_vectorizer
        text_features = tfidf_vectorizer.transform([review_text])
        
        # --- FIX FOR 1000 FEATURES MISMATCH ---
        # The model expects 1000 features.
        # We pad or truncate the features to exactly 1000.
        missing_features = 1000 - text_features.shape[1]
        if missing_features > 0:
            padding = sp.csr_matrix(np.zeros((1, missing_features)))
            final_features = sp.hstack([text_features, padding])
        elif missing_features < 0:
            # If for some reason we have more features than expected, truncate them
            final_features = text_features[:, :1000]
        else:
            final_features = text_features
        # -------------------------------------
        
        # 4. Predict using the model
        prediction_num = model.predict(final_features)[0]
        
        # Try to get probability for confidence
        try:
            probabilities = model.predict_proba(final_features)[0]
            confidence = float(np.max(probabilities))
        except AttributeError:
            confidence = 1.0 # fallback if model doesn't support predict_proba
            
        # 5. Use label_encoder to convert prediction to label
        sentiment_label = label_encoder.inverse_transform([prediction_num])[0]
        
        # 6. Return JSON
        return jsonify({
            "sentiment": str(sentiment_label),
            "confidence": round(confidence, 4)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
