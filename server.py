from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import asyncio
from extractFeatures import load_reference_urls, extract_features, extract_all_features

app = Flask(__name__)

# Get the current directory of the server.py script
current_directory = os.path.dirname(__file__)

# Define the path to the model file relative to the current directory
model_path = os.path.join(current_directory, 'train_model', 'RandomForest_selfCollect.pkl')

# Load the model from the pkl file
model = joblib.load(model_path)

# Retrieve feature names from the model
feature_names = model.feature_names_in_

# Define the path to the reference URLs file relative to the current directory
reference_urls_path = os.path.join(current_directory, 'url_reference', 'ReferenceURLs.csv')

# Load reference URLs
reference_urls = load_reference_urls(reference_urls_path)

# Define the feature extraction function
def extract_features_for_url(url):
    urls_to_predict = pd.DataFrame({'URL': [url]})
    extracted_features = extract_features(urls_to_predict, reference_urls)
    return extracted_features

@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data['url']
    print(f"Received URL: {url}")
    features_df = extract_features_for_url(url)

    # Ensure we extract all additional features asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    additional_features_df = loop.run_until_complete(extract_all_features(features_df))
    loop.close()

    # Merge the extracted features with the additional features
    complete_features_df = features_df.merge(additional_features_df, on='URL')
    complete_features_df = complete_features_df.drop(columns=['URL'])

    print("Extracted features:", complete_features_df.columns.tolist())
    print("Expected features:", feature_names.tolist())

    missing_features = set(feature_names) - set(complete_features_df.columns)
    if missing_features:
        print(f"Missing features: {missing_features}")
        return jsonify({'error': 'Missing features', 'missing_features': list(missing_features)}), 400

    try:
        prediction = model.predict(complete_features_df[feature_names])[0]
    except KeyError as e:
        print(f"KeyError during prediction: {e}")
        return jsonify({'error': str(e)}), 500

    result = {'isPhishing': bool(prediction)}
    print(f"Prediction result: {result}")
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
