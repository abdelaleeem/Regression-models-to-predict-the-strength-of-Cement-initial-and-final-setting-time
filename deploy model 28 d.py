import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import flask
from flask import Flask, request, jsonify , render_template


# Load the trained model
with open("random_forest_28d_strength_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = ['CaO', 'SO3', 'A.Eq', 'C4AF', '2 d', 'Blaine', '45 m R', '90 m R', 'Cao_free']
    input_data = [float(request.form[feature]) for feature in features]

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=features)

    # Predict using the model
    predictions = model.predict(input_df)

    # Return the prediction result
    return render_template('index.html', prediction=predictions[0])

if __name__ == '__main__':
    app.run(debug=True)

