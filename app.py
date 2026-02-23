from flask import Flask, request, render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# -------------------------------
# Load column data
# -------------------------------
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# -------------------------------
# Price Prediction Function
# -------------------------------
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location.lower() in data_columns:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1

    price = model.predict([x])[0]
    return round(price, 2)

# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def home():
    locations = data_columns[3:]
    return render_template('index.html', locations=locations)

# -------------------------------
# Prediction Route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location', 'none')
        sqft = float(request.form.get('sqft', 0))
        bath = int(request.form.get('bath', 0))
        bhk = int(request.form.get('bhk', 0))

        price = predict_price(location, sqft, bath, bhk)

        # Convert Lakhs → Crores
        if price >= 100:
            formatted_price = f"₹ {round(price/100, 2)} Cr"
        else:
            formatted_price = f"₹ {price} Lakhs"

        return render_template(
            'index.html',
            prediction_text=formatted_price,
            locations=data_columns[3:]
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="Error calculating price",
            locations=data_columns[3:]
        )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)