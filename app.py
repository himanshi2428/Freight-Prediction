from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data.csv")

# Handle missing values and preprocess date
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.strftime('%y%j')

# Create a mapping of original state names to their corresponding airports without duplicates
state_airport_dict = df.groupby('state_name')['airport'].apply(lambda x: list(set(x))).to_dict()

# Initialize LabelEncoder for state names
le_state = LabelEncoder()
le_state.fit(list(state_airport_dict.keys()))

# Create a dictionary to map encoded state names to airports
state_airport_dict_encoded = {}
for key, value in state_airport_dict.items():
    encoded_key = le_state.transform([key])[0]
    state_airport_dict_encoded[encoded_key] = value

# Prepare model for predictions
X = df[["date", "state_name", "airport"]]
y = df["freight"]

# Encode categorical columns
df['state_name_encoded'] = le_state.transform(df['state_name'])
le_airport = LabelEncoder()
df['airport_encoded'] = le_airport.fit_transform(df['airport'])
X = df[["date", "state_name_encoded", "airport_encoded"]]
y = df["freight"]

# Train the model
rf = RandomForestRegressor()
rf.fit(X, y)

@app.route('/')
def index():
    state_names = list(state_airport_dict.keys())
    return render_template('index.html', states=state_names)

@app.route('/get_airports/<state_name>', methods=['GET'])
def get_airports(state_name):
    try:
        encoded_state_name = le_state.transform([state_name])[0]
        airports = state_airport_dict_encoded.get(encoded_state_name, [])
    except ValueError:
        airports = []

    return jsonify(airports)

@app.route('/', methods=['POST'])
def predict():
    date = request.form['date']
    state_name = request.form['state']
    airport_name = request.form['airport']
    
    # Encode inputs for prediction
    try:
        encoded_state = le_state.transform([state_name])[0]
        encoded_airport = le_airport.transform([airport_name])[0]
        date_encoded = int(pd.to_datetime(date).strftime('%y%j'))

        # Prepare input for prediction
        input_data = np.array([[date_encoded, encoded_state, encoded_airport]])
        predicted_freight = rf.predict(input_data)[0]
        
        return render_template('index.html', states=list(state_airport_dict.keys()), predicted_freight=predicted_freight)

    except Exception as e:
        error = str(e)
        return render_template('index.html', states=list(state_airport_dict.keys()), error=error)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
