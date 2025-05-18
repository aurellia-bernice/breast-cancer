from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import numpy as np
import sqlite3
import joblib

# Load the trained model
model = joblib.load('model/cancer_model.pkl')

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for flash messages

# Database setup function
def init_db():
    conn = sqlite3.connect('database/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            mean_concavity REAL,
            worst_area REAL,
            worst_concave_points REAL,
            worst_radius REAL,
            area_error REAL,
            worst_concavity REAL,
            mean_concave_points REAL,
            worst_symmetry REAL,
            radius_error REAL,
            worst_texture REAL,
            prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Insert the data into the database
def insert_data(user_name, features, prediction):
    conn = sqlite3.connect('database/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (user_name, mean_concavity, worst_area, worst_concave_points, 
        worst_radius, area_error, worst_concavity, mean_concave_points, worst_symmetry, 
        radius_error, worst_texture, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_name, *features, prediction))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            # Get the input values from the form
            user_name = request.form['user_name']
            mean_concavity = float(request.form['mean_concavity'])
            worst_area = float(request.form['worst_area'])
            worst_concave_points = float(request.form['worst_concave_points'])
            worst_radius = float(request.form['worst_radius'])
            area_error = float(request.form['area_error'])
            worst_concavity = float(request.form['worst_concavity'])
            mean_concave_points = float(request.form['mean_concave_points'])
            worst_symmetry = float(request.form['worst_symmetry'])
            radius_error = float(request.form['radius_error'])
            worst_texture = float(request.form['worst_texture'])

            # Create an array of the input features
            input_features = np.array([[
                mean_concavity, worst_area, worst_concave_points, worst_radius,
                area_error, worst_concavity, mean_concave_points, worst_symmetry,
                radius_error, worst_texture
            ]])

            # Make the prediction
            prediction_proba = model.predict(input_features)
            prediction = "Malignant" if prediction_proba == 1 else "Benign"

            # Insert the data into the database
            insert_data(user_name, [
                mean_concavity, worst_area, worst_concave_points, worst_radius,
                area_error, worst_concavity, mean_concave_points, worst_symmetry,
                radius_error, worst_texture
            ], prediction)

            flash("Prediction saved successfully!", "success")  # Success message

        except ValueError:
            flash("Invalid input. Please enter valid numeric values for all fields.", "danger")  # Error message

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    init_db()  # Initialize the database when the app starts
    app.run(debug=True)
