from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import sqlite3

# Load the trained model
model = joblib.load('model/cancer_model.pkl')

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messaging

# Set up database
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

# Insert data into database
def insert_data(user_name, features, prediction):
    conn = sqlite3.connect('database/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            user_name, mean_concavity, worst_area, worst_concave_points, 
            worst_radius, area_error, worst_concavity, mean_concave_points, 
            worst_symmetry, radius_error, worst_texture, prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_name, *features, prediction))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
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

            input_features = np.array([[
                mean_concavity, worst_area, worst_concave_points, worst_radius,
                area_error, worst_concavity, mean_concave_points, worst_symmetry,
                radius_error, worst_texture
            ]])

            prediction_proba = model.predict(input_features)
            prediction = "Benign" if prediction_proba == 1 else "Malignant"

            insert_data(user_name, [
                mean_concavity, worst_area, worst_concave_points, worst_radius,
                area_error, worst_concavity, mean_concave_points, worst_symmetry,
                radius_error, worst_texture
            ], prediction)

            flash("✅ Prediction saved successfully!", "success")

        except ValueError:
            flash("⚠️ Please enter valid numeric values in all fields.", "danger")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
