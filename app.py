from flask import Flask, request, render_template
import numpy as np
import pickle
import sqlite3
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from io import BytesIO
import base64

# Loading the ML model
model = pickle.load(open("model.pkl", "rb"))

# Loading the standard scaler
standard_scaler = pickle.load(open("standard_scaler.pkl", "rb"))

# Initiating sql database
con = sqlite3.connect("history.db")
print("Database opened successfully")

# Creating table in database
con.execute("""create table IF NOT EXISTS prediction_history(
                Pregnancies REAL NOT NULL, 
                Glucose REAL NOT NULL,
                BloodPressure REAL NOT NULL,
                SkinThickness REAL NOT NULL,
                Insulin REAL NOT NULL,
                BMI REAL NOT NULL,
                DPF REAL NOT NULL,
                Age INTEGER NOT NULL,
                Outcome INTEGER NOT NULL,
                Probability REAL NOT NULL)""")

print("Table created successfully")

# Backend
app = Flask(__name__)


def get_features():
    preg = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bp = float(request.form['bloodpressure'])
    st = float(request.form['skinthickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = float(request.form['age'])

    features = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

    return features


def store_to_database(features, prediction, positive_probability):
    with sqlite3.connect("history.db") as con:
        cur = con.cursor()
        cur.execute("""INSERT into prediction_history 
                        (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age, Outcome, Probability) 
                        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (features[0][0], features[0][1], features[0][2], features[0][3], features[0][4],
                     features[0][5], features[0][6], features[0][7], int(prediction), positive_probability))
        con.commit()
        print("Record entered in database")


def predict(scaled_features):
    prediction = int(model.predict(scaled_features))
    probability = model.predict_proba(scaled_features)
    positive_probability = probability[0][1]

    return prediction, positive_probability


def report(prediction, positive_probability):
    if prediction:
        prediction_text = "Diabetic"
        prediction_image = "diabetic.png"
    else:
        prediction_text = "Non-Diabetic"
        prediction_image = "nondiabetic.png"

    probability_text = "Likelihood to be diabetic"

    positive_percentage = positive_probability * 100
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = 'black'

    labels = ['Diabetic', 'Non Diabetic']
    sizes = [positive_percentage, 100 - positive_percentage]
    plt.figure(figsize=(2, 2))
    my_circle = plt.Circle((0, 0), 0.7, color='#e7e7e7')
    x = 0
    y = -0.1

    label = plt.annotate(str(int(positive_percentage)) + '%', xy=(x, y), fontsize=16, ha="center")
    patches = plt.pie(sizes, startangle=90, labeldistance=1.1, pctdistance=0.845)

    plt.axis('equal')
    plt.gca().add_artist(my_circle)

    # Store in temp buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True)

    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return prediction_image, prediction_text, f"data:image/png;base64,{data}", probability_text


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def diagnose():
    # Getting features and scaling them
    features = get_features()
    scaled_features = standard_scaler.transform(features)
    print(scaled_features)

    # Making Prediction
    prediction, positive_probability = predict(scaled_features)

    # Store the features and predictions to database
    store_to_database(features, prediction, positive_probability)

    # Reporting the diagnosis
    prediction_image, prediction_text, probability_image, probability_text = report(prediction, positive_probability)

    return render_template("index.html", prediction_image=prediction_image, prediction_text=prediction_text,
                           probability_image=probability_image, probability_text=probability_text,
                           scroll="result", results="Results:")


if __name__ == "__main__":
    app.run()
