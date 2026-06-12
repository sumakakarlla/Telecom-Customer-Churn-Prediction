from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("Logistic_Regression_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.form["gender_Male"]),
                float(request.form["tenure"]),
                float(request.form["MonthlyCharges"]),
                float(request.form["TotalCharges_yeo"]),
                float(request.form["PaperlessBilling_Yes"]),
                float(request.form["PaymentMethod_Credit card (automatic)"]),
                float(request.form["PaymentMethod_Electronic check"]),
                float(request.form["PaymentMethod_Mailed check"]),
                float(request.form["sims_bsnl"]),
                float(request.form["sims_jio"]),
                float(request.form["sims_vodaphone"]),
                float(request.form["Partner_odinal"]),
                float(request.form["Dependents_odinal"]),
                float(request.form["PhoneService_odinal"]),
                float(request.form["MultipleLines_odinal"]),
                float(request.form["InternetService_odinal"]),
                float(request.form["OnlineSecurity_odinal"]),
                float(request.form["OnlineBackup_odinal"]),
                float(request.form["DeviceProtection_odinal"]),
                float(request.form["TechSupport_odinal"]),
                float(request.form["StreamingTV_odinal"]),
                float(request.form["StreamingMovies_odinal"]),
                float(request.form["Contract_odinal"]),
                

            ]

            # Convert to numpy array
            features_array = np.array([features])

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            if prediction == 0:
                return render_template("index.html", prediction='Bad Customer')
            else:
                return render_template("index.html", prediction="Good Customer")

        except Exception as e:
            # If an error occurs during POST, render the template with the error message
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction) # <--- Added return here for error handling

    # This handles the initial GET request (when the page is loaded)
    # and any scenario where the POST block didn't execute or encountered an error.
    return render_template("index.html", prediction=prediction) # <--- ADDED REQUIRED RETURN STATEMENT

if __name__ == "__main__":
    app.run(debug=True)
