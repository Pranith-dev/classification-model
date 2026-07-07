from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Class names
flower_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the HTML form
        sepal_length = float(request.form["sl"])
        sepal_width = float(request.form["sw"])
        petal_length = float(request.form["pl"])
        petal_width = float(request.form["pw"])

        # Predict
        prediction = model.predict([[sepal_length,
                                     sepal_width,
                                     petal_length,
                                     petal_width]])

        result = flower_names[prediction[0]]

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
