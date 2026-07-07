from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

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

    sepal_length = float(request.form["sl"])
    sepal_width = float(request.form["sw"])
    petal_length = float(request.form["pl"])
    petal_width = float(request.form["pw"])

    prediction = model.predict([[sepal_length,
                                 sepal_width,
                                 petal_length,
                                 petal_width]])

    result = flower_names[prediction[0]]

    return render_template("index.html",
                           prediction=result)


if __name__ == "__main__":
    app.run(debug=True)