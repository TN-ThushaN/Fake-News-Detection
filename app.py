from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        news = request.form["news"]

        # Convert text to vector
        vector = vectorizer.transform([news])

        # Get probability
        proba = model.predict_proba(vector)[0]

        confidence = round(max(proba) * 100, 2)

        # Prediction
        if proba[1] > proba[0]:
            prediction = "Real News"
        else:
            prediction = "Fake News"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)