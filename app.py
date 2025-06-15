
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    input_features = [float(x) for x in request.form.values()]
    features_array = np.array([input_features])
    prediction = model.predict(features_array)
    output = round(prediction[0], 2)
    return render_template("index.html", prediction_text=f"Predicted House Price: ${output}k")

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    input_features = [float(x) for x in request.form.values()]
    features_array = np.array([input_features])
    prediction = model.predict(features_array)
    output = round(prediction[0], 2)
    return render_template("index.html", prediction_text=f"Predicted House Price: ${output}k")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
