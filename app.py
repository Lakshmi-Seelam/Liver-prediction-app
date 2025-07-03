from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("Liver2.pkl", "rb"))

@app.route('/')
def home():
    return render_template("liver.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])
        result = model.predict(final_input)
        outcome = "Liver Disease Detected" if result[0] == 1 else "No Liver Disease"
        return render_template("result.html", prediction_text=outcome)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
