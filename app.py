from flask import Flask, render_template, request
import pickle
import numpy as np
import traceback  # Add this

app = Flask(__name__)

# Load model
model = pickle.load(open("Liver2.pkl", "rb"))

@app.route('/')
def home():
    return render_template("liver.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])
        result = model.predict(final_input)
        outcome = "Liver Disease Detected" if result[0] == 1 else "No Liver Disease"
        return render_template("result.html", prediction_text=outcome)
    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()
        return "Something went wrong. Check your Render logs.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

