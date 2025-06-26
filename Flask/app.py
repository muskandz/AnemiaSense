from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Loading saved model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        hemoglobin = int(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])

        features = np.array([[gender, hemoglobin, mch, mchc, mcv]])
        prediction = model.predict(features)

        result = "Anemia Detected" if prediction[0] == 1 else "Normal"
        return render_template('predict.html', prediction=result)
                               
if __name__ == '__main__':
    app.run(debug=True)