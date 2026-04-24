from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# ===== Load model =====
model = joblib.load('bmw_model.pkl')
scaler = joblib.load('bmw_scaler.pkl')
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ===== Get input =====
        dealership_count = request.form.get('dealership_count')

        if dealership_count is None or dealership_count == "":
            return render_template('index.html', prediction_text="Please enter a value")

        dealership_count = float(dealership_count)

        # ===== Preprocess =====
        input_data = np.array([[dealership_count]])
        input_scaled = scaler.transform(input_data)

        # ===== Predict =====
        prediction = model.predict(input_scaled)[0]

        # ===== Plot =====
        plt.figure()

        plt.scatter(X_test, y_test, label="Actual")
        plt.scatter(X_test, model.predict(X_test), label="Predicted")

        # 🔥 Important fix
        plt.scatter(input_scaled[0][0], prediction, s=100, label="Your Prediction")

        plt.xlabel("Dealership Count (scaled)")
        plt.ylabel("Units Sold")
        plt.legend()

        # ===== Save plot =====
        plt.savefig('static/plot.png')  # ⚠️ static folder must exist
        plt.close()


    except Exception as e:
        print("ERROR:", e)
        return render_template(
            'index.html',
            prediction_text=f'Predicted Units Sold: {prediction:.2f}'
        )


if __name__ == "__main__":
    app.run(debug=True)