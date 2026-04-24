import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ===== Load dataset =====
data = pd.read_csv('bmw_global_sales_dataset.csv')
# ===== Features & target =====
X = data[['dealership_count']]
y = data['units_sold']

# ===== Scaling =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ===== Train =====
model = LinearRegression()
model.fit(X_train, y_train)

# ===== Predict =====
y_pred = model.predict(X_test)

# ===== Evaluate =====
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# ===== Save model =====
joblib.dump(model, 'bmw_model.pkl')
joblib.dump(scaler, 'bmw_scaler.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')

