from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
# Generate Dataset
X, y = make_regression(n_samples=500,
n_features=1,
noise=15,
random_state=42)
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Train Mini-Batch Gradient Descent Model
model = SGDRegressor(
max_iter=1000,
learning_rate='constant',
eta0=0.01,
random_state=42
)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
