import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)

# 100 samples, 4 features (input)
X = np.random.rand(100, 4)

# 100 samples, 3 outputs (output)
# For simplicity, we'll make the outputs a linear combination of inputs
coefficients = np.array([[3, -1, 2, 0.5], 
                         [1, 4, -2, 3], 
                         [-0.5, 2, 1, -3]])

y = X @ coefficients.T + np.random.randn(100, 3) * 0.1  # Adding some noise

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show coefficients and intercept
print("Model coefficients:")
print(model.coef_)
print("\nModel intercepts:")
print(model.intercept_)

# Optional: Plot actual vs predicted values for the first output dimension
plt.scatter(y_test[:, 0], y_pred[:, 0])
plt.xlabel('Actual values (1st output)')
plt.ylabel('Predicted values (1st output)')
plt.title('Actual vs Predicted (1st Output)')
plt.show()