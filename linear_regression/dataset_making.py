import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
SAMPLE_NUM = 1000

X = np.random.rand(SAMPLE_NUM, 4)

# 100 samples, 3 outputs (output)
# For simplicity, we'll make the outputs a linear combination of inputs
coefficients = np.array([[3, -1, 2, 0.5], 
                         [1, 4, -2, 3], 
                         [-0.5, 2, 1, -3]])

y = X @ coefficients.T + np.random.randn(SAMPLE_NUM, 3) * 0.1  # Adding some noise

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_test.npy', X_test)
np.save('data/Y_test.npy', y_test)