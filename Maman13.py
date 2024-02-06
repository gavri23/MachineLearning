import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class LinearRegressionLeastSquares:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add a bias term to the features
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Calculate weights using the least squares method
        XTX = np.dot(X_bias.T, X_bias)
        XTY = np.dot(X_bias.T, y)
        self.weights = np.dot(np.linalg.inv(XTX), XTY)

    def predict(self, X):
        # Add a bias term to the features
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Make predictions using the calculated weights
        return np.dot(X_bias, self.weights)


'''
This section prepares the data before training and testing
'''
NUMBER_OF_DIGITS = 10
TRAIN_SIZE = 60000
TEST_SIZE = 10000
STATS_ROWS = 10
STATS_COLUMNS = 4
# Load the MNIST dataset
print("preparing the data")
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# Prepare the data
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Normalize the input data to range [0, 1]
X /= 255.0
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# Add bias term to the input features
X_train_scaled_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_scaled_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
# Initialize and fit the linear regression model
linear_reg = LinearRegressionLeastSquares()
linear_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_reg.predict(X_test)

# Round predictions to the nearest integer to get digit labels
y_pred_rounded = np.round(y_pred).astype('int')

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rounded)
print(f'Accuracy: {accuracy}')
