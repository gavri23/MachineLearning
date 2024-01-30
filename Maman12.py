from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

NUMBER_OF_DIGITS = 10
TRAIN_SIZE = 60000
TEST_SIZE = 10000
STATS_ROWS = 10
STATS_COLUMNS = 4
# Load the MNIST dataset
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


# Softmax Regression Implementation

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def softmax_cost(X, y, W):
    m = X.shape[0]
    scores = np.dot(X, W.T)
    probabilities = softmax(scores)
    loss = -np.sum(np.log(probabilities[y])) / m
    grad = probabilities
    grad[y] -= 1
    grad /= m
    return loss, grad


def gradient_descent(X, y, num_classes, learning_rate):
    num_features = X.shape[1]
    W = np.zeros((num_classes, num_features))
    m = X.shape[0]
    labels = y.keys()
    for i, label in zip(range(m), labels):
        X_point = X[i]
        y_point = y[label]
        loss, grad = softmax_cost(X_point, y_point, W)
        W -= learning_rate * np.outer(grad, X_point)
    return W


l_rate = 0.01
W = gradient_descent(X_train_scaled_bias, y_train, NUMBER_OF_DIGITS, l_rate)

# Testing the Model
test_scores = np.dot(X_test_scaled_bias, W.T)
test_probabilities = softmax(test_scores)
predictions = np.argmax(test_probabilities, axis=1)

# Evaluating Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
