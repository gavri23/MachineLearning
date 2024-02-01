from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

'''
This section prepares the data before training and testing
'''
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

''' 
Softmax Regression Implementation:

Brief summary for Softmax Regression:
The algorithm is divided to several methods

1) softmax: The math function itself that will also be used for the tested data, it calculates the probabilities of what
 digit it's likely to be.
 
2) softmax_cost: calculates the loss and the gradient of the softmax function.

3) gradient_descent: The training method, it depends on the number of tries (epochs) and learning rate.
   the epochs are for how many times we want to repeat the training session and the learning rate used for controlling 
   the step size during the training and optimizing the algorithm convergence to the minimum of the loss function.
   The learning rate and the epochs are the keys of the successfulness of the test. 
   If we have very small learning rate, we wont get to our goal which is to minimize the loss function. 
   if we have a higher learning rate can lead to faster convergence but may risk overshooting the minimum.
   The epochs helps to balance to learning rate so if we have a small learning rate so by repeating the test more and 
   more may help us converge to the minimum and vice versa.
'''


def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


#
def softmax_cost(x_data, y_data, weights):
    m = x_data.shape[0]
    scores = np.dot(x_data, weights.T)
    probabilities = softmax(scores)
    loss = -np.sum(np.log(probabilities[y_data])) / m

    # update the gradient
    grad = probabilities
    grad[y_data] -= 1
    grad /= m
    return loss, grad


def gradient_descent(x_training, y_training, num_classes, learning_rate, tries):
    num_features = x_training.shape[1]
    weights = np.zeros((num_classes, num_features))
    nof_training_images = x_training.shape[0]
    labels = y_training.keys()
    for epoch in range(tries):
        for idx, label in zip(range(nof_training_images), labels):
            x_point = x_training[idx]
            y_point = y_training[label]
            loss, grad = softmax_cost(x_point, y_point, weights)
            weights -= learning_rate * np.outer(grad, x_point)  # update the weights.
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    print(f"Converged. Final Loss: {loss:.6f}")
    return weights


l_rate = 0.001
epochs = 150
weight_mat = gradient_descent(x_training=X_train_scaled_bias, y_training=y_train, num_classes=NUMBER_OF_DIGITS,
                     learning_rate=l_rate,
                     tries=epochs)

# Testing the Model
test_scores = np.dot(X_test_scaled_bias, weight_mat.T)
test_probabilities = softmax(test_scores)
predictions = np.argmax(test_probabilities, axis=1)

# Evaluating Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
digit_conf_matrix = confusion_matrix(y_test.values, predictions)

# Now we can get the TP,FP,FN,TN for each digit from the Confusion Mat above.
# The rows are the digits and the columns are the 4 classes:
# Column 0 - TP
# Column 1 - FP
# Column 2 - FN
# Column 3 - TN
# Initialize 10*4 mat to hold all the stats data, each digit has 4 stats as written above
stats = np.zeros((STATS_ROWS, STATS_COLUMNS))

mat_sum = np.sum(digit_conf_matrix)
for i in range(NUMBER_OF_DIGITS):
    stats[i][0] = digit_conf_matrix[i, i]  # True Positives
    stats[i][1] = np.sum(digit_conf_matrix[:, i]) - digit_conf_matrix[i, i]  # False Positives
    stats[i][2] = np.sum(digit_conf_matrix[i, :]) - digit_conf_matrix[i, i]  # False Negatives
    stats[i][3] = mat_sum - stats[i][2] - stats[i][1] - stats[i][0]  # True Negatives
    print(f"digit {i} stats are:")
    print(f"TP,FP,FN,TN is: {stats[i][0]},{stats[i][1]},{stats[i][2]},{stats[i][3]}")
    acc = (stats[i][0] + stats[i][3]) / np.sum(stats[i, :])
    print(f"ACC is {acc}")
    tpr = stats[i][0] / (stats[i][0] + stats[i][2])
    print(f"TPR is {tpr}")
    tnr = stats[i][3] / (stats[i][3] + stats[i][1])
    print(f"TNR is {tnr}\n")
