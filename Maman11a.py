from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

NUMBER_OF_DIGITS = 10
TRAIN_SIZE = 60000
TEST_SIZE = 10000
STATS_ROWS = 10
STATS_COLUMNS = 4
PROGRESS_TRAINING_PERCENTAGE = TRAIN_SIZE / 100
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# Prepare the data
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Normalize the input data to range [0, 1]
X /= 255.0
X['bias'] = np.ones(X.shape[0])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=42)


def one_hot_encode(labels, num_classes):
    one_hot_labels = -np.ones((len(labels), num_classes))
    for idx, label in enumerate(labels):
        one_hot_labels[idx, label] = 1
    return one_hot_labels


HOT_VECTORS = one_hot_encode([x for x in range(10)], 10)


class MultiClassPerceptron:
    def __init__(self, num_classes, input_size):
        self.num_classes = num_classes
        self.input_size = input_size
        self.weights = np.zeros((self.input_size, self.num_classes))  # +1 for bias term

    def predict_test(self, inputs):
        scores = np.dot(inputs, self.weights)  # Weighted sum
        return np.argmax(scores)  # Class with the highest score

    def predict_learning(self, inputs):
        scores = np.dot(inputs, self.weights)  # Weighted sum
        return np.sign(scores)  # Class with the highest score

    def train(self, inputs, labels):
        images_tags = inputs.keys()
        counter = 0
        for label, tag in zip(labels, images_tags):
            pixels = inputs[tag]
            prediction = self.predict_learning(pixels)
            correction = HOT_VECTORS[label] - prediction
            self.weights += np.outer(pixels, correction)
            counter += 1
            if counter % PROGRESS_TRAINING_PERCENTAGE == 0:  # size of training data is 60000 so 1 percent is 600
                print(f"progress of learning so far {counter / PROGRESS_TRAINING_PERCENTAGE}%")


# Create and train the multi-class perceptron model
model = MultiClassPerceptron(NUMBER_OF_DIGITS, X.shape[1])
print("start training")
model.train(X_train.T, y_train)
print("end training")
# Make predictions
print("start testing")
test_mat = X_test.T
test_keys = X_test.T.keys()
predictions = [model.predict_test(test_mat[tag]) for tag in test_keys]
print("end testing")

# Calculate accuracy of whole predictions
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")

# Create the Confusion Matrix which holds in each cell the amount of the true vs predicted.
# for example if digit_conf_matrix[4][3] = 50 it means that the label 4 was predicted as 3 50 times.
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

print("END PART A")
