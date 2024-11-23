
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
train_data = np.load("train4_2.npz")
X_train, y_train = train_data["X"], train_data["y"]
test_data = np.load("test4_2.npz")
X_test, y_test = test_data["X"], test_data["y"]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Add bias 
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#  L2 regularization
def nll(X, y, w, lambda_reg):
    z = X @ w
    return -np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))) + (lambda_reg / 2) * np.sum(w**2)

def gradient_descent(X, y, learning_rate=1e-4, tolerance=1e-6, max_iter=10000, lambda_reg=0.01):
    w = np.zeros(X.shape[1])
    prev_nll = float("inf")
    for _ in range(max_iter):
        z = X @ w
        gradient = (X.T @ (sigmoid(z) - y)) / X.shape[0] + lambda_reg * w
        w -= learning_rate * gradient
        current_nll = nll(X, y, w, lambda_reg)
        if abs(prev_nll - current_nll) < tolerance:
            break
        prev_nll = current_nll
    return w

# Train 
w = gradient_descent(X_train, y_train)


y_train_pred = sigmoid(X_train @ w) > 0.5

# Calculate train accuracy
train_accuracy = np.mean(y_train_pred == y_train)
print(f"Train Accuracy: {train_accuracy}")


y_pred = sigmoid(X_test @ w) > 0.5

# Calculate test accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")
