import numpy as np
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocesamiento_words  

# Training Code
def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

# L2 regularization
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

# Load data
train_data = np.load("train4_2.npz")
X_train, y_train = train_data["X"], train_data["y"]
test_data = np.load("test4_2.npz")
X_test, y_test = test_data["X"], test_data["y"]

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term to training and test sets
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Train the model
w = gradient_descent(X_train, y_train)

# Evaluate on training and test sets
y_train_pred = sigmoid(X_train @ w) > 0.5
train_accuracy = np.mean(y_train_pred == y_train)
print(f"Train Accuracy: {train_accuracy}")

y_pred = sigmoid(X_test @ w) > 0.5
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

# Prediction Code (using trained weights)
def load_vocab(vocab_file):
    """
    Load vocabulary from vocab.txt.
    """
    vocab = {}
    with open(vocab_file, "r") as f:
        for line in f:
            idx, word = line.strip().split(": ")
            vocab[word] = int(idx)
    return vocab


def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))


def predict_email(email_file, vocab_file, w):
    """
    Predict if an email is spam or not using the trained weights `w`.
    """
    # Load vocabulary
    vocab = load_vocab(vocab_file)

    # Preprocess the email content
    with open(email_file, "r") as f:
        email_content = f.read()
    email_words = preprocesamiento_words(email_content)

    # Ensure the feature vector has the same size as the weight vector (2001 elements)
    email_vector = np.zeros(len(w))  # Initialize feature vector to match weight vector size

    # Set the count for words in the email that exist in the vocab
    for word in email_words:
        if word in vocab and vocab[word] < len(w):  # Ensure the vocab index is within bounds
            email_vector[vocab[word]] += 1  # Increment count for words in the vocab

    # Calculate spam probability
    z = np.dot(email_vector, w)  # Perform matrix multiplication (dot product)
    spam_probability = sigmoid(z)
    is_spam = spam_probability > 0.5

    return is_spam, spam_probability


# File paths for prediction
email_file = "mail.txt"
vocab_file = "vocab.txt"

# Predict using the trained weights `w` from earlier parts
is_spam, spam_probability = predict_email(email_file, vocab_file, w)

# Output the results
if is_spam:
    print(f"The email is SPAM with a probability of {spam_probability:.2f}")
else:
    print(f"The email is NOT SPAM with a probability of {1 - spam_probability:.2f}")
