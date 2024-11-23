# Email Spam Checker

## Project Overview

### Problem Statement
Email spam poses a significant challenge globally, impacting individuals and organizations by overloading inboxes, wasting time, and potentially exposing users to phishing attacks or malware.

### Solution
This project implements a machine learning-based email spam checker that classifies incoming emails as either "spam" or "ham" (legitimate), providing a robust solution to automate spam detection.

### Project Goal
The primary goal is to demonstrate the development and evaluation of a machine learning model for a real-world application. This includes showcasing expertise in data preprocessing, feature engineering, model training, evaluation, and prediction.

---

## Data

### Source
The project utilizes a subset of the **Enron Spam Dataset**, a publicly available collection of emails from the Enron Corporation.

### Description
The dataset includes thousands of emails labeled as "spam" or "ham," comprising subject lines and message bodies.

### Preprocessing
To prepare the data for machine learning, the following preprocessing steps are applied:
1. **Lowercasing**: Standardizes text to lowercase for uniformity.
2. **HTML Tag Removal**: Strips unnecessary HTML tags.
3. **Normalization**: Replaces URLs, email addresses, numbers, and dollar signs with generic tokens (e.g., `httpaddr`, `emailaddr`, `number`, `dollar`) to reduce data dimensionality.
4. **Punctuation Removal**: Eliminates non-alphanumeric characters to focus on the text content.
5. **Stemming**: Applies the Snowball Stemmer to reduce words to their root form.
6. **Stop Word Removal**: Filters out common words like "the" or "is" to enhance meaningful feature extraction.
7. **TF-IDF Feature Extraction**: Converts text into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method, emphasizing key terms indicative of spam or ham.

### Output
- **`spam_ham.csv`**: Contains preprocessed features and labels.
- **`vocab.txt`**: Lists vocabulary terms and their corresponding indices.

---

## Model Training

### Algorithm
**Logistic Regression** is used for its simplicity, effectiveness, and interpretability in binary classification tasks. The model incorporates L2 regularization to prevent overfitting.

### Implementation
1. **Feature Scaling**: Standardizes features to zero mean and unit variance using `StandardScaler`.
2. **Gradient Descent**: Optimizes model weights by minimizing the loss function (negative log-likelihood with L2 regularization).
3. **Regularization**: Penalizes large weights to enhance model generalization.
4. **Train-Test Split**: Splits the data into training and testing sets, ensuring robust performance evaluation.
5. **Dimensionality Reduction**: Uses Principal Component Analysis (PCA) to reduce features to the top 50 components.

### Evaluation
The model achieves:
- **Training Accuracy**: 96.29%
- **Testing Accuracy**: 95.60%

---

## Prediction

### Workflow
1. **Load Model and Vocabulary**: Loads trained weights and vocabulary from `vocab.txt`.
2. **Preprocess Input**: Applies the same preprocessing steps to new emails from `mail.txt`.
3. **Spam Classification**: Computes the probability of spam and outputs the classification.

---

## Usage Instructions

### Steps
1. Paste email content into `mail.txt`.
2. Run the prediction script:
   ```bash
   python predict.py
