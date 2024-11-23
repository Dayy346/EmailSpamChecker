import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load data 
df = pd.read_csv("spam_ham.csv")

# Separate features and labels
X = df.iloc[:, :-1].values  # Features 
y = df.iloc[:, -1].values  # Labels 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=500, random_state=42
)

# Apply PCA to reduce dimensions to 50
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Save the data
np.savez("train", X=X_train_pca, y=y_train)
np.savez("test", X=X_test_pca, y=y_test)