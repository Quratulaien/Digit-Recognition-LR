import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import os

def load_and_preprocess_data():
    """
    Load the digits dataset, preprocess it, and return train/test splits.
    """
    # Load the digits dataset
    digits = load_digits()
    X = digits.data  # Already flattened 8x8 images (64 features)
    y = digits.target  # Labels (0-9)

    # No missing values in this dataset, but check anyway
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")

    # Normalization: Scale pixel values to [0,1]
    X = X / 16.0  # Since pixels are 0-16

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train Logistic Regression model.
    """
    # Logistic Regression parameters:
    # - C: Inverse of regularization strength (smaller values specify stronger regularization)
    # - max_iter: Maximum number of iterations for convergence
    # - random_state: For reproducibility
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1, cm

def save_model(model, filename='logistic_regression_model.pkl'):
    """
    Save the trained model to a file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def load_model(filename='logistic_regression_model.pkl'):
    """
    Load a trained model from file.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)