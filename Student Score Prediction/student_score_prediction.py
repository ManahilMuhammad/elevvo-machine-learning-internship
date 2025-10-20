# --> BEGINNING OF: importing libraries
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import kagglehub
# <-- END OF: importing libraries

# --> BEGINNING OF: configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
POLY_DEGREE = 2  # for polynomial regression
EXTRA_FEATURES = []  # e.g., ["sleep_hours", "attendance"]
# <-- END OF: configuration

# --> BEGINNING OF: loading data
path = kagglehub.dataset_download("lainguyn123/student-performance-factors")

# locating file
CSV_PATH = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        CSV_PATH = os.path.join(path, file)
        break

# error handling
if not CSV_PATH:
    raise FileNotFoundError("File not found")
# <-- END OF: loading data

# --> BEGINNING OF: function to split data into training and testing sets
def split_data(X, y, test_size=0.2, seed=42):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    test_n = int(math.ceil(n * test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
# <-- END OF: function to split data into training and testing sets

# --> BEGINNING OF: function to add bias and column of 1s to absorb bias
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
# <-- END OF: function to add bias and column of 1s to absorb bias

# --> BEGINNING OF: function to return weights including bias
def fit_linear_regression(X, y):
    # add bias
    Xb = add_bias(X)
    
    # compute vector containing weights including bias
    try:
        theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
        
    # exception handling
    except np.linalg.LinAlgError:
        theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        
    return theta
# <-- END OF: function to return weights including bias

# --> BEGINNING OF: function to compute predictions using bias and weights
def predict_with_theta(theta, X):
    return add_bias(X) @ theta
# <-- END OF: function to compute predictions using bias and weights

# --> BEGINNING OF: function to expand data to include nonlinear features
def polynomial_features(X, degree=2):
    # only perform polynomial expansion if degree is not 1
    if degree == 1:
        return X
    
    # initialise list of original and squared features
    feats = [X, X**2]
    
    # only generate interaction terms if there is more than one feature
    if X.shape[1] > 1:
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
    
    return np.hstack(feats)
# <-- END OF: function to expand data to include nonlinear features

# --> BEGINNING OF: function to compute mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
# <-- END OF: function to compute mean squared error

# --> BEGINNING OF: function to compute mean absolute error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
# <-- END OF: function to compute mean absolute error

# --> BEGINNING OF: function to compute R^2 score
def r2(y_true, y_pred):
    # compute residual sum of squares
    ss_res = np.sum((y_true - y_pred)**2)
    
    # compute total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    # compute and return R^2 score
    # perfect prediction if 1 returned
    # predictions always equal to mean if 0 returned
    # predictions worse than mean if less than 0 returned
    return 1 - ss_res / ss_tot
# <-- END OF: function to compute R^2 score

# --> BEGINNING OF: function to visualise hours studied vs exam score, and the distribution of exam scores
def visualise(df):
    # plot exam score against hours studied
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.6)
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Exam Score vs Hours Studied")
    plt.grid(True)
    plt.show()

    # plot histogram of exam scores
    plt.hist(df["Exam_Score"].dropna(), bins=20)
    plt.title("Distribution of Exam Scores")
    plt.show()
# <-- END OF: function to visualise hours studied vs exam score, and the distribution of exam scores

# --> BEGINNING OF: function to plot predictions vs actual values
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.show()
# <-- END OF: function to plot predictions vs actual values

# --> BEGINNING OF: function to run programme
def run_pipeline():
    # load data
    df = pd.read_csv(CSV_PATH)

    # clean data
    features = ["Hours_Studied"] + [f for f in EXTRA_FEATURES if f in df.columns]
    df = df.dropna(subset=features + ["Exam_Score"])
    X = df[features].to_numpy(float)
    y = df["Exam_Score"].to_numpy(float)

    # visualise data
    visualise(df)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED)

    # normalise data
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    # predict using linear regression
    theta_lin = fit_linear_regression(X_train_s, y_train)
    y_pred_test = predict_with_theta(theta_lin, X_test_s)

    print("------------------------")
    print("Linear Regression:")
    print("MSE:", mse(y_test, y_pred_test))
    print("MAE:", mae(y_test, y_pred_test))
    print("R² :", r2(y_test, y_pred_test))
    print("------------------------")
    
    # plot predictions vs actual values using linear regression
    plot_predictions(y_test, y_pred_test, "Linear Regression Predictions")

    # predict using polynomial regression
    X_train_poly = polynomial_features(X_train_s, POLY_DEGREE)
    X_test_poly = polynomial_features(X_test_s, POLY_DEGREE)
    theta_poly = fit_linear_regression(X_train_poly, y_train)
    y_pred_poly = predict_with_theta(theta_poly, X_test_poly)

    print("------------------------")
    print("Polynomial Regression:")
    print("MSE:", mse(y_test, y_pred_poly))
    print("MAE:", mae(y_test, y_pred_poly))
    print("R² :", r2(y_test, y_pred_poly))
    print("------------------------")
    
    # plot predictions vs actual values using polynomial regression
    plot_predictions(y_test, y_pred_poly, f"Polynomial Regression (deg={POLY_DEGREE})")

    artifacts = {
        "features": features,
        "theta_lin": theta_lin,
        "theta_poly": theta_poly,
        "mean": mean,
        "std": std,
        "poly_degree": POLY_DEGREE
    }
    return artifacts
# <-- END OF: function to run programme

# --> BEGINNING OF: function to create UI
class PredictorUI:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.root = tk.Tk()
        self.root.title("✎ Student Score Predictor")
        self.entries = {}
        self.create_ui()

    def create_ui(self):
        ttk.Label(self.root, text="Enter your data", font=("Arial", 13, "bold")).grid(column=0, row=0, columnspan=2, pady=10)

        for i, f in enumerate(self.artifacts["features"]):
            ttk.Label(self.root, text=f).grid(column=0, row=i + 1, sticky="w", padx=5, pady=2)
            e = ttk.Entry(self.root, width=15)
            e.grid(column=1, row=i + 1, padx=5, pady=2)
            self.entries[f] = e

        # buttons to control the type of regression to use to make predictions
        self.model_choice = tk.StringVar(value="linear")
        ttk.Radiobutton(self.root, text="Linear", variable=self.model_choice, value="linear").grid(column=0, row=len(self.entries) + 2)
        ttk.Radiobutton(self.root, text="Polynomial", variable=self.model_choice, value="poly").grid(column=1, row=len(self.entries) + 2)

        # button to compute prediction
        ttk.Button(self.root, text="Predict Score", command=self.predict).grid(column=0, row=len(self.entries) + 3, columnspan=2, pady=10)
        
        self.result = ttk.Label(self.root, text="", font=("Arial", 12))
        self.result.grid(column=0, row=len(self.entries) + 4, columnspan=2)

    def predict(self):
        try:
            # process input
            vals = [float(self.entries[f].get()) for f in self.artifacts["features"]]
            X = np.array(vals).reshape(1, -1)
            Xs = (X - self.artifacts["mean"]) / self.artifacts["std"]

            # predict using linear regression
            if self.model_choice.get() == "linear":
                pred = predict_with_theta(self.artifacts["theta_lin"], Xs)
            
            # predict using polynomial features
            else:
                Xp = polynomial_features(Xs, self.artifacts["poly_degree"])
                pred = predict_with_theta(self.artifacts["theta_poly"], Xp)

            # display result
            self.result.config(text=f"Predicted Final Score: {pred[0]:.4f}")
        
        # exception handling
        except Exception as e:
            messagebox.showerror("Error", str(e))
# <-- END OF: function to create UI

# --> BEGINNING OF: function to start the pipeline
if __name__ == "__main__":
    artifacts = run_pipeline()
    PredictorUI(artifacts).root.mainloop()
# <-- END OF: function to start the pipeline
