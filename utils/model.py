from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
import shap

import matplotlib.pyplot as plt
import pandas as pd

def calculate_rmse(actual, predicted):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse

def calculate_rmae(actual, predicted):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the RMAE
    rmae = np.sqrt(np.mean(np.abs(actual - predicted)))
    return rmae

def calculate_r2(actual, predicted):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the total sum of squares (SST) and residual sum of squares (SSR)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    
    # Calculate R2
    r2 = 1 - (ss_residual / ss_total)
    return r2

def calculate_adjusted_r2(actual, predicted, num_predictors=13):
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate the total sum of squares (SST) and residual sum of squares (SSR)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    
    # Calculate R2
    r2 = 1 - (ss_residual / ss_total)
    
    # Calculate Adjusted R2
    n = len(actual)  # Number of observations
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - num_predictors - 1)
    
    return adjusted_r2

def calculate_RSR(actual, prediction):
    rmse = calculate_rmae(actual, prediction)
    scale = np.std(actual)
    return rmse / scale

def calculate_mape(actual, prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

def calcuate_nmbe(actual, prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)
    return np.sum(prediction - actual) / np.sum(actual) * 100



class RidgeRegressionModel():
    def __init__(self, inputs, column_names):
        self.inputs = inputs
        self.models = [linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))] * 10
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []
        self.shape_values = []
        self.feature_names = column_names

    def fit(self):
        for idx, (x_train, y_train, x_test, y_test) in enumerate(self.inputs):
            self.models[idx].fit(x_train, y_train)
            predictions = self.models[idx].predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])

            # explain the model's predictions using SHAP
            x_train_df = pd.DataFrame(x_train, columns=self.feature_names)
            xtest_df = pd.DataFrame(x_test, columns=self.feature_names)
            sv = self.explain(x_train_df, self.models[idx].predict, xtest_df, f"results/ridge_regression/shap_plot_{idx}.png")
            self.shape_values.append(sv)

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }

    def get_predictions(self):
        return self.predictions

    def explain(self, x_summary, x_pred, xtest, save_path):
        ex = shap.Explainer(x_pred, x_summary.values)
        shap_values = ex.shap_values(xtest.values)
        shap.summary_plot(shap_values, xtest,show=False)
        plt.savefig(save_path)
        plt.close("all")
        return shap_values
    
    def get_shape_values(self):
        return self.shape_values

class RandomForestModel():
    def __init__(self, inputs, column_names):
        self.inputs = inputs
        self.models = [RandomForestRegressor(random_state=42)] * 10
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []
        self.shape_values = []
        self.feature_names = column_names

    def fit(self):
        for idx, (x_train, y_train, x_test, y_test) in enumerate(self.inputs):
            self.models[idx].fit(x_train, y_train)
            predictions = self.models[idx].predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])

            # explain the model's predictions using SHAP
            x_train_df = pd.DataFrame(x_train, columns=self.feature_names)
            xtest_df = pd.DataFrame(x_test, columns=self.feature_names)
            sv = self.explain(x_train_df, self.models[idx].predict, xtest_df, f"results/random_forest_regression/shap_plot_{idx}.png")
            self.shape_values.append(sv)

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }

    def get_predictions(self):
        return self.predictions

    def explain(self, x_summary, x_pred, xtest, save_path):
        ex = shap.Explainer(x_pred, x_summary.values)
        shap_values = ex.shap_values(xtest.values)
        shap.summary_plot(shap_values, xtest,show=False)
        plt.savefig(save_path)
        plt.close("all")

        return shap_values

class SVMRegressor():
    def __init__(self, inputs, column_names):
        self.inputs = inputs
        self.models = [SVR()] * 10
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []
        self.shape_values = []
        self.feature_names = column_names

    def fit(self):
        for idx, (x_train, y_train, x_test, y_test) in enumerate(self.inputs):
            self.models[idx].fit(x_train, y_train)
            predictions = self.models[idx].predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])
            # explain the model's predictions using SHAP
            x_train_df = pd.DataFrame(x_train, columns=self.feature_names)
            xtest_df = pd.DataFrame(x_test, columns=self.feature_names)
            sv = self.explain(x_train_df, self.models[idx].predict, xtest_df, f"results/svm_regression/shap_plot_{idx}.png")
            self.shape_values.append(sv)


    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }
    
    def get_predictions(self):
        return self.predictions

    def explain(self, x_summary, x_pred, xtest, save_path):
        ex = shap.Explainer(x_pred, x_summary.values)
        shap_values = ex.shap_values(xtest.values)
        shap.summary_plot(shap_values, xtest,show=False)
        plt.savefig(save_path)
        plt.close("all")

        return shap_values

class XGBregressor():
    def __init__(self, inputs, column_names):
        self.inputs = inputs
        self.models = [xgb.XGBRegressor(n_estimators=1000)] * 10
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []
        self.shape_values = []
        self.feature_names = column_names

    def fit(self):
        for idx, (x_train, y_train, x_test, y_test) in enumerate(self.inputs):
            self.models[idx].fit(x_train, y_train)
            predictions = self.models[idx].predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])
            # explain the model's predictions using SHAP
            x_train_df = pd.DataFrame(x_train, columns=self.feature_names)
            xtest_df = pd.DataFrame(x_test, columns=self.feature_names)
            sv = self.explain(x_train_df, self.models[idx].predict, xtest_df, f"results/xgboost_regression/shap_plot_{idx}.png")
            self.shape_values.append(sv)

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }
    
    def get_predictions(self):
        return self.predictions
    
    def explain(self, x_summary, x_pred, xtest, save_path):
        ex = shap.Explainer(x_pred, x_summary.values)
        shap_values = ex.shap_values(xtest.values)
        shap.summary_plot(shap_values, xtest,show=False)
        plt.savefig(save_path)
        plt.close("all")

        return shap_values

class ANNregressor():
    def __init__(self, inputs, column_names):
        self.inputs = inputs
        self.models = [MLPRegressor(hidden_layer_sizes=(32, 128, 128), max_iter=1000, random_state=42, tol=1e-6, batch_size=64)] * 10
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []
        self.shape_values = []
        self.feature_names = column_names

    def fit(self):
        for idx, (x_train, y_train, x_test, y_test) in enumerate(self.inputs):
            self.models[idx].fit(x_train, y_train)
            predictions = self.models[idx].predict(x_test)
            self.RMSE.append(calculate_rmse(y_test, predictions))
            self.RMAE.append(calculate_rmae(y_test, predictions))
            self.R2.append(r2_score(y_test, predictions))
            self.RSR.append(calculate_RSR(y_test, predictions))
            self.MAPE.append(calculate_mape(y_test, predictions))
            self.NMBE.append(calcuate_nmbe(y_test, predictions))
            self.predictions.append([y_test, predictions])
            # explain the model's predictions using SHAP
            x_train_df = pd.DataFrame(x_train, columns=self.feature_names)
            xtest_df = pd.DataFrame(x_test, columns=self.feature_names)
            sv = self.explain(x_train_df, self.models[idx].predict, xtest_df, f"results/ann_regression/shap_plot_{idx}.png")
            self.shape_values.append(sv)

    def get_results(self):
        return {
            "RMSE": self.RMSE,
            "RMAE": self.RMAE,
            "R2": self.R2,
            "RSR": self.RSR,
            "MAPE": self.MAPE,
            "NMBE": self.NMBE,
        }
    
    def get_predictions(self):
        return self.predictions
    
    def explain(self, x_summary, x_pred, xtest, save_path):
        ex = shap.Explainer(x_pred, x_summary.values)
        shap_values = ex.shap_values(xtest.values)
        shap.summary_plot(shap_values, xtest,show=False)
        plt.savefig(save_path)
        plt.close("all")

        return shap_values