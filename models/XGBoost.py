import os
import numpy as np
import pandas as pd
import pickle as pkl
import shap
import matplotlib.pyplot as plt

import xgboost as xgb
from utils.evaluators import *
from sklearn.metrics import r2_score

class XGBregressor():
    def __init__(self, configs):
        inputs_path = configs["Dataset"]["data_path"]
        with open(inputs_path, "rb") as f:
            self.inputs = pkl.load(f)["ten_fold_inputs"]
        
        n_estimators = configs["Model"]["n_estimators"]
        k_fold = configs["k_fold"]
        self.models = [xgb.XGBRegressor(n_estimators=n_estimators)] * k_fold
        self.RMSE = []
        self.RMAE = []
        self.R2 = []
        self.RSR = []
        self.MAPE = []
        self.NMBE = []
        self.predictions = []
        self.shape_values = []
        self.feature_names = list(configs["Dataset"]["column_names"])
        self.save_path = configs["Dataset"]["save_path"]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        
    def save(self):
        with open(f"{self.save_path}/xgboost_regression.pkl", "wb") as f:
            pkl.dump(self.models, f)
    
    def load(self):
        if not os.path.exists(f"{self.save_path}/xgboost_regression.pkl"):
            print("Model checkpoint not found!")
            return
        with open(f"{self.save_path}/xgboost_regression.pkl", "rb") as f:
            self.models = pkl.load(f)

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
            sv = self.explain(x_train_df, self.models[idx].predict, xtest_df, f"{self.save_path}/shap_plot_{idx}.png")
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
    
    def eval_on_other_dataset(self, path, sheet_name="Sheet1", ylabel="Strength"):
        # suppose the path is xlsx
        df = pd.read_excel(path, sheet_name)
        xtest = df[self.feature_names].values
        ytest = df[[ylabel]].values
        predictions = []
        rmse_mean = []
        rmae_mean = []
        r2_mean = []
        rsr_mean = []
        mape_mean = []
        nmbe_mean = []

        for model in self.models:
            prediction = model.predict(xtest)
            predictions.append(prediction)
            rmse = calculate_rmse(ytest, prediction)
            rmae = calculate_rmae(ytest, prediction)
            r2 = r2_score(ytest, prediction)
            rsr = calculate_RSR(ytest, prediction)
            mape = calculate_mape(ytest, prediction)
            nmbe = calcuate_nmbe(ytest, prediction)

            rmse_mean.append(rmse)
            rmae_mean.append(rmae)
            r2_mean.append(r2)
            rsr_mean.append(rsr)
            mape_mean.append(mape)
            nmbe_mean.append(nmbe)
        
        return {
            "RMSE": np.mean(rmse_mean), 
            "RMAE": np.mean(rmae_mean), 
            "R2": np.mean(r2_mean), 
            "RSR": np.mean(rsr_mean), 
            "MAPE": np.mean(mape_mean), 
            "NMBE": np.mean(nmbe_mean)
        }