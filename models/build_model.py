import os
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.RidgeRegression import RidgeRegressionModel
from models.RandomForest import RandomForestModel
from models.XGBoost import XGBregressor
from models.Ann import ANNregressor

def build_model(configs):
    types = configs["Type"]
    if types == "ANN":
        return ANNregressor(configs)

    elif types == "RANDOM_FOREST":
        return RandomForestModel(configs)
    
    elif types == "XGBOOST":
        return XGBregressor(configs)
    
    elif types == "RIDGE":
        return RidgeRegressionModel(configs)
    
    else:
        raise NotImplementedError(f"Model type {types} not implemented")

def train_model(model):
    model.fit()
    results = model.get_results()
    save_path = model.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    df = pd.DataFrame()
    for k, v in results.items():
        df[k] = v
    
    df.to_csv(f"{save_path}/results_summary.csv")
    predictions = model.get_predictions()
    df = pd.DataFrame()
    for i in range(len(predictions)):
        df[f"{i}_fold"] = predictions[i]
    df.to_csv(f"{save_path}/results.csv")

    model.save()
    return results

def predict_model_on_other_dataset(configs, model):
    addition_path = configs["Dataset"]["additional_test_data_path"]
    model.load()
    results = model.eval_on_other_dataset(addition_path)
    return results

def analysis_shap_value(configs, model):
    model.load()
    save_path = model.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    k_fold = configs["k_fold"]
    feature_names = model.feature_names
    for i in range(k_fold):
        xtrain, ytrain, xtest, ytest = model.inputs[i]
        xtrain_df = pd.DataFrame(xtrain, columns=feature_names)
        xtest_df = pd.DataFrame(xtest, columns=feature_names)
        predict = model.models[i].predict
        ex = shap.Explainer(predict, xtrain_df.values, feature_names=feature_names)
        shap_values = ex.shap_values(xtest_df.values)
        shap.summary_plot(shap_values, xtest,show=False, feature_names=feature_names)
        fig_save_path = f"{save_path}/shap_plot_{i}.png"
        plt.savefig(fig_save_path)
        plt.close("all")
        
        shap.summary_plot(shap_values, xtest,show=False, feature_names=feature_names, plot_type="bar")
        fig_save_path = f"{save_path}/shap_plot_{i}_bar.png"
        plt.savefig(fig_save_path)
        plt.close("all")

        feature_importance = pd.DataFrame()
        feature_importance["feature"] = feature_names
        feature_importance["importance"] = np.abs(shap_values).mean(0)
        fig_save_path = f"{save_path}/feature_importance_{i}_pie.png"
        
        norm = mpl.colors.Normalize(vmin=feature_importance["importance"].min(), vmax=feature_importance["importance"].max())
        cmap = plt.get_cmap("viridis")
        colors = [cmap(norm(value)) for value in feature_importance["importance"]]

        plt.pie(feature_importance["importance"], labels=feature_importance["feature"], autopct='%1.1f%%', colors=colors)
        plt.title("feature importance of ann regression")
        plt.savefig(fig_save_path)
        plt.close("all")

        feature_name = feature_importance["feature"]
        importance = feature_importance["importance"]

        merge_feature_name = []
        merge_importance = []
        merged_nano = 0
        for n, v in zip(feature_name, importance):
            if 'nano' in n:
                merged_nano += v
            else:
                merge_feature_name.append(n)
                merge_importance.append(v)

        merge_feature_name.append("nano")
        merge_importance.append(merged_nano)
        merged_feature_importance_df = pd.DataFrame()
        merged_feature_importance_df["feature"] = merge_feature_name
        merged_feature_importance_df["importance"] = merge_importance

        # # draw pie chart
        plt.pie(merged_feature_importance_df["importance"], labels=merged_feature_importance_df["feature"], autopct='%1.1f%%', colors=colors)
        plt.title("merged feature importance of ann regression")
        fig_save_path = f"{save_path}/merge_feature_importance_{i}_pie.png"
        plt.savefig(fig_save_path)
        plt.close("all")



