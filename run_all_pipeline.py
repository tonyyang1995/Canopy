import importlib
import utils.helper as helper
from utils.helper import read_excel
from utils.helper import draw_hex_contour_plot, show_joint_plots, draw_heatmap
import matplotlib.pyplot as plt
from utils.helper import create_cross_validation, read_csv, draw_box_plot, draw_regplots
import os
import numpy as np
import pandas as pd
import seaborn as sns

# ridge_regression results
import utils.model
from utils.model import RidgeRegressionModel
from utils.model import RandomForestModel
from utils.model import SVMRegressor
from utils.model import XGBregressor
from utils.model import ANNregressor



if __name__ == '__main__':
    # 1. define_args
    print("step1: define args and load dataset")
    excel_file_path = "./Data_Collection_V3.xlsx" # dataset name
    sheet_page = "Sheet1" # sheet page name
    label = "Strength" # the label for the pred column
    ridge_save_path = "results/ridge_regression"
    rf_save_path = "results/rf_regression"
    svm_save_path = "results/svm_regression"
    xgb_save_path = "results/xgb_regression"
    ann_save_path = "results/ann_regression"

    box_plot_colors = sns.color_palette("Set2")
    kfold = 10
    split_rate=0.1
    facecolor = "#FF8C94"
    patchcolor = 'red'
    alpha = 0.3
    # xlim = (0, 120)
    # ylim = (0, 120)
    xlim = None
    ylim = None

    df = read_excel(excel_file_path, sheet_page)
    column_names = list(df.columns)

    save_paths = []
    for feature in column_names[1:13]:
        sp = draw_hex_contour_plot(df, feature, label, "#4CB391")
        save_paths.append(sp)

    # 2. draw joint plots
    print("step2: analysis dataset and draw plots")
    row = 4
    col= 3
    count = 0
    fig, axs = plt.subplots(row,col, figsize=(20,20))
    for i in range(row):
        for j in range(col):
            show_joint_plots(save_paths[count], axs[i][j])
            count += 1
    plt.tight_layout()
    # plt.show()

    # 3. draw heatmap
    print("step3: drawing heatmap")
    plt.close("all")
    plt.figure(figsize=(15, 15))
    # cmap options: ["YlGnBu", "Blues", "coolwarm", "BuPu", "Greens", "Oranges", "Reds", "Purples", "YlOrBr"]
    draw_heatmap(df, drop_column=["Data"], cmap='Greens')
    plt.title("Heatmap")
    plt.savefig('results/heatmap.png')
    # plt.show()

    # 4. create cross validation
    print("step4: create cross validation datasets")
    ten_fold_inputs = create_cross_validation(df, n_split=kfold, use_random=True, split_rate=split_rate)
    # print(len(ten_fold_inputs))
    
    # 5. Regression
    # 5.1 Ridge Regression
    print("step5.1: Ridge Regression")
    plt.close("all")
    ridgeregression = RidgeRegressionModel(ten_fold_inputs, column_names[1:13])
    ridgeregression.fit()
    ridge_regression_results = ridgeregression.get_results()
    save_path = ridge_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame()
    for k, v in ridge_regression_results.items():
        df[k] = v
    df.to_csv(f"{save_path}/results_summary.csv")
    prediction = ridgeregression.get_predictions()
    df = pd.DataFrame()
    for i in range(len(prediction)):
        df[f"{i}_fold"] = prediction[i]
    df.to_csv(f"{save_path}/results.csv")

    # 5.2 RF regression results
    print("step5.2: RF Regression")
    plt.close("all")
    RFregression = RandomForestModel(ten_fold_inputs, column_names[1:13])
    RFregression.fit()
    rf_regression_results = RFregression.get_results()
    save_path = rf_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame()
    for k, v in rf_regression_results.items():
        df[k] = v
    df.to_csv(f"{save_path}/results_summary.csv")
    prediction = RFregression.get_predictions()
    df = pd.DataFrame()
    for i in range(len(prediction)):
        df[f"{i}_fold"] = prediction[i]
    df.to_csv(f"{save_path}/results.csv")

    # 5.3 svm regression results
    print("step5.3: SVM Regression")
    plt.close("all")
    SVMregression = SVMRegressor(ten_fold_inputs, column_names[1:13])
    SVMregression.fit()
    SVM_regression_results = SVMregression.get_results()
    save_path = svm_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame()
    for k, v in SVM_regression_results.items():
        df[k] = v
    df.to_csv(f"{save_path}/results_summary.csv")
    prediction = SVMregression.get_predictions()
    df = pd.DataFrame()
    for i in range(len(prediction)):
        df[f"{i}_fold"] = prediction[i]
    df.to_csv(f"{save_path}/results.csv")

    # 5.4 xgb regression results
    print("step5.4: XGB Regression")
    plt.close("all")
    XGBregression = XGBregressor(ten_fold_inputs, column_names[1:13])
    XGBregression.fit()
    XGB_regression_results = XGBregression.get_results()
    save_path = xgb_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame()
    for k, v in XGB_regression_results.items():
        df[k] = v
    df.to_csv(f"{save_path}/results_summary.csv")
    prediction = XGBregression.get_predictions()
    df = pd.DataFrame()
    for i in range(len(prediction)):
        df[f"{i}_fold"] = prediction[i]
    df.to_csv(f"{save_path}/results.csv")

    # 5.5 ANN regression results
    print("step5.5: ANN Regression")
    plt.close("all")
    ANNregression = ANNregressor(ten_fold_inputs, column_names[1:13])
    ANNregression.fit()
    ANN_regression_results = ANNregression.get_results()
    save_path = ann_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame()
    for k, v in ANN_regression_results.items():
        df[k] = v
    df.to_csv(f"{save_path}/results_summary.csv")
    prediction = ANNregression.get_predictions()
    df = pd.DataFrame()
    for i in range(len(prediction)):
        df[f"{i}_fold"] = prediction[i]
    df.to_csv(f"{save_path}/results.csv")

    # 6. draw box plot
    print("drawing box plot")
    plt.close("all")
    ridge_result_summary = f"{ridge_save_path}/results_summary.csv"
    rf_result_summary = f"{rf_save_path}/results_summary.csv"
    svm_result_summary = f"{svm_save_path}/results_summary.csv"
    xgb_result_summary = f"{xgb_save_path}/results_summary.csv"
    ann_result_summary = f"{ann_save_path}/results_summary.csv"

    ridge_result_summary = read_csv(ridge_result_summary)
    rf_result_summary = read_csv(rf_result_summary)
    svm_result_summary = read_csv(svm_result_summary)
    xgb_result_summary = read_csv(xgb_result_summary)
    ann_result_summary = read_csv(ann_result_summary)

    # make sure the order match the name lists!!!
    result_lists = [ridge_result_summary, ann_result_summary, rf_result_summary, xgb_result_summary]
    name_lists = ["RIDGE","ANN", "RF", "XGB"]
    metric_lists = ["RMSE", "RMAE", "R2", "RSR", "MAPE", "NMBE"]
    plt.close("all")
    df = draw_box_plot(result_lists, name_lists, metric_lists, box_plot_colors=box_plot_colors)

    # 7. relative plots
    print("drawing relative plots")
    plt.close("all")
    ridge_result = f"{ridge_save_path}"
    rf_result = f"{rf_save_path}"
    svm_result = f"{svm_save_path}"
    xgb_result = f"{xgb_save_path}"
    ann_result = f"{ann_save_path}"
    result_lists = [ridge_result, rf_result, xgb_result, ann_result]
    draw_regplots(result_lists, name_lists, folds=kfold, facecolor=facecolor, patchcolor=patchcolor, alpha=alpha, xlim=xlim, ylim=ylim)
