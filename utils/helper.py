import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_theme(style="ticks")

import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split

# read excel
def read_excel(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)
    return df

def draw_hex_contour_plot(df, feature, label, color, ax=None):
    sns.jointplot(x=df[feature], y=df[label], kind='hex', color=color)
    # savefig
    os.makedirs("results", exist_ok=True)
    feature = feature.replace("/", "-")
    save_path = f"results/hex_contour_{feature}_{label}.png"
    # remove the space from string
    plt.savefig(save_path)
    plt.close("all")
    return save_path

def show_joint_plots(save_path, ax):
    # read the png from save_path
    img = plt.imread(save_path)
    ax.imshow(img)

def draw_heatmap(df, drop_column=["Data"]):
    new_df = df.fillna(0)
    new_df.drop(columns=drop_column, inplace=True)
    corr_matrix = new_df.corr()
    sns.heatmap(corr_matrix, annot=True)

def create_cross_validation(df, n_split=10):
    new_df = df.fillna(0)
    new_df.drop(columns=["Data"], inplace=True)
    values =  df.to_numpy()
    x = values[:, :-1]
    y = values[:, -1]
    
    inputs = []
    for i in range(n_split):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42 + i)
        inputs.append((x_train, y_train, x_test, y_test))
    return inputs