import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_theme(style="ticks")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


from sklearn.model_selection import train_test_split, KFold

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

def draw_heatmap(df, drop_column=["Data"], cmap='coolwarm'):
    new_df = df.fillna(0)
    new_df.drop(columns=drop_column, inplace=True)
    corr_matrix = new_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)

def create_cross_validation(df, n_split=10, use_random=True, split_rate=0.1):
    if use_random:
        new_df = df.fillna(0)
        new_df.drop(columns=["Data"], inplace=True)
        values =  new_df.to_numpy()
        x = values[:, :-1]
        y = values[:, -1]
        
        inputs = []
        for i in range(n_split):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_rate, random_state=42 + i)
            inputs.append((x_train, y_train, x_test, y_test))
        return inputs
    else:
        # TODO need to be tested
        # use k-fold from sklearn
        inputs = []
        kf = KFold(n_splits=n_split)
        for train_index, test_index in kf.split(df):
            train = df.iloc[train_index]
            x_train = train.iloc[:, :-1]
            y_train = train.iloc[:, -1]
            test = df.iloc[test_index]
            x_test = test.iloc[:, :-1]
            y_test = test.iloc[:, -1]
            inputs.append((x_train, y_train, x_test, y_test))
        return inputs
    
def read_csv(file_path):
    return pd.read_csv(file_path)

def summary2df(summary, name='ridge'):
    values = summary.to_numpy()
    rmse = list(values[:, 1])
    rmae = list(values[:, 2])
    r2 = list(values[:, 3])
    rsr = list(values[:, 4])
    mape = list(values[:, 5])
    nmbe = list(abs(values[:, 6]))
    method = [name] * len(rmse)
    return method, rmse, rmae, r2, rsr, mape, nmbe

def draw_box_plot(result_lists, name_lists, metric_lists, box_plot_colors=sns.color_palette("Set2")):
    df = pd.DataFrame()
    methods = []
    rmses = []
    rmaes = [] 
    r2s = []
    rsrs = []
    mapes = []
    nmbes = []

    for result, name in zip(result_lists, name_lists):
        method, rmse, rmae, r2, rsr, mape, nmbe = summary2df(result, name=name)
        methods.extend(method)
        rmses.extend(rmse)
        rmaes.extend(rmae)
        r2s.extend(r2)
        rsrs.extend(rsr)
        mapes.extend(mape)
        nmbes.extend(nmbe)

    df["method"] = methods
    df["RMSE"] = rmses
    df["MAE"] = rmaes
    df["R2"] = r2s
    df["RSR"] = rsrs
    df["MAPE"] = mapes
    df["NMBE"] = nmbes

    for metric in metric_lists:
        plt.close("all")
        if metric not in df.columns:
            print(f"{metric} is not in the dataframe")
            continue
        sns.boxplot(x=df["method"], y=df[metric], palette=box_plot_colors[:len(result_lists)], hue=df["method"])
        plt.savefig(f"results/box_plot_{metric}.png")

    return df

def convert_str2_list(list_str):
    list_str = list_str[1:-1]
    list_str = ' '.join(list_str.split())
    list_str = list_str.split(" ")
    list_str = [float(a) for a in list_str]
    return list_str

def draw_regplot(save_path, title, folds=10, facecolor="#FF8C94", patchcolor='red', alpha=0.3, xlim=(0,120), ylim=(0,120)):
    results = read_csv(f"{save_path}/results.csv")
    for idx in range(0, folds):
        plt.close("all")
        res = results[f'{idx}_fold']
        gts, pred = res[0], res[1]
        gts = convert_str2_list(gts)
        preds = convert_str2_list(pred)
        
        gts = np.array(gts)
        preds = np.array(preds)
        df_values = np.stack([gts, preds], axis=1)
        df = pd.DataFrame(df_values, columns=["GroundTruth", "Prediction"])
        plot = sns.lmplot(data=df, x="GroundTruth", y="Prediction", legend=True, markers=['x'], scatter_kws={"s":3})
        
        ax = plot.axes[0, 0]  # Get the axis from lmplot
        sns.regplot(x="GroundTruth", y="Prediction", data=df, scatter=False, ci=95, line_kws={'color':'black'}, 
                ax=ax, 
                scatter_kws={'s': 3}, 
                fit_reg=True)
        ax.collections[1].set_facecolor(facecolor)
        ax.collections[1].set_alpha(alpha)  # Adjust transparency if needed
        
        red_patch = mpatches.Patch(color=patchcolor, alpha=alpha, label="95% Confidence Interval")
        plt.legend(handles=[red_patch], loc='upper left', frameon=True)

        plt.tight_layout()
        if (xlim is not None) and (ylim is not None):
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.title(title)
        
        plt.savefig(f'{save_path}/lmplot_{idx}_fold.png')

def draw_regplots(save_paths, titles, folds=10, facecolor="#FF8C94", patchcolor='red', alpha=0.3, xlim=(0,120), ylim=(0,120)):
    for save_path, title in zip(save_paths, titles):
        draw_regplot(save_path, title, folds=folds, facecolor=facecolor, patchcolor=patchcolor, alpha=alpha, xlim=xlim, ylim=ylim)

def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
