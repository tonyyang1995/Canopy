# Predicting Compressive Strength of Concrete with SCMs and Nanoparticles using Machine Learning Approaches
Nanomaterials and supplementary cementitious materials (SCMs) are typically used together in efforts to enhance the performance of concrete and mitigating the environmental impact of concrete construction. Compressive strength, being the most well-measured mechanical property of concrete, required an excessive amount of resources and time. To address such issue, this study proposes using several machine learning algorithms to predict the compressive strength of concrete containing nanoparticles (i.e., nano-TiO2 and nano-SiO2) and SCMs (i.e., fly ash, slag, and silica fume). A database composed of twelve input variables is utilized to train four different machine leaning models, including Ridge Regression (Ridge), Artificial Neural Network (ANN), Random Forest (RF), and Extreme Gradient Boost (XGB). Furthermore, the performance of the trained machine leaning models are evaluated by five criteria, including Root Mean Square Error (RMSE), Mean Absolute Error (MAE), coefficient of correlation (R2), Normalized Mean Bias Error (NMBE), and Mean Absolute Percentage Error (MAPE). Among the four models studied, XGB (R2=0.93) model presents the highest performance in the prediction of compressive strength of concrete. Shapley Additive exPlanations (SHAP) method is adapted to perform the feature importance analysis using XGB model, providing a more comprehensive explanatory information on identifying the most influential inputs and quantifying the impact of input variables on the compressive strength of concrete. In the end, this study provides an open-source tool Canopy for concrete researchers in mix design optimization and data analysis.
<img width="480" alt="image" src="https://github.com/user-attachments/assets/808c61c9-85bf-41b7-ad15-b945c4f01a8c" />



## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```

## Tutorial for Canopy
1. Data processing: [here](https://github.com/tonyyang1995/Canopy/blob/main/Preprocess_data.ipynb)
2. Train Canopy: [here](https://github.com/tonyyang1995/Canopy/blob/main/train.ipynb)
3. Explain Canopy: [here](https://github.com/tonyyang1995/Canopy/blob/main/Explain.ipynb)

## Data structure
```
├── requirement.txt
├── dataset
│      ├── your_own_data.xlsx
│      ├── Concrete_Data_test.xlsx

```

After Data processing, you should have the following pickle file. The pickle file will split the dataset to 10 cross-validation subsets.
```
├── requirement.txt
├── dataset
│      ├── your_own_data.xlsx
│      ├── Concrete_Data_test.xlsx
│      ├── input_train_test.pkl
```
## Results directory
```
├── results
│      ├── xgboost_regression
│          ├── lmplot.png
│          ├── shap_plot.png
│          ├── wo_nano_scatter.png
│          ├── result.csv
│          ├── result_summary.csv
│          ├── xgboost_regression.pkl
│          ├── 。。。
|      ...
│      ├── ann_regression
```
