# Predicting Compressive Strength of Concrete with SCMs and Nanoparticles using Machine Learning Approaches
Nanomaterials and supplementary cementitious materials (SCMs) are typically used together in efforts to enhance the performance of concrete and mitigate the environmental impact of concrete construction. However, the complex interactions between nanomaterials, SCMs, and cement make concrete mix design a challenging, iterative, and labor-intensive process, often relying on trial-and-error experimentation. Machine learning (ML) offers an opportunity to better understand the influence of input parameters and to accelerate the optimization of mix designs through data-driven insights. This study proposes an open-source and easy-to-access framework, Canopy, to support the concrete research community in optimizing mix design. Using a dataset collected from the literature, we conducted detailed analyses and identified Extreme Gradient Boosting (XGB) as the most effective ML algorithm for predicting compressive strength. Furthermore, the framework incorporates post-analysis tools, such as Shapley Additive exPlanations (SHAP), to provide interpretable insights into the importance of various input parameters. Our findings highlight the critical role of nanomaterials, contributing 8% to the overall improvement in compressive strength, underscoring their significance in concrete performance modification. By combining predictive modeling with interpretability, this framework aims to streamline the design process and reduce experimental workload. Beyond its technical contributions, this study emphasizes the broader impact of integrating machine learning into concrete research, paving the way for more sustainable, efficient, and data-driven approaches in the development of advanced construction materials.
<img width="465" alt="image" src="https://github.com/user-attachments/assets/f13c271f-bfed-4ae3-bb71-fdbf97f14af8" />



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
