# Welcome to DSA4263 fraud detection project!

Here is the link to the phishing domain and malicious URLs dataset:
- [Phishing Domain Detection Dataset](https://www.kaggle.com/datasets/michellevp/dataset-phishing-domain-detection-cybersecurity)
- [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)

# Overview
In the digital age, cybersecurity threats pose a significant challenge to individuals and organizations alike. Phishing attacks, in particular, have emerged as a prevalent method used by malicious actors to deceive users and gain unauthorized access to sensitive information. In response to the growing threat, we present our efforts to **develop a machine learning model specifically tailored for identifying phishing URLs**.

# Getting Started
Follow these steps to set up your running environment.

### Download Data
There are two ways to download data from:
- [Kaggle Website](https://www.kaggle.com/)
- [API calls - How To Efficiently Download Any Dataset from Kaggle](https://ravi-chan.medium.com/how-to-download-any-data-set-from-kaggle-7e2adc152d7f)

***In our [notebooks](notebooks/), we uses API calls to download data, so you need to follow the guide in the **<Getting Ready>** section. If failed, you can also downloaded form Kaggle, and save it under [data](data/).***

### Understand Data
Refer to [Data Dictionary](references/datadictionary.txt) for the definition of each column in the phishing dataset.

### Installation
The python version requirement is **Python >= 3.12**.

- **Step 1: Clone this repository to your local machine**
```bash
git clone https://github.com/Oliverzhang0314/dsa4263-fraud-detection.git
```

- **Step 2: Navigate to the project directory**
```bash
cd dsa4263-fraud-detection
```

- **Step 3: Install the requirements dependencies**
```bash
pip install -e .
```
OR
```bash
pip install -r requirements.txt
```

***If your development introduces new packages as dependencies, list them in the `requirements.txt` file. Format each dependency on a new line, specifying the exact version to maintain consistency across environments***

# How to use
**File Structure**
```bash
dsa4263-fraud-detection
    ├───data
    │   ├───raw
    │   │   ├───dataset-phishing-domain-detection-cybersecurity
    │   │   └───malicious-urls-dataset
    │   └───processed
    │       ├───dataset-phishing-domain-detection-cybersecurity
    │       └───malicious-urls-dataset
    ├───fraud
    │   ├───data
    │   │   └───make_dataset.py
    │   ├───features
    │   │   └───pre_processing.py
    │   │   └───build_features.py
    │   ├───models
    │   │   └───lazy_predict.py
    │   │   └───train_model.py
    │   │   └───predict_model.py
    │   └───visualization
    │       └───visualize.py
    ├───notebooks
    │   └───EDA.py
    │   └───phishing_domain.py
    │   └───malicious_url.py
    │   └───feature_comparison.py
    ├───models
    ├───plots
    ├───features
    ├───references
    ├───setup.py
    ├───requirements.txt
    ├───README.md
    └───LICENSE

```

**Functions**

- ***Download and Load Data*** \
download_and_load(path="./", head_num=None) - Download and load 1st URL dataset \
upload(file, path="./") - Upload processed 1st URL dataset \
download_and_load_new(path="./", head_num=None) - Download and load 2nd URL dataset \
upload_new(file, path="./") - Upload processed 2nd URL dataset

- ***Preprocessing*** \
preprocess(data) - The pre-processing steps involved: Remove redundant columns, Remove cols not available in the new dataset, Min max scaling of features \
preprocess_with_feature_selection(data, topk=20, remove_missing_col=False) - The pre-processing steps add an extra step of Select top K features by mutual information

- ***Feature Engineering*** \
process_new_url(df, path="../data/processed") - Parse 'url' column of df into domain, directory, file and params \
reformat_df(df) - Modify df inplace, adding in extracted features such as url_shortened, qty_params, url_google_index, domain_google_index, email_in_url, qty_tld_url, tld_present_params

- ***Model Fitting*** \
lazy_predict(data, exclude=None) - Initial prediction using multiple choices of classifiers \
train(model, X_train, y_train, param_grid=None, iteration=50, path="./") - Train and tune hyperparameters for best performed model \
predict(model, X_test, y_test=None) - Predict using fine-tuned model

- ***Evaluation*** \
distribution_plot(data, col_name) - Plot distribution of selected categorical column \
feat_importance(model, feat_cols, path="./") - Rank feature importance among all features \
confusion_plot(y_test, y_pred) - Plot confusion matrix \
precision_recall(y_test, y_pred) - Plot precision recall \
roc_curve(y_test, y_pred) - Plot ROC curve \
calibration_disp(model, X_test, y_test) - Plot calibration display \
decision_boundary(model, best_params, X_train, y_train) - Plot decision boundary \
learning_curv(model, X_train, y_train) - Plot learning curve \
plot_accuracy_vs_k(data, y_column, k_values, seed=4263) - Plot test accuracy against k, where k is the number of top predictors selected by mutual information \
shap_plot(data, y_column, k, seed=4263) - Plot SHAP value for first k rows \
lime_plot(X_train, X_test, y_test, k, model) - Plot LIME value for k random rows \
ice_curve(col_of_interest, X_train, model) - Plot ICE curve for column of interest

**Hands-on**

The model on Phishing Domain Detection Dataset can be found here: [Phishing Domain](notebooks/phishing_domain.ipynb)

The model on Malicious URLs Dataset can be found here: [Malicious URLs](notebooks/malicious_url.ipynb)

The evaluation of Jaccrd Similarity between feature importance of two models can be found here: [Feature Importance](notebooks/feature_comparison.ipynb)

# Contributor
A team of 5 Data Science and Analytics Students from National University of Singapore: Duan Tianyu, Qiu qishuo, Zhang Xiangyu, Zhao Xi, Zhu Yi Fu

# License
[MIT License](LICENSE)