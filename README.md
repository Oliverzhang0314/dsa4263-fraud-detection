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

***In our [notebooks](notebooks/), we uses API calls to download data, so you need to follow the guide in the **Getting Ready** section. If failed, you can also downloaded form Kaggle, and save it under [data](data/).***

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


**Hands-on**
The model on Phishing Domain Detection Dataset can be found here: [Phishing Domain](notebooks/phishing_domain.ipynb)

The model on Malicious URLs Dataset can be found here: [Malicious URLs](notebooks/malicious_url.ipynb)

The evaluation of Jaccrd Similarity between feature importance of two models can be found here: [Feature Importance](notebooks/feature_comparison.ipynb)

# Contributor
A team of 5 Data Science and Analytics Students from National University of Singapore: Duan Tianyu, Qiu qishuo, Zhang Xiangyu, Zhao Xi, Zhu Yi Fu

# License
[MIT License](LICENSE)