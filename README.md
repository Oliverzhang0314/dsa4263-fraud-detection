# Welcome to DSA4263 fraud detection project

Here is the link to the dataset for btcusdt (2018-2022 i.e 5 years) and (2020-2023 i.e 4 years) :
- [Phishing Domain Detection Dataset](https://www.kaggle.com/datasets/michellevp/dataset-phishing-domain-detection-cybersecurity)
- [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)

# Overview
In the digital age, cybersecurity threats pose a significant challenge to individuals and organizations alike. Phishing attacks, in particular, have emerged as a prevalent method used by malicious actors to deceive users and gain unauthorized access to sensitive information. In response to the growing threat, we present our efforts to develop a machine learning model specifically tailored for identifying phishing URLs.

# Getting Started
Follow these steps to set up your running environment.

## Download Data
There are two ways to download data, through **[Kaggle Download](https://www.kaggle.com/)** or through **[API calls - How To Efficiently Download Any Dataset from Kaggle](https://ravi-chan.medium.com/how-to-download-any-data-set-from-kaggle-7e2adc152d7f)**

In our [notebooks](notebooks/), we uses API calls to download data, so you do need to follow the guide in the **Getting Ready** section. If failed, you can also downloaded form Kaggle, and save it under [data](data/).

## Installation
The python version requirement is **Python >= 3.12**. 

Step 1: Clone this repository to your local machine

```bash
git clone <repository-url>
```

Step 2: Navigate to the project directory

```bash
cd <project-directory>
```

Step 3: Install the requirements dependencies
```bash
pip install -e .
```
OR
```bash
pip install -r requirements.txt
```

If your development introduces new Python packages as dependencies, ensure to list them in the `requirements.txt` file. Format each dependency on a new line, specifying the exact version to maintain consistency across environments

## How to use
Functions
