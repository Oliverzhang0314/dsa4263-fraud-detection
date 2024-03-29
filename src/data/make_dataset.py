import os
import opendatasets as od
import pandas as pd


def download_and_load(path="./", head_num=None):
    '''Download and load URL dataset.'''
    download(path)
    return load(path, head_num=head_num)

def download(path="./"):
    '''Download URL dataset to selected path.'''
    
    ###
    ### Getting username and key ###
    # Before we start downloading the data set, we need the Kaggle API token. To get that
    # Login into your Kaggle account
    # Get into your account settings page
    # Click on Create a new API token
    # This will prompt you to download the .json file into your system. Save the file, and we will use it in the next step.
    ###

    # Assign the Kaggle data set URL into variable
    dataset = 'https://www.kaggle.com/datasets/michellevp/dataset-phishing-domain-detection-cybersecurity/data'

    # Using opendatasets let's download the data sets as zip file
    return od.download(dataset, path)

def load(path="./", head_num=None):
    '''Load URL dataset from selected path showing first n rows.'''
    
    # Download and load dataset 
    filename = "dataset-phishing-domain-detection-cybersecurity/dataset_cybersecurity_michelle.csv"
    file_path = os.path.join(path, filename)
    if head_num is not None:
        data = pd.read_csv(file_path).head(head_num)
        
    else:
        data = pd.read_csv(file_path)
    
    # Overview of data
    # print(data.info())
    # print(data.describe())
    
    return data

def upload(file, path="./"):
    '''Upload processed URL dataset.'''
    
    # Upload dataset 
    filename="dataset-phishing-domain-detection-cybersecurity/dataset_cybersecurity_michelle.csv"
    file_path = os.path.join(path, filename)
    file.to_csv(file_path)
    
    return file
    
if __name__ == "__main__":
    download_and_load()