import pandas as pd
from sklearn.feature_selection import *

def preprocess(data):
    '''
    The preprocessing steps involved:
        - Remove redundant columns
        - Min max scaling of features
        - 
    '''
    
    # Preprocessing steps
    data = remove_redundant_cols(data)
    data = scale_cols(data)
    
    # Overview of data
    # print(data.info())
    # print(data.describe())
    
    return data

def preprocess_with_feature_selection(data, topk, remove_missing_col):
    '''
    The pre-processing steps with feature selection added:
        - Remove redundant columns (min==max)
        - Remove cols not available in the new dataset
        - Select top K features, by mutual information
        - min max scaling'''
    data = remove_redundant_cols(data)
    if remove_missing_col:
        data = remove_missing_cols(data)
    data = select_top_k_based_on_mutual_information(data, topk=20)
    data = scale_cols(data)

    return data
    
def remove_redundant_cols(data):
    '''Remove redundant column where min value equals to max value.'''
    
    # Check if min value equals to max value for each feature
    remove_cols = []
    for col in data.columns:
        if data[col].min() == data[col].max():
            remove_cols.append(col)
    
    # Remove such columns as no information can be extracted
    data = data.drop(remove_cols, axis=1)
    
    return data


def remove_missing_cols(data):
    ''' Remove columns that are not readily available for other url datasets collected. 
    This is to ensure fairness when testing generalisability of this model.'''

    # These are the cols not present in the new 'malicious-urls-dataset' that we found.
    missing_cols = ['asn_ip', 'domain_in_ip', 'qty_ip_resolved', 'qty_mx_servers', 'qty_nameservers', 'qty_redirects', 'server_client_domain', 'time_domain_activation', 'time_domain_expiration', 'time_response', 'tls_ssl_certificate', 'ttl_hostname']
    # generate a new df that remove these cols
    data = data.drop(columns=missing_cols)

    return data

def select_top_k_based_on_mutual_information(data, topk):
    '''
    select top k feature based on mutual information.
    mutual information: 
    - used for univariate feature selection.
    - able to be used for both categorical and numerical variables.
    - measures the the amount of information that one variable contains about another, regardless of the type of relationship. This includes linear, non-linear, and any other form of dependency.
    '''

    # Selecting the top k features based on mutual information
    X = data.drop('phishing', axis=1)  
    y = data[['phishing']] 
    mi_selector = SelectKBest(mutual_info_classif, k=topk)
    X_kbest = mi_selector.fit_transform(X, y)

    # To get the DataFrame back with selected features
    selected_features = X.columns[mi_selector.get_support(indices=True)]
    X_kbest_df_mi = pd.DataFrame(X_kbest, columns=selected_features)
    
    # combine X and y together
    result_df = pd.concat([X_kbest_df_mi, y], axis=1)
    return result_df


def scale_cols(data):
    '''Apply min-max scaling to all feature columns.'''
    
    # # Select all columns except the last one
    # cols_to_scale = data.columns[:-1]

    # # Apply the operation to selected columns
    # data[cols_to_scale] = (data[cols_to_scale] - data[cols_to_scale].min()) / \
    #                     (data[cols_to_scale].max() - data[cols_to_scale].min())
    
    # data.isnull().sum()
    
    return data

# def replace_missing():
    # Check for missing values
    # data.isnull().sum()
    # any(data<0)
    # data.dropna(inplace=True)
    # data.fillna(data.mean(), inplace=True)
    
# # Create a new DataFrame 'cleaned_df' with -1 replaced by mean
# cleaned_data = data.copy()  # Make a copy of the original DataFrame

# # Replace -1 with the mean of each column
# for col in cleaned_data .columns[:-1]:
#     if -1 in cleaned_data[col].values:
#         mean_val = cleaned_data [col][cleaned_data[col] != -1].mean()
#         cleaned_data[col] = cleaned_data[col].replace(to_replace=-1, value=mean_val)

# # Train test split
# clean_X = cleaned_data.drop("phishing", axis=1)[:100]
# clean_y = cleaned_data["phishing"][:100]
# clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(clean_X, clean_y, test_size=0.2, random_state=20, stratify=cleaned_data["phishing"][:100])

# # Rf Classifier
# clean_clf = RandomForestClassifier().fit(clean_X_train, clean_y_train)

# # Make predictions on the test set
# clean_y_pred = clean_clf.predict(clean_X_test)
 
# # Calculate accuracy
# accuracy = accuracy_score(clean_y_test, clean_y_pred)
# recall = recall_score(clean_y_test, clean_y_pred)
# print("Accuracy:", accuracy)
# print("Recall:", recall)


