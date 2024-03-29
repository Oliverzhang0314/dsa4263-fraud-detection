import pandas as pd


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

def scale_cols(data):
    '''Apply min-max scaling to all feature columns.'''
    
    # Select all columns except the last one
    cols_to_scale = data.columns[:-1]

    # Apply the operation to selected columns
    data[cols_to_scale] = (data[cols_to_scale] - data[cols_to_scale].min()) / \
                        (data[cols_to_scale].max() - data[cols_to_scale].min())
    
    data.isnull().sum()
    
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


