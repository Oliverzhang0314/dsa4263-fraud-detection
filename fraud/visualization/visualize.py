import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from pycebox.ice import ice, ice_plot
import shap
from sklearn.model_selection import train_test_split
from lime import lime_tabular, lime_text

def pre_plot(data):
    '''Plot visualization to explore data before training.'''
    
    # Distribution of target
    distribution_plot(data, "phishing")

def distribution_plot(data, col_name):
    '''Plot distribution of selected categorical column.'''
    data[col_name].value_counts().sort_index().plot.bar(x="Target Value", y="Number of Occurrences", title="Distribution of Fraud Label")
    plt.savefig("../plots/distribution_plot.png")
    plt.show()

def feat_importance(model, feat_cols):
    '''Rank feature importance among all features.'''
    # feature_importance = clf.feature_importances_
    feature_importances = np.mean([
        tree.feature_importances_ for tree in model.best_estimator_
    ], axis=0)
    sorted_idx = np.argsort(feature_importances)[-20:]
    fig = plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), feat_cols[sorted_idx])
    plt.title("Feature Importance of Classification Model")
    plt.savefig("../plots/feat_importance.png")
    plt.show()
    
def confusion_plot(y_test, y_pred):
    '''Plot confusion matrix.'''
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Confusion Matrix of Classification Model")
    plt.savefig("../plots/confusion_plot.png")
    plt.show()

def precision_recall(y_test, y_pred):
    '''Plot precision recall.'''
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.title("Precision-Recall of Classification Model")
    plt.savefig("../plots/precision_recall.png")
    plt.show()

def roc_curve(y_test, y_pred):
    '''Plot ROC curve.'''
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="RandomForest")
    display.plot(color='black', linestyle='-')
    # plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.plot([0, 0], [0, 1], color='red', linestyle='-.')
    plt.plot([0, 1], [1, 1], color='red', linestyle='-.')
    plt.title("ROC Curve of Classification Model")
    plt.savefig("../plots/roc_curve.png")
    plt.show()
    
def calibration_disp(model, X_test, y_test):
    '''Plot calibration display.'''
    CalibrationDisplay.from_estimator(model, X_test, y_test)
    plt.title("Calibration Display of Classification Model")
    plt.savefig("../plots/calibration_disp.png")
    plt.show()

def decision_boundary(model, best_params, X_train, y_train):
    '''Plot decision boundary.'''
    # Reconstruct best model
    n_estimators = best_params["n_estimators"]
    min_samples_split = best_params["min_samples_split"]
    max_features = best_params["max_features"]
    max_depth = best_params["max_depth"]
    tree = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_features=max_features, max_depth=max_depth)
    
    feature_importances = np.mean([
        tree.feature_importances_ for tree in model.best_estimator_
    ], axis=0)
    sorted_idx = np.argsort(feature_importances)[-20:]
    
    tree = tree.fit(X_train[X_train.columns[sorted_idx[-2:]]], y_train)
    display = DecisionBoundaryDisplay.from_estimator(
        tree, X_train[X_train.columns[sorted_idx[-2:]]], response_method="predict",
        xlabel=X_train.columns[sorted_idx[-1]], ylabel=X_train.columns[sorted_idx[-2]],
        alpha=0.5,
    )
    clean_df = pd.concat([X_train, y_train], axis=1)
    samples = clean_df.sample(frac=0.001, replace=False, random_state=1)
    display.ax_.scatter(X_train[X_train.columns[sorted_idx[-1]]], X_train[X_train.columns[sorted_idx[-2]]], c=y_train, edgecolor="k")
    plt.title("Decision Boundary of Classification Model")
    plt.savefig("../plots/decision_boundary.png")
    plt.show()

def learning_curv(model, X_train, y_train):
    '''Plot learning curve.'''
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train)
    display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores, score_name="Score")
    display.plot()
    plt.title("Learning Curve of Classification Model")
    plt.savefig("../plots/learning_curv.png")
    plt.show()
    
# # Histogram
# plt.hist(data["numerical_column"], bins=20, color="skyblue", edgecolor="black")
# plt.title("Histogram of Numerical Column")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

# # Box plot
# plt.boxplot(data["numerical_column"])
# plt.title("Box plot of Numerical Column")
# plt.ylabel("Value")
# plt.show()

# # Scatter plot
# plt.scatter(data["numerical_column1"], data["numerical_column2"], color="green")
# plt.title("Scatter plot of Numerical Column1 vs Numerical Column2")
# plt.xlabel("Numerical Column1")
# plt.ylabel("Numerical Column2")
# plt.show()

# # Line plot
# plt.plot(data["index_column"], data["numerical_column"], marker="o", color="blue", linestyle="-")
# plt.title("Line plot of Numerical Column over Index Column")
# plt.xlabel("Index")
# plt.ylabel("Numerical Column")
# plt.show()

# # Bar plot
# plt.bar(data["category_column"], data["numerical_column"], color="orange")
# plt.title("Bar plot of Numerical Column across Categories")
# plt.xlabel("Category")
# plt.ylabel("Numerical Column")
# plt.xticks(rotation=45)
# plt.show()
def lime_plot(clean_X_train,clean_X_test,clean_y_test,clean_clf):
    #LIME tabular explainer
    explainer = lime_tabular.LimeTabularExplainer(clean_X_train.values,
                                              feature_names=clean_X_train.columns,
                                              class_names=['Not Phishing', 'Phishing'],
                                              discretize_continuous=True)
    
#  Generate 3 random indices
    idx = np.random.randint(0, clean_X_test.shape[0], size=3)

# Extract instances and true classes using the indices
    instances = clean_X_test.iloc[idx].values
    true_classes = clean_y_test.iloc[idx]

# Loop through the instances and explain each prediction
    for i, instance in enumerate(instances):
        true_class = true_classes.iloc[i]
    
    # Explain the prediction for this instance
    explanation = explainer.explain_instance(instance,
                                             clean_clf.predict_proba,
                                             num_features=6,
                                             top_labels=1)
    
    print(f'Instance {i+1}:')
    print('True Class:', 'Phishing' if true_class == 1 else 'Not Phishing')
    print('Predicted Class:', 'Phishing' if clean_clf.predict([instance])[0] == 1 else 'Not Phishing')
    print('Explanation for Predicted Class:')
    explanation.show_in_notebook()

def predict_fn(X,clean_clf):
        # Predict probabilities for each row
        proba = clean_clf.predict_proba(X)
        # Return the probabilities of the positive class (assuming binary classification)
        return proba[:, 1]
def iceplot(col_of_interest,xlim,clean_X_train,clean_clf):
  
    # Create ICE Data for the col_of_interest
    ice_data = ice(clean_X_train.iloc[1:100,], col_of_interest, lambda x: predict_fn(x, clean_clf))
    # Plot ICE Curves
    fig, ax = plt.subplots()
    ice_plot(ice_data,linewidth=1,ax=ax)
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Probability of phishing")
    ax.set_title("ICE Curves for Random Forest Model")
    ax.set_xlim(0, xlim)  # Limit x-axis from 0 to 0.1
    plt.tight_layout()
    plt.show()
def shap_plot(cleaned_data,size):
    clean_X = cleaned_data.drop("phishing", axis=1)[1:size]
    clean_y = cleaned_data["phishing"][1:size]
    clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(clean_X, clean_y, test_size=0.2, random_state=20, stratify=cleaned_data["phishing"][1:size])

    # Rf Classifier
    clean_clf = RandomForestClassifier().fit(clean_X_train, clean_y_train)
    explainer = shap.Explainer(clean_clf)
    shap_values = explainer.shap_values(clean_X)
    shap_values = shap_values[..., 0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    print(f"Explainer expected value: {expected_value}")

    shap.summary_plot(shap_values, clean_X)
