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



def pre_plot(data):
    '''Plot visualization to explore data before training.'''
    
    # Distribution of target
    distribution_plot(data, "phishing")

def distribution_plot(data, col_name):
    '''Plot distribution of selected categorical column.'''
    data[col_name].value_counts().sort_index().plot.bar(x="Target Value", y="Number of Occurrences", title="Distribution of Fraud Label")
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
    plt.show()
    
def confusion_plot(y_test, y_pred):
    '''Plot confusion matrix.'''
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.show()

def precision_recall(y_test, y_pred):
    '''Plot precision recall.'''
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
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
    plt.show()
    
def calibration_disp(model, X_test, y_test):
    '''Plot calibration display.'''
    CalibrationDisplay.from_estimator(model, X_test, y_test)
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
    plt.show()

def learning_curv(model, X_train, y_train):
    '''Plot learning curve.'''
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train)
    display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores, score_name="Score")
    display.plot()
    
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