import sklearn
from sklearn.model_selection import train_test_split
from lazypredict import Supervised
from lazypredict.Supervised import LazyClassifier


def lazy_predict(data, exclude=None):
    '''Initial prediction using multiple choices of classifiers'''
    
    # Train test split
    X = data.drop("phishing", axis=1)
    y = data["phishing"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=data["phishing"])
    
     # Remove problematic/bad performance classifiers from Supervised.CLASSIFIERS
    exclude_lst = [
                sklearn.svm.NuSVC,
                sklearn.svm.SVC,
                sklearn.svm.LinearSVC,
                sklearn.calibration.CalibratedClassifierCV,
                sklearn.semi_supervised._label_propagation.LabelPropagation,
                sklearn.semi_supervised._label_propagation.LabelSpreading,
                sklearn.naive_bayes.CategoricalNB
            ]
    
    if exclude is not None:
        exclude_lst.extend(exclude)
    
    Supervised.CLASSIFIERS = [classifier for classifier in Supervised.CLASSIFIERS
                            if classifier[1] not in exclude_lst]

    # Lazy prediction
    clf = LazyClassifier(verbose=1, ignore_warnings=False, custom_metric=None, predictions=True, classifiers="all") # verbose set as 1 to monitor progress
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    return models