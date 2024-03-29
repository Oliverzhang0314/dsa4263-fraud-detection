from sklearn.metrics import accuracy_score, recall_score

def predict(model, X_test, y_test=None):
    '''Predict using fine-tuned model'''
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    if y_test is not None:
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
    
    return y_pred