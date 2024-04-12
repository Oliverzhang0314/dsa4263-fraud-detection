import pickle
from sklearn.model_selection import RandomizedSearchCV

def train(model, X_train, y_train, param_grid=None, path="./"):
    '''Train and tune hyperparameters for best performed model.'''
    
    # Grid search for best hyperparameter combinations
    if param_grid is not None:
        model = RandomizedSearchCV(model, param_grid, cv=5, n_iter=50, verbose=1)
    model.fit(X_train, y_train)
  
    # Get the best hyperparameters
    if param_grid is not None:
        best_params = model.best_params_
        print(f"Best Hyperparameters: {best_params}")
        
        # Save model
        with open(path, "wb") as f:
            pickle.dump(model, f)
        
        return model, best_params
    
    # Save model
    with open(path, "wb") as f:
        pickle.dump(model, f)
        
    return model