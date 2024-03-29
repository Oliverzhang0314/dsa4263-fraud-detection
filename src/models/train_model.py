from sklearn.model_selection import RandomizedSearchCV

def train(model, X_train, y_train, param_grid=None):
    '''Train and tune hyperparameters for best performed model.'''
    
    # Grid search for best hyperparameter combinations
    if param_grid is not None:
        model = RandomizedSearchCV(model, param_grid, cv=5, n_iter=20, verbose=1)
    model.fit(X_train, y_train)

    # Get the best hyperparameters
    if param_grid is not None:
        best_params = model.best_params_
        print(f"Best Hyperparameters: {best_params}")
        return model, best_params
    
    return model