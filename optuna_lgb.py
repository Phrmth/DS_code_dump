
def train_model_optuna(trial, X_train, X_valid, y_train, y_valid):
    """
    A function to train a model using different hyperparamerters combinations provided by Optuna. 
    Loss of validation data predictions is returned to estimate hyperparameters effectiveness.
    """
    preds = 0
    
        
    #A set of hyperparameters to optimize by optuna
    lgbm_params = {
                    "objective": trial.suggest_categorical("objective", ['binary']),
                    "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt']),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                    "max_depth": trial.suggest_int("max_depth", 1, 16),
                    "learning_rate": trial.suggest_float("learning_rate", 0.1, 1, step=0.01),
                    "n_estimators": trial.suggest_categorical("n_estimators", [40000]),        
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 100.0, step=0.1),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, step=0.1),
                    "random_state": trial.suggest_categorical("random_state", [42]),
                    "bagging_seed": trial.suggest_categorical("bagging_seed", [42]),
                    "feature_fraction_seed": trial.suggest_categorical("feature_fraction_seed", [42]), 
                    "n_jobs": trial.suggest_categorical("n_jobs", [4]), 
                    "subsample": trial.suggest_float("subsample", 0.6, 1, step=0.01),
                    "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1, step=0.01),
#                     "device_type": trial.suggest_categorical("device_type", ["GPU"]),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'min_child_weight': trial.suggest_categorical('min_child_weight', [256]),
        
                    }
