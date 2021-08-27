import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 12000, step=100),
        "max_depth":trial.suggest_int("max_depth", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0, step=0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 100.),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 100.),
    }
    
    
    model = xgb.XGBRegressor(
        **params,
        n_jobs=-1, 
#         tree_method='gpu_hist', 
#         gpu_id=0
    )
    
    model.fit(X_train, y_train, 
              early_stopping_rounds=300, 
              eval_set=[(X_valid, y_valid)],
              verbose=0)
    
    y_hat = model.predict(X_valid)
    
    return mean_squared_error(y_valid, y_hat, squared=False)

study = optuna.create_study()
study.optimize(objective, n_trials=25)
