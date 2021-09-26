# Predictions from different models
feat = ['ebc_pred_1', 'xgb_pred_1', 'cb_pred_1', 'ri_pred_1', 'lgbm_pred_1', 'lr_pred_1']

def objective(trial):

# Suggested weights for the differents predictions

  w_1 = trial.suggest_uniform('w_1', 0, 1)
  w_2 = trial.suggest_uniform('w_2', 0, 1)
  w_3 = trial.suggest_uniform('w_3', 0, 1)
  w_4 = trial.suggest_uniform('w_4', 0, 1)
  w_5 = trial.suggest_uniform('w_5', 0, 1)
  w_6 = trial.suggest_uniform('w_6', 0, 1)

  fold = 0

  X_train = train[train.kfold != fold].reset_index(drop=True)
  X_valid = train[train.kfold == fold].reset_index(drop=True)

  y_train = X_train['claim']
  y_valid = X_valid['claim']

  X_train = X_train[feat]
  X_valid = X_valid[feat]

  pred_0=X_valid.ebc_pred_1
  pred_1=X_valid.xgb_pred_1
  pred_2= X_valid.cb_pred_1
  pred_3=X_valid.ri_pred_1
  pred_4=X_valid.lgbm_pred_1
  pred_5=X_valid.lr_pred_1

  pred = (w_1*pred_0 + w_2*pred_1 + w_3*pred_2 + w_4*pred_3 + w_5*pred_4 + w_6*pred_5) / 6

  acc = np.mean((y_valid - pred)**2)

  return acc


study1 = optuna.create_study(direction="minimize")
study1.optimize(objective, n_trials= 200, show_progress_bar=True)
