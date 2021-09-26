def cross_validate_model(class_name, class_params, train_data, test_data, n_splits=5):
    
    X = train_data[features].to_numpy()
    Y = train_data[TARGET]
    X_test = test_data[features].to_numpy()
    
    skfolds = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    oof_preds, oof_y = [], []
    
    test_preds = np.zeros((X_test.shape[0]))
    
    for i, (train_index, val_index) in enumerate(skfolds.split(X, Y)):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        
        print(f"{'-'*10} Fold {i+1} Started {'-'*10}")
        clf = class_name(**class_params)
    
        clf = clf.fit(x_train, y_train)
        preds = clf.predict_proba(x_val)
        
        oof_preds.extend(preds[:, 1])
        oof_y.extend(y_val)
        
        test_preds += clf.predict_proba(X_test)[:, 1]
        
        ra_score = metrics.roc_auc_score(y_val, preds[:, 1])
    
        print(f"ROC AUC of current fold is {ra_score}")
        
    ra_score = metrics.roc_auc_score(oof_y, oof_preds)
    
    print(f"\nOverall ROC AUC is {ra_score}")
    
    return oof_preds, test_preds / n_splits
  
  
  
  xgb_params = {
    'n_estimators' : 3600,
    'reg_lambda' : 3,
    'reg_alpha' : 26,
    'subsample' : 0.6000000000000001,
    'colsample_bytree' : 0.6000000000000001,
    'max_depth' : 9,
    'min_child_weight' : 5,
    'gamma' : 13.054739572819486,
    'learning_rate': 0.01,
    'tree_method': 'gpu_hist',
    'booster': 'gbtree'
}

lgbm_params = {
    "objective": "binary",
    "learning_rate": 0.008,
    'device': 'gpu',
    'n_estimators': 3205,
    'num_leaves': 184,
    'min_child_samples': 63,
    'feature_fraction': 0.6864594334728974,
    'bagging_fraction': 0.9497327922401265,
    'bagging_freq': 1,
    'reg_alpha': 19,
    'reg_lambda': 19,
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

catb_params = {
    'iterations': 15585, 
    'objective': 'CrossEntropy', 
    'bootstrap_type': 'Bernoulli', 
    'od_wait': 1144, 
    'learning_rate': 0.023575206684596582, 
    'reg_lambda': 36.30433203563295, 
    'random_strength': 43.75597655616195, 
    'depth': 7, 
    'min_data_in_leaf': 11, 
    'leaf_estimation_iterations': 1, 
    'subsample': 0.8227911142845009,
    'task_type' : 'GPU',
    'devices' : '0',
    'verbose' : 0
}



lv1_oof = pd.DataFrame()
lv1_test = pd.DataFrame()


oof_preds, test_preds = cross_validate_model(XGBClassifier, 
                                             xgb_params, 
                                             train_data, 
                                             test_data,
                                             N_FOLDS)

lv1_oof['xgb'] = oof_preds
lv1_test['xgb'] = test_preds

catb_params['random_state'] = 42
oof_preds, test_preds = cross_validate_model(CatBoostClassifier, 
                                             catb_params, 
                                             train_data, 
                                             test_data,
                                             N_FOLDS)

lv1_oof['catb_1'] = oof_preds
lv1_test['catb_1'] = test_preds

catb_params['random_state'] = 2021
oof_preds, test_preds = cross_validate_model(CatBoostClassifier, 
                                             catb_params, 
                                             train_data, 
                                             test_data,
                                             N_FOLDS)

lv1_oof['catb_2'] = oof_preds
lv1_test['catb_2'] = test_preds
