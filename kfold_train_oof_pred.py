# training code for a model using kfold and oof predictions

X = train[cols].copy()
cat_test_pred = np.zeros(len(test))
mse = []
kf = KFold(n_splits=5, shuffle=True)

for trn_idx, val_idx in tqdm(kf.split(X,y)):
    x_train_idx = X.iloc[trn_idx]
    y_train_idx = y.iloc[trn_idx]
    x_valid_idx = X.iloc[val_idx]
    y_valid_idx = y.iloc[val_idx]

    cat_model = CatBoostRegressor(learning_rate = .01, task_type = 'GPU')
    cat_model.fit(x_train_idx, y_train_idx, eval_set = ((x_valid_idx,y_valid_idx)),verbose = 100, early_stopping_rounds = 200,cat_features=cat_cols)  
    cat_test_pred += cat_model.predict(test)/5
    mse.append(mean_squared_error(y_valid_idx, cat_model.predict(x_valid_idx)))
    
print(np.mean(mse))
