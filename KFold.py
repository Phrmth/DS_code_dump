from sklearn.model_selection import KFold


kf = KFold(n_splits=5,random_state=48,shuffle=True)
final_prediction = np.zeros(test.shape[0])
rmse=[]  # list contains rmse for each fold
n=0
for trn_idx, test_idx in kf.split(X, y):
    X_tr,X_val=X.iloc[trn_idx],X.iloc[test_idx]
    y_tr,y_val=train['target'].iloc[trn_idx],train['target'].iloc[test_idx]
    
#     meta_model = Lasso(alpha =0.00001,max_iter=10000)
#     meta_model.fit(X_tr,y_tr)
    meta_model = xgb.XGBRegressor(**params_xgb)
    meta_model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=200,verbose=False)
    
    final_prediction +=meta_model.predict(X_test[columns+ ['lgbm', 'ElasticNet', 'LinearRegression', 'xgb']])/kf.n_splits
    rmse.append(mean_squared_error(y_val, meta_model.predict(X_val), squared=False))
    print(f"fold: {n+1}, rmse: {rmse[n]}")
    n+=1
