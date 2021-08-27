from sklearn.model_selection import KFold

pred1 = np.zeros(train.shape[0])
pred2 = np.zeros(train.shape[0])
pred3 = np.zeros(train.shape[0])
pred4 = np.zeros(train.shape[0])

test1 = np.zeros(test.shape[0])
test2 = np.zeros(test.shape[0])
test3 = np.zeros(test.shape[0])
test4 = np.zeros(test.shape[0])

kf = KFold(n_splits=5,random_state=48,shuffle=True)
n=0

for trn_idx, test_idx in kf.split(train[columns],train['target']):
    print(f"fold: {n+1}")
    X_tr,X_val=train[columns].iloc[trn_idx],train[columns].iloc[test_idx]
    y_tr,y_val=train['target'].iloc[trn_idx],train['target'].iloc[test_idx]
    
    
    model1 = lgb.LGBMRegressor(**params_lgb)
    model1.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=200,verbose=False)
    pred1[test_idx] = model1.predict(X_val)
    test1 += model1.predict(test[columns])/kf.n_splits
    rmse1 = mean_squared_error(y_val, model1.predict(X_val), squared=False)
    print(": model1 rmse = {}".format(rmse1))

    model2 = ElasticNet(alpha=0.00001)
    model2.fit(X_tr,y_tr)
    pred2[test_idx] = model2.predict(X_val)
    test2 += model2.predict(test[columns])/kf.n_splits
    rmse2 = mean_squared_error(y_val, model2.predict(X_val), squared=False)
    print(": model2 rmse = {}".format(rmse2))
    
    model3 = LinearRegression()
    model3.fit(X_tr,y_tr)
    pred3[test_idx] = model3.predict(X_val)
    test3 += model3.predict(test[columns])/kf.n_splits
    rmse3 = mean_squared_error(y_val, model3.predict(X_val), squared=False)
    print(": model3 rmse = {}".format(rmse3))
    
    model4 = xgb.XGBRegressor(**params_xgb)
    model4.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=200,verbose=False)
    pred4[test_idx] = model4.predict(X_val)
    test4 += model4.predict(test[columns])/kf.n_splits
    rmse4 = mean_squared_error(y_val, model4.predict(X_val), squared=False)
    print(": model4 rmse = {}".format(rmse4))
    print(": average all models rmse = {}".format((rmse1+rmse2+rmse3+rmse4)/4))

    n+=1
