# Code to run the kfold CV with mutltiple random seeds 

final_valid_predictions = defaultdict(list)
final_predictions = []

for i in [1, 343 , 20, 129, 12, 54 , 65 , 23 , 89, 67] :
    xgb_params = {
        'random_state': i, 
        # gpu
    #     'tree_method': 'gpu_hist', 
    #     'gpu_id': 0, 
    #     'predictor': 'gpu_predictor',
        # cpu
        'n_jobs': 4,
        'booster': 'gbtree',
        'n_estimators': 8000,
        # optimized params
        'learning_rate': 0.03628302216953097,
        'reg_lambda': 0.0008746338866473539,
        'reg_alpha': 23.13181079976304,
        'subsample': 0.7875490025178415,
        'colsample_bytree': 0.11807135201147481,
        'max_depth': 3
    }

    
    scores = []
    for fold in range(5):
        xtrain = df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()
        ids = xvalid.id
        ytrain = xtrain.target8uib bkkiik 
        yvalid = xvalid.target

        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

        ordinal_encoder = preprocessing.OrdinalEncoder()
        xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
        xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
        xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

        model= XGBRegressor(**xgb_params)
        model.fit(
            xtrain, ytrain,
            early_stopping_rounds=300,
            eval_set=[(xvalid, yvalid)], 
            verbose=5000
        )
        preds_valid = model.predict(xvalid)
        test_preds = model.predict(xtest)
        final_predictions.append(test_preds)
        final_valid_predictions.update(dict(zip(ids, preds_valid)))
#         final_valid_predictions[ids].append(preds_valid)
        df[f'pred_{i}'] = df.id.map(final_valid_predictions)
        rmse = mean_squared_error(yvalid, preds_valid, squared=False)
        scores.append(rmse)
        print("Seed:",i, fold, rmse)

    print(np.mean(scores), np.std(scores))
