# Dealing with categorical data using get_dummies

dummies = pd.get_dummies(X_train.append(X_test)[cat_cols])
X_train[dummies.columns] = dummies.iloc[:len(X_train), :]
X_test[dummies.columns] = dummies.iloc[len(X_train): , :]
del(dummies)

# Dealing with categorical data using OrdinalEncoder (only when there are 3 or more levels)
ordinal_encoder = OrdinalEncoder()
X_train[cat_cols[3:]] = ordinal_encoder.fit_transform(X_train[cat_cols[3:]]).astype(int)
X_test[cat_cols[3:]] = ordinal_encoder.transform(X_test[cat_cols[3:]]).astype(int)
X_train = X_train.drop(cat_cols[:3], axis="columns")
X_test = X_test.drop(cat_cols[:3], axis="columns")


