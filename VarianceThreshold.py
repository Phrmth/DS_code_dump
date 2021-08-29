from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold= .01).fit(X_train)
train2 = pd.DataFrame(sel.transform(X_train))
test2 = pd.DataFrame(sel.transform(X_test))

print(X_train.shape, train2.shape, test2.shape, X_test.shape)
