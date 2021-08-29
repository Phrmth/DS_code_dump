from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Stratifying the data

pca = PCA(n_components=16, random_state=0)
km = KMeans(n_clusters=32, random_state=0)

pca.fit(X_train)
km.fit(pca.transform(X_train))

print(np.unique(km.labels_, return_counts=True))

y_stratified = km.labels_


# Creating your folds for repeated use (for instance, stacking)

folds = 10
seed = 42

skf = StratifiedKFold(n_splits=folds,
                      shuffle=True, 
                      random_state=seed)

fold_idxs = list(skf.split(X_train, y_stratified))

#Checking the produced folds
for k, (train_idx, validation_idx) in enumerate(fold_idxs):
    print(f"fold {k} train idxs: {len(train_idx)} validation idxs: {len(validation_idx)}")
