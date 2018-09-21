from sklearn.decomposition import PCA

liste = [[1,5,4]]

pca = PCA(n_components=2)
print(pca.fit_transform(liste))