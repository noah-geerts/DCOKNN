import os
import numpy as np
from fvec import read_fvecs, to_fvecs
from sklearn.decomposition import PCA

source = './'
dataset = 'gist'

if __name__ == "__main__":
    # path for data
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    # read data vectors
    print(f"Reading {dataset} from {data_path}.")
    X = read_fvecs(data_path)
    D = X.shape[1]

    # Perform PCA using Scikit-learn
    # Keep all principal components so we can choose which to sample later during the queries
    pca = PCA(n_components=D) # O(D^3) + #O(N*D^2)

    # Compute and store transformed vectors with dimensions sorted by decreasing variance
    X_transformed = pca.fit_transform(X) #O(N*D^2)
    transformed_path = os.path.join(path, f'{dataset}_transformed.fvecs')
    to_fvecs(transformed_path, X_transformed)

    # Get and store projection matrix with dimensions sorted by decreasing variance
    pca_projection_matrix = pca.components_.T
    projection_path = os.path.join(path, f'{dataset}_projection.fvecs')
    to_fvecs(projection_path, pca_projection_matrix)

    # Compute the covariance matrix (D x D) and store it
    cov_matrix = pca.components_.T @ np.diag(pca.explained_variance_) @ pca.components_ #O(D^3)
    cov_path = os.path.join(path, f'{dataset}_covariance.fvecs')
    to_fvecs(cov_path, cov_matrix)