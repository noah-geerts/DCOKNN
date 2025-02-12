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
    print(X_transformed.shape)
    transformed_path = os.path.join(path, f'PCA{dataset}_base.fvecs')
    to_fvecs(transformed_path, X_transformed)

    # Get and store projection matrix with dimensions sorted by decreasing variance
    pca_projection_matrix = pca.components_.T
    print(pca_projection_matrix.shape)
    projection_path = os.path.join(path, f'PCA.fvecs')
    to_fvecs(projection_path, pca_projection_matrix)

    # Compute the covariance matrix (D x D) and store it
    cov_matrix = pca.components_.T @ np.diag(pca.explained_variance_) @ pca.components_ #O(D^3)
    print(cov_matrix.shape)
    cov_path = os.path.join(path, f'COV.fvecs')
    to_fvecs(cov_path, cov_matrix)

    # Get and store the mean of the data
    mean_vector_2d = np.expand_dims(pca.mean_, axis=0)
    print(mean_vector_2d.shape)
    mean_path = os.path.join(path, f'PCA_mean.fvecs')
    to_fvecs(mean_path, mean_vector_2d)

    # Calculate variance along each dimension
    variance_vector = np.var(X, axis=0, keepdims=True)  # Shape: (1, D)
    print("Variance shape:", variance_vector.shape)
    variance_path = os.path.join(path, f'{dataset}_variances.fvecs')
    to_fvecs(variance_path, variance_vector)

    # Calculate and store magnitude of each data vector
    magnitudes = np.square(np.linalg.norm(X_transformed, axis=1, keepdims=True).T)  # Shape: (1, N)
    print("Magnitudes shape:", magnitudes.shape)
    magnitudes_path = os.path.join(path, f'{dataset}_magnitudes.fvecs')
    to_fvecs(magnitudes_path, magnitudes)