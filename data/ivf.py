import numpy as np
import faiss
import os
from fvec import read_fvecs, to_fvecs

source = './'
dataset = 'gist'
dataset_N = 10000

#K is the number of clusters
sqrt_N = int(np.sqrt(dataset_N))
k_list = [sqrt_N]
k_list = [int(k) for k in k_list]

if __name__ == '__main__':
    for K in k_list:
        print(f"Clustering - {dataset} with {K} clusters")

        # data paths
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')

        # centroid paths
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        randomized_centroids_path = os.path.join(path, f"O{dataset}_centroid_{K}.fvecs")
        pca_centroids_path = os.path.join(path, f"PCA{dataset}_centroid_{K}.fvecs")

        # random transformation and projection paths
        transformation_path = os.path.join(path, 'O.fvecs')
        projection_path = os.path.join(path, f'PCA.fvecs')

        # read data vectors
        X = read_fvecs(data_path)
        O = read_fvecs(transformation_path)
        PCA = read_fvecs(projection_path)

        D = X.shape[1]
        
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        to_fvecs(centroids_path, centroids)

        # randomized centroids
        centroids_randomized = np.dot(centroids, O)
        print(centroids_randomized.shape)
        to_fvecs(randomized_centroids_path, centroids_randomized)

        # pca-space centroids
        centroids_pca = np.dot(centroids, PCA)
        print(centroids_pca.shape)
        to_fvecs(pca_centroids_path, centroids_pca)
