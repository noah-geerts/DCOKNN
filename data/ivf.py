import numpy as np
import faiss
import os
from fvec import read_fvecs, to_fvecs

source = './'
datasets = ['gist']
# the number of clusters
K = 41

if __name__ == '__main__':

    for dataset in datasets:
        print(f"Clustering - {dataset}")
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        randomized_cluster_path = os.path.join(path, f"O{dataset}_centroid_{K}.fvecs")
        transformation_path = os.path.join(path, 'O.fvecs')

        # read data vectors
        X = read_fvecs(data_path)
        P = read_fvecs(transformation_path)

        D = X.shape[1]
        
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        to_fvecs(centroids_path, centroids)

        # randomized centroids
        centroids = np.dot(centroids, P)
        to_fvecs(randomized_cluster_path, centroids)
