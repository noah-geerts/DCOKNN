import numpy as np
import faiss
import os
from fvec import read_fvecs, to_fvecs
import time

source = './'
dataset = 'word2vec'

if __name__ == '__main__':
    # data paths
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    # random transformation and projection paths
    transformation_path = os.path.join(path, 'O.fvecs')
    projection_path = os.path.join(path, f'PCA.fvecs')

    # read data vectors
    X = read_fvecs(data_path)
    O = read_fvecs(transformation_path)
    PCA = read_fvecs(projection_path)

    D = X.shape[1]

    K = int(np.sqrt(X.shape[0]))
    print(f"Clustering - {dataset} with {K} clusters")

    # centroid paths
    centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
    randomized_centroids_path = os.path.join(path, f"O{dataset}_centroid_{K}.fvecs")
    pca_centroids_path = os.path.join(path, f"PCA{dataset}_centroid_{K}.fvecs")
    
    # Time the clustering process
    start_time = time.perf_counter()
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    end_time = time.perf_counter()
    clustering_time = int((end_time - start_time) * 1e6)  # Convert to microseconds
    
    # Save clustering time
    result_dir = os.path.join('..', 'index_results', dataset)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'IVF_clustering.log'), 'w') as f:
        f.write(f"{clustering_time}\n")

    # Time the randomized projection
    start_time = time.perf_counter()
    centroids_randomized = np.dot(centroids, O)
    end_time = time.perf_counter()
    randomized_time = int((end_time - start_time) * 1e6)  # Convert to microseconds
    
    # Save randomized projection time
    with open(os.path.join(result_dir, 'IVF_ADS_clustering.log'), 'w') as f:
        f.write(f"{randomized_time}\n")

    # Time the PCA projection
    start_time = time.perf_counter()
    mean_path = os.path.join(path, f'PCA_mean.fvecs')
    pca_mean = read_fvecs(mean_path)
    centroids_pca = np.dot(centroids - pca_mean, PCA)
    end_time = time.perf_counter()
    pca_time = int((end_time - start_time) * 1e6)  # Convert to microseconds
    
    # Save PCA projection time
    with open(os.path.join(result_dir, 'IVF_APCA_clustering.log'), 'w') as f:
        f.write(f"{pca_time}\n")

    print(f"Clustering time: {clustering_time} microseconds")
    print(f"Randomized projection time: {randomized_time} microseconds")
    print(f"PCA projection time: {pca_time} microseconds")

    # Write all vectors and matrices after timing
    to_fvecs(pca_centroids_path, centroids_pca)
    to_fvecs(randomized_centroids_path, centroids_randomized)
    to_fvecs(centroids_path, centroids)
