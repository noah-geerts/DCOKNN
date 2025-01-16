import numpy as np
import struct
import os
from sklearn.neighbors import NearestNeighbors

source = './'
datasets = ['gist']

def read_fvecs(filename, c_contiguous=True):
    # read entire file as binary, reutrn empty array if no data
    # .fvecs contains a sequence of:
    # int32 specifying the dimensionality of a vector, then 'dim' float32 values representing the vector
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    
    # check if the data is in the correct format
    dim = fv.view(np.int32)[0] #fv.view(np.int32) interprets the data as an array of 32 bit integers
    assert dim > 0

    # reshape into a series of rows of length 1 + dim (1 for the int32, dim for the float32's)
    fv = fv.reshape(-1, 1 + dim)

    # verfiy that all vectors have the same dimensionality
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    
    # return only the float32 rows (remove the int32 row)
    fv = fv[:, 1:]

    # this step ensures the numpy array is contiguous in memory
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename, c_contiguous=True):
    # read entire file as binary, return empty array if no data
    # .ivecs contains a sequence of:
    # int32 specifying the dimensionality of a vector, then 'dim' int32 values representing the vector
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    
    # check if the data is in the correct format
    dim = iv[0]  # The first int32 value specifies the dimensionality
    assert dim > 0

    # reshape into a series of rows of length 1 + dim (1 for the int32, dim for the int32's)
    iv = iv.reshape(-1, 1 + dim)

    # verify that all vectors have the same dimensionality
    if not all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    
    # return only the int32 rows (remove the int32 row)
    iv = iv[:, 1:]

    # this step ensures the numpy array is contiguous in memory
    if c_contiguous:
        iv = iv.copy()
    return iv

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def to_ivecs(filename, data):
    print(f"Writing File - {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('i', x)
                fp.write(a)

def reduce(dataset):
    # write reduced number of vectors
    for subset, reduction_factor in zip(["_base.fvecs", "_query.fvecs"], [100, 10]):
        # get path to raw data
        path = os.path.join(source, dataset + "_raw")
        data_path = os.path.join(path, f'{dataset}{subset}')

        # read the raw data
        X = read_fvecs(data_path)
        n = X.shape[0]
        
        # randomly sample 1% of the original vectors
        X = X[np.random.choice(n, n//reduction_factor, replace=False)]

        # write the reduced vectors to a new file
        path = os.path.join(source, dataset)
        out_path = os.path.join(path, f'{dataset}{subset}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        to_fvecs(out_path, X)

    # get ground truth for reduced number of query vectors
    X = read_fvecs(os.path.join(source, dataset, f'{dataset}_base.fvecs'))
    Y = read_fvecs(os.path.join(source, dataset, f'{dataset}_query.fvecs'))
    GT = ground_truth(X, Y)

    # write ground truth as well
    to_ivecs(os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs'), GT)

def ground_truth(X, Y):
    # compute ground truth for a set of vectors
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)

    # return indices of 100 nearest neighbors of query vectors
    distances, indices = nbrs.kneighbors(Y)
    return indices
    
if __name__ == "__main__":
    for dataset in datasets:
        # Run this file first: grabs raw data that you downloaded as fvecs and renamed to (data name)_raw.fvecs,
        # Then reduces the number of vectors in the dataset by 100x and 10x for base and query vectors respectively,
        # Then computes the ground truth for the reduced query vectors so we now have a smaller dataset to work with
        # This produces a dataset_base.fvecs, dataset_query.fvecs, and dataset_groundtruth.ivecs

        # From this smaller dataset, first run randomize.py that randomly orthogonally rotates the dataset
        # This produces an Odataset_base.fvecs and O.fvecs, the first of which is the rotated dataset and the second
        # Of which is the rotation matrix used to rotate queries later

        # Then run ivf.py, which produces the centroids of the dataset and the randomized centroids of the dataset.
        # It needs to be run after because it uses the same O.fvecs and Odataset_base.fvecs files to build the randomized
        # Centroids. This produces dataset_centroid_K.fvecs and Odataset_centroid_K.fvecs
    
        reduce(dataset)