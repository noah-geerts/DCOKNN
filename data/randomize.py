import os
import numpy as np
from fvec import read_fvecs, to_fvecs

source = './'
dataset = 'gist'

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

if __name__ == "__main__":
    # seed for reproducibility
    np.random.seed(0)

    # path for data
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    # read data vectors
    print(f"Reading {dataset} from {data_path}.")
    X = read_fvecs(data_path)
    D = X.shape[1]

    # generate random orthogonal matrix, store it and apply it
    print(f"Randomizing {dataset} of dimensionality {D}.")
    P = Orthogonal(D)
    XP = np.dot(X, P)

    projection_path = os.path.join(path, 'O.fvecs')
    transformed_path = os.path.join(path, f'O{dataset}_base.fvecs')

    to_fvecs(projection_path, P)
    to_fvecs(transformed_path, XP)
