import os
import numpy as np
from fvec import read_fvecs, to_fvecs
import time

source = './'
dataset = 'word2vec'

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

    # Start timing after data is read
    start_time = time.perf_counter()

    # generate random orthogonal matrix, store it and apply it
    print(f"Randomizing {dataset} of dimensionality {D}.")
    P = Orthogonal(D)
    XP = np.dot(X, P)

    projection_path = os.path.join(path, 'O.fvecs')
    transformed_path = os.path.join(path, f'O{dataset}_base.fvecs')

    # End timing and save results
    end_time = time.perf_counter()
    total_time = int((end_time - start_time) * 1e6)  # Convert to microseconds
    
    # Save timing results
    result_dir = os.path.join('..', 'index_results', dataset)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'ADS_projection.log'), 'w') as f:
        f.write(f"{total_time}\n")
    
    print(f"Total orthogonal matrix generation and projection time: {total_time} microseconds")

    to_fvecs(projection_path, P)
    to_fvecs(transformed_path, XP)
