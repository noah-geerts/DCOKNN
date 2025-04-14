import sys
import os
import numpy as np
import random
from fvec import read_fvecs, to_fvecs, to_ivecs
from sklearn.neighbors import NearestNeighbors

# === Global Dataset Name ===
dataset = 'gist'
source = './'

def compute_ground_truth(X, Y, k=100):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    _, indices = nbrs.kneighbors(Y)
    return indices

def reduce_dimensions(percentage):
    assert 0 < percentage <= 100, "Percentage must be in (0, 100]"

    input_dir = os.path.join(source, dataset)
    output_dir = os.path.join(source, f"{dataset}-{percentage}-percent")
    os.makedirs(output_dir, exist_ok=True)

    reduced_data = {}

    for subset in ['_base.fvecs', '_query.fvecs']:
        in_file = os.path.join(input_dir, f"{dataset}{subset}")
        vectors = read_fvecs(in_file)

        n, d = vectors.shape
        d_new = max(1, int(d * (percentage / 100)))
        selected_dims = sorted(random.sample(range(d), d_new))

        reduced_vectors = vectors[:, selected_dims]
        reduced_data[subset] = reduced_vectors

        out_file = os.path.join(output_dir, f"{dataset}{subset}")
        to_fvecs(out_file, reduced_vectors)

        print(f"{subset}: Reduced from {d} to {d_new} dimensions")

    # Compute and write ground truth
    GT = compute_ground_truth(reduced_data['_base.fvecs'], reduced_data['_query.fvecs'])
    gt_file = os.path.join(output_dir, f"{dataset}_groundtruth.ivecs")
    to_ivecs(gt_file, GT)
    print("Ground truth saved:", gt_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reduce_dims.py <percentage>")
        sys.exit(1)

    try:
        percentage = float(sys.argv[1])
    except ValueError:
        print("Percentage must be a number.")
        sys.exit(1)

    reduce_dimensions(percentage)
