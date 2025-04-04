import numpy as np
import os
import argparse
from sklearn.neighbors import NearestNeighbors
from fvec import read_fvecs, to_ivecs

def calculate_ground_truth(base_path, query_path, output_path):
    """
    Calculate ground truth nearest neighbors for a full dataset.
    Always finds 100 nearest neighbors to match fvec.py behavior.
    
    Args:
        base_path (str): Path to the base vectors file (.fvecs)
        query_path (str): Path to the query vectors file (.fvecs)
        output_path (str): Path to save the ground truth file (.ivecs)
    """
    print(f"Reading base vectors from {base_path}...")
    X = read_fvecs(base_path)
    print(f"Base vectors shape: {X.shape}")
    
    print(f"Reading query vectors from {query_path}...")
    Y = read_fvecs(query_path)
    print(f"Query vectors shape: {Y.shape}")
    
    print("Computing 100 nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)
    
    print("Finding nearest neighbors for all query vectors...")
    distances, indices = nbrs.kneighbors(Y)
    
    print(f"Ground truth shape: {indices.shape}")
    
    print(f"Saving ground truth to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    to_ivecs(output_path, indices)
    
    print("Ground truth calculation complete!")
    return indices

def main():
    parser = argparse.ArgumentParser(description='Calculate ground truth for a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., gist)')
    parser.add_argument('--source', type=str, default='./', help='Source directory (default: ./)')
    
    args = parser.parse_args()
    
    # Construct paths exactly as in fvec.py
    base_path = os.path.join(args.source, args.dataset, f'{args.dataset}_base.fvecs')
    query_path = os.path.join(args.source, args.dataset, f'{args.dataset}_query.fvecs')
    output_path = os.path.join(args.source, args.dataset, f'{args.dataset}_groundtruth.ivecs')
    
    # Calculate ground truth
    calculate_ground_truth(base_path, query_path, output_path)

if __name__ == "__main__":
    main() 