import numpy as np
import os
import argparse
from fvec import read_fvecs, read_ivecs

def verify_ground_truth(base_path, query_path, ground_truth_path, sample_size=10):
    """
    Verify ground truth vectors by comparing with direct distance calculations.
    
    Args:
        base_path (str): Path to the base vectors file (.fvecs)
        query_path (str): Path to the query vectors file (.fvecs)
        ground_truth_path (str): Path to the ground truth file (.ivecs)
        sample_size (int): Number of query vectors to verify (default: 10)
    """
    print(f"Reading base vectors from {base_path}...")
    X = read_fvecs(base_path)
    print(f"Base vectors shape: {X.shape}")
    
    print(f"Reading query vectors from {query_path}...")
    Y = read_fvecs(query_path)
    print(f"Query vectors shape: {Y.shape}")
    
    print(f"Reading ground truth from {ground_truth_path}...")
    GT = read_ivecs(ground_truth_path)
    print(f"Ground truth shape: {GT.shape}")
    
    # Verify shapes match
    if GT.shape[0] != Y.shape[0]:
        print(f"ERROR: Ground truth has {GT.shape[0]} rows but query has {Y.shape[0]} rows")
        return False
    
    if GT.shape[1] != 100:
        print(f"ERROR: Ground truth has {GT.shape[1]} columns, expected 100")
        return False
    
    # Sample a few query vectors to verify
    num_queries = min(sample_size, Y.shape[0])
    sample_indices = np.random.choice(Y.shape[0], num_queries, replace=False)
    
    print(f"\nVerifying {num_queries} random query vectors...")
    
    all_correct = True
    for i, query_idx in enumerate(sample_indices):
        print(f"\nVerifying query vector {i+1}/{num_queries} (index {query_idx})...")
        
        # Get the query vector
        query = Y[query_idx]
        
        # Get the ground truth indices for this query
        gt_indices = GT[query_idx]
        
        # Calculate distances to all base vectors
        print("Calculating distances to all base vectors...")
        distances = np.sum((X - query) ** 2, axis=1)
        
        # Get the indices of the 100 nearest neighbors
        true_indices = np.argsort(distances)[:100]
        
        # Compare with ground truth
        gt_set = set(gt_indices)
        true_set = set(true_indices)
        
        # Check if all ground truth indices are in the true nearest neighbors
        missing = gt_set - true_set
        if missing:
            print(f"ERROR: Ground truth contains {len(missing)} indices not in the true 100 nearest neighbors")
            print(f"Missing indices: {list(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            all_correct = False
        else:
            print("All ground truth indices are in the true 100 nearest neighbors")
        
        # Check if the order is correct (optional, as order might not matter)
        # For simplicity, we'll just check if the first few match
        first_few = min(5, len(gt_indices))
        if not np.array_equal(gt_indices[:first_few], true_indices[:first_few]):
            print(f"WARNING: First {first_few} indices in ground truth don't match the true order")
            print(f"Ground truth first {first_few}: {gt_indices[:first_few]}")
            print(f"True first {first_few}: {true_indices[:first_few]}")
        else:
            print(f"First {first_few} indices in ground truth match the true order")
    
    if all_correct:
        print("\nVERIFICATION PASSED: Ground truth appears to be correct!")
    else:
        print("\nVERIFICATION FAILED: Ground truth contains errors!")
    
    return all_correct

def main():
    parser = argparse.ArgumentParser(description='Verify ground truth for a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., gist)')
    parser.add_argument('--source', type=str, default='./', help='Source directory (default: ./)')
    parser.add_argument('--sample', type=int, default=10, help='Number of query vectors to verify (default: 10)')
    
    args = parser.parse_args()
    
    # Construct paths
    base_path = os.path.join(args.source, args.dataset, f'{args.dataset}_base.fvecs')
    query_path = os.path.join(args.source, args.dataset, f'{args.dataset}_query.fvecs')
    ground_truth_path = os.path.join(args.source, args.dataset, f'{args.dataset}_groundtruth.ivecs')
    
    # Verify ground truth
    verify_ground_truth(base_path, query_path, ground_truth_path, args.sample)

if __name__ == "__main__":
    main() 