import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from fvec import read_fvecs

def analyze_distributions(dataset):
    # Read the dataset vectors
    data_path = f"./{dataset}/{dataset}_base.fvecs"
    vectors = read_fvecs(data_path)
    
    # Calculate skewness and kurtosis for each dimension
    n_dimensions = vectors.shape[1]
    skewness = np.zeros(n_dimensions)
    kurtosis = np.zeros(n_dimensions)
    
    for i in range(n_dimensions):
        skewness[i] = stats.skew(vectors[:, i])
        kurtosis[i] = stats.kurtosis(vectors[:, i])
    
    # Calculate average skewness and kurtosis
    avg_skew = np.mean(skewness)
    avg_kurtosis = np.mean(kurtosis)
    
    # Calculate deviation from normal distribution (0 skewness, 0 kurtosis)
    deviation = np.abs(skewness) + np.abs(kurtosis)
    
    # Get indices of dimensions with highest deviation
    top_dimensions = np.argsort(deviation)[-3:]  # Get top 3 most non-normal dimensions
    
    # Create subplots for the top non-normal dimensions
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    for idx, dim in enumerate(top_dimensions):
        ax = axes[idx]
        ax.hist(vectors[:, dim], bins=1000, density=True, alpha=0.6, color='g')
        
        # Add normal distribution for comparison with higher granularity
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)  # Use actual xmin and xmax
        p = stats.norm.pdf(x, np.mean(vectors[:, dim]), np.std(vectors[:, dim]))
        ax.plot(x, p, 'k', linewidth=1.5, alpha=1)  # Thinner line with slight transparency
        
        ax.set_title(f'Dimension {dim}\nSkew: {skewness[dim]:.2f}, Kurtosis: {kurtosis[dim]:.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    # Add text with average statistics
    fig.text(0.02, 0.02, f'Average across all dimensions:\nSkewness: {avg_skew:.2f}\nKurtosis: {avg_kurtosis:.2f}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'./{dataset}/vector_distributions.png', dpi=300)  # Higher DPI for better line quality
    plt.close()

if __name__ == "__main__":
    dataset = "glove1.2m"  # You can change this to your dataset name
    analyze_distributions(dataset) 