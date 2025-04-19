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
    
    # Calculate statistics of skewness and kurtosis
    skew_mean = np.mean(skewness)
    skew_std = np.std(skewness)
    kurt_mean = np.mean(kurtosis)
    kurt_std = np.std(kurtosis)
    
    # Calculate deviation from normal distribution (0 skewness, 0 kurtosis)
    deviation = np.abs(skewness) + np.abs(kurtosis)
    
    # Get indices of dimensions with highest deviation
    top_dimensions = np.argsort(deviation)[-3:]  # Get top 3 most non-normal dimensions
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    # First subplot: histograms of the most non-normal dimensions
    ax1 = fig.add_subplot(gs[0])
    for idx, dim in enumerate(top_dimensions):
        ax = fig.add_subplot(gs[0].subgridspec(3, 1)[idx])
        ax.hist(vectors[:, dim], bins=1000, density=True, alpha=0.6, color='g')
        
        # Add normal distribution for comparison
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        p = stats.norm.pdf(x, np.mean(vectors[:, dim]), np.std(vectors[:, dim]))
        ax.plot(x, p, 'k', linewidth=1.5, alpha=1)
        
        ax.set_title(f'Dimension {dim}\nSkew: {skewness[dim]:.2f}, Kurtosis: {kurtosis[dim]:.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    # Second subplot: distribution of skewness and kurtosis across all dimensions
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(skewness, bins=50, alpha=0.5, label='Skewness', color='blue')
    ax2.hist(kurtosis, bins=50, alpha=0.5, label='Kurtosis', color='red')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Reference line at 0
    ax2.set_title('Distribution of Skewness and Kurtosis across all Dimensions')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    # Add text with statistics
    stats_text = (f'Across all dimensions:\n'
                 f'Skewness: mean = {skew_mean:.2f} ± {skew_std:.2f}\n'
                 f'Kurtosis: mean = {kurt_mean:.2f} ± {kurt_std:.2f}\n'
                 f'% dimensions with |skewness| > 1: {np.mean(np.abs(skewness) > 1)*100:.1f}%\n'
                 f'% dimensions with |kurtosis| > 1: {np.mean(np.abs(kurtosis) > 1)*100:.1f}%')
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'./{dataset}/vector_distributions.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    dataset = "gist"  # You can change this to your dataset name
    analyze_distributions(dataset) 