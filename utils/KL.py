import numpy as np
from scipy.stats import gaussian_kde, entropy
import matplotlib.pyplot as plt

def compute_kl_divergence(sample1, sample2, num_points=1000):
    """Compute KL divergence between two sets of samples using KDE."""
    kde1 = gaussian_kde(sample1)
    kde2 = gaussian_kde(sample2)
    
    # Define a grid for evaluation
    x_eval = np.linspace(min(min(sample1), min(sample2)), max(max(sample1), max(sample2)), num_points)
    p = kde1(x_eval) + 1e-10  # Avoid zero densities
    q = kde2(x_eval) + 1e-10
    
    # Normalize the densities
    p /= np.sum(p)
    q /= np.sum(q)
    
    # Calculate KL divergence
    return entropy(p, q)
