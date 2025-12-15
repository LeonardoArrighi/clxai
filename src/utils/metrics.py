"""
Statistical metrics and analysis utilities for CLXAI.
"""

import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
from typing import List, Tuple, Optional, Dict
import torch


def compute_auc(curve: np.ndarray, normalize: bool = True) -> float:
    """
    Compute Area Under Curve using trapezoidal rule.
    
    Args:
        curve: Array of values at each perturbation step
        normalize: Whether to normalize by number of steps
    
    Returns:
        AUC value
    """
    x = np.linspace(0, 1, len(curve))
    auc = trapezoid(curve, x)
    return auc


def compute_auc_deletion(predictions: np.ndarray, target_class: int) -> float:
    """
    Compute AUC for deletion metric (removing important pixels first).
    Lower values indicate more faithful explanations.
    
    Args:
        predictions: Array of prediction probabilities at each step
        target_class: Original predicted class
    
    Returns:
        AUC-Deletion value
    """
    if len(predictions.shape) > 1:
        # Extract probability for target class
        probs = predictions[:, target_class]
    else:
        probs = predictions
    
    return compute_auc(probs)


def compute_auc_insertion(predictions: np.ndarray, target_class: int) -> float:
    """
    Compute AUC for insertion metric (adding important pixels first).
    Higher values indicate more faithful explanations.
    
    Args:
        predictions: Array of prediction probabilities at each step
        target_class: Original predicted class
    
    Returns:
        AUC-Insertion value
    """
    if len(predictions.shape) > 1:
        probs = predictions[:, target_class]
    else:
        probs = predictions
    
    return compute_auc(probs)


def compute_monotonicity(curve: np.ndarray) -> float:
    """
    Compute monotonicity as Spearman correlation between
    perturbation level and prediction change.
    
    For deletion: we expect negative correlation (removing important pixels decreases confidence)
    For insertion: we expect positive correlation
    
    Args:
        curve: Prediction curve as function of perturbation level
    
    Returns:
        Spearman correlation coefficient
    """
    x = np.arange(len(curve))
    correlation, p_value = stats.spearmanr(x, curve)
    return correlation


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_fn: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level
        seed: Random seed
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    n = len(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    point_estimate = statistic_fn(data)
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return point_estimate, ci_lower, ci_upper


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Tuple[float, float]:
    """
    Perform paired t-test between two sets of scores.
    
    Args:
        scores_a: Scores from model/method A
        scores_b: Scores from model/method B
    
    Returns:
        (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return t_stat, p_value


def kendall_tau_matrix(rankings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute Kendall's tau correlation matrix between different ranking methods.
    
    Args:
        rankings: Dictionary mapping method name to importance rankings
    
    Returns:
        Correlation matrix
    """
    methods = list(rankings.keys())
    n_methods = len(methods)
    
    tau_matrix = np.ones((n_methods, n_methods))
    
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i < j:
                tau, _ = stats.kendalltau(rankings[method_i], rankings[method_j])
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau
    
    return tau_matrix, methods


def embedding_drift(
    original_embedding: np.ndarray,
    perturbed_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute L2 distance between original and perturbed embeddings.
    
    Args:
        original_embedding: Original embedding vector (D,)
        perturbed_embeddings: Embeddings at each perturbation level (N, D)
    
    Returns:
        Array of distances at each perturbation level
    """
    if original_embedding.ndim == 1:
        original_embedding = original_embedding.reshape(1, -1)
    
    distances = np.linalg.norm(
        perturbed_embeddings - original_embedding,
        axis=1
    )
    return distances


def cluster_distance(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Compute distance from each embedding to its class centroid.
    
    Args:
        embeddings: Embedding vectors (N, D)
        centroids: Class centroids (C, D)
        labels: Class labels (N,)
    
    Returns:
        Distances to respective class centroids
    """
    distances = []
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        dist = np.linalg.norm(emb - centroids[label])
        distances.append(dist)
    return np.array(distances)


def trajectory_smoothness(embeddings: np.ndarray) -> float:
    """
    Compute trajectory smoothness as mean second derivative magnitude.
    Lower values indicate smoother trajectories.
    
    Args:
        embeddings: Embeddings along trajectory (N, D)
    
    Returns:
        Smoothness score (lower = smoother)
    """
    # First derivative (velocity)
    velocity = np.diff(embeddings, axis=0)
    
    # Second derivative (acceleration)
    acceleration = np.diff(velocity, axis=0)
    
    # Mean magnitude of acceleration
    smoothness = np.mean(np.linalg.norm(acceleration, axis=1))
    
    return smoothness


def compute_class_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 10
) -> np.ndarray:
    """
    Compute class centroids from embeddings.
    
    Args:
        embeddings: Embedding vectors (N, D)
        labels: Class labels (N,)
        n_classes: Number of classes
    
    Returns:
        Centroids (C, D)
    """
    dim = embeddings.shape[1]
    centroids = np.zeros((n_classes, dim))
    
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = embeddings[mask].mean(axis=0)
    
    return centroids


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Test AUC
    curve = np.array([1.0, 0.9, 0.7, 0.4, 0.2, 0.1])
    print(f"AUC: {compute_auc(curve):.4f}")
    
    # Test monotonicity
    print(f"Monotonicity: {compute_monotonicity(curve):.4f}")
    
    # Test bootstrap CI
    data = np.random.randn(100)
    est, lo, hi = bootstrap_ci(data)
    print(f"Bootstrap CI: {est:.4f} [{lo:.4f}, {hi:.4f}]")
    
    # Test trajectory smoothness
    embeddings = np.cumsum(np.random.randn(10, 128), axis=0)
    smoothness = trajectory_smoothness(embeddings)
    print(f"Trajectory smoothness: {smoothness:.4f}")
