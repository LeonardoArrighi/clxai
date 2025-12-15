"""
Visualization utilities for CLXAI experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_faithfulness_curves(results: Dict[str, Dict], output_path: Optional[str] = None):
    """Plot deletion and insertion curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors
    
    for i, (name, res) in enumerate(results.items()):
        curve = res.get('mean_deletion_curve', np.array([]))
        if len(curve) > 0:
            fractions = np.linspace(0, 1, len(curve))
            axes[0].plot(fractions, curve, label=name, color=colors[i % 10], linewidth=2)
        
        curve = res.get('mean_insertion_curve', np.array([]))
        if len(curve) > 0:
            fractions = np.linspace(0, 1, len(curve))
            axes[1].plot(fractions, curve, label=name, color=colors[i % 10], linewidth=2)
    
    axes[0].set_xlabel('Fraction removed')
    axes[0].set_ylabel('Confidence')
    axes[0].set_title('Deletion')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Fraction added')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Insertion')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_embedding_trajectory(results: Dict[str, Dict], output_path: Optional[str] = None):
    """Plot embedding drift curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    
    for i, (name, res) in enumerate(results.items()):
        curve = res.get('mean_drift_curve', np.array([]))
        if len(curve) > 0:
            fractions = np.linspace(0, 1, len(curve))
            ax.plot(fractions, curve, label=name, color=colors[i % 10], linewidth=2)
    
    ax.set_xlabel('Fraction perturbed')
    ax.set_ylabel('Embedding drift (L2)')
    ax.set_title('Embedding Trajectory Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_method_agreement(tau_matrix: np.ndarray, methods: List[str], output_path: Optional[str] = None):
    """Plot agreement matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tau_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(methods)
    
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f'{tau_matrix[i,j]:.2f}', ha='center', va='center')
    
    plt.colorbar(im, label="Kendall tau")
    plt.title('XAI Method Agreement')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_trajectory_animation(emb_2d, fractions, output_path, title="Trajectory"):
    """Create animation of embedding trajectory."""
    try:
        import imageio
        frames = []
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i in range(len(fractions)):
            ax.clear()
            ax.scatter(emb_2d[:i+1, 0], emb_2d[:i+1, 1], c=range(i+1), cmap='viridis', s=50)
            ax.plot(emb_2d[:i+1, 0], emb_2d[:i+1, 1], 'k-', alpha=0.5)
            ax.set_title(f'{title} - {fractions[i]*100:.0f}%')
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
        
        plt.close()
        imageio.mimsave(output_path, frames, fps=5)
    except ImportError:
        print("imageio not available")
