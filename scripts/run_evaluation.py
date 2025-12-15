#!/usr/bin/env python3
"""
Main evaluation script for CLXAI experiments.
Runs pixel flipping and embedding analysis for CE vs SCL comparison.
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.resnet import ResNet18, ResNet18Encoder
from src.models.classifiers import KNNClassifier, LinearClassifier
from src.utils.data import get_cifar10_loaders
from src.xai.saliency import SaliencyExtractor
from src.xai.pixel_flipping import run_pixel_flipping_experiment
from src.analysis.embedding_analysis import analyze_embedding_trajectory
from src.analysis.faithfulness import compute_faithfulness_metrics
from src.analysis.visualization import plot_faithfulness_curves, plot_embedding_trajectory


def load_ce_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load Cross-Entropy trained model."""
    model = ResNet18(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded CE model from {checkpoint_path}")
    print(f"  Test accuracy: {checkpoint.get('test_acc', 'N/A')}")
    return model


def load_scl_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load SCL trained encoder."""
    model = ResNet18Encoder(embedding_dim=128)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded SCL model from {checkpoint_path}")
    print(f"  kNN accuracy: {checkpoint.get('knn_acc', 'N/A')}")
    return model


def run_experiments(config: dict):
    """Run full experiment suite."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results_dir = Path(config['output']['results_dir'])
    figures_dir = Path(config['output']['figures_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-10 data...")
    _, test_loader = get_cifar10_loaders(
        data_dir='./data',
        batch_size=64,
        num_workers=4,
        augment=False
    )
    
    # Load models
    print("\nLoading models...")
    models = {}
    
    # CE model
    ce_config = config['models'].get('ce_baseline', {})
    if ce_config and Path(ce_config['checkpoint']).exists():
        models['CE'] = load_ce_model(ce_config['checkpoint'], device)
    
    # SCL model
    scl_config = config['models'].get('scl_knn', {})
    if scl_config and Path(scl_config['checkpoint']).exists():
        models['SCL'] = load_scl_model(scl_config['checkpoint'], device)
    
    if not models:
        print("No models found. Please train models first.")
        return
    
    # Run experiments for each model
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name} model")
        print('='*50)
        
        # Get target layer for GradCAM
        if hasattr(model, 'encoder'):
            target_layer = model.encoder.layer4[-1].conv2
        else:
            target_layer = model.layer4[-1].conv2 if hasattr(model, 'layer4') else None
        
        # Initialize saliency extractor
        extractor = SaliencyExtractor(model, device=device, target_layer=target_layer)
        
        model_results = {}
        
        # Run pixel flipping for each XAI method
        for xai_method in config.get('xai_methods', ['integrated_grad']):
            print(f"\n  XAI Method: {xai_method}")
            
            for perturbation in config.get('perturbations', ['mean']):
                print(f"    Perturbation: {perturbation}")
                
                try:
                    pf_results = run_pixel_flipping_experiment(
                        model=model,
                        data_loader=test_loader,
                        saliency_extractor=extractor,
                        perturbation=perturbation,
                        saliency_method=xai_method,
                        n_samples=config['pixel_flipping'].get('n_samples', 100),
                        n_steps=config['pixel_flipping'].get('n_steps', 20),
                        device=device
                    )
                    
                    key = f"{xai_method}_{perturbation}"
                    model_results[key] = {
                        'deletion_auc_mean': pf_results['mean_deletion_auc'],
                        'deletion_auc_std': pf_results['std_deletion_auc'],
                        'insertion_auc_mean': pf_results['mean_insertion_auc'],
                        'insertion_auc_std': pf_results['std_insertion_auc'],
                        'mean_deletion_curve': pf_results['mean_deletion_curve'].tolist(),
                        'mean_insertion_curve': pf_results['mean_insertion_curve'].tolist()
                    }
                    
                    print(f"      Deletion AUC: {pf_results['mean_deletion_auc']:.4f}")
                    print(f"      Insertion AUC: {pf_results['mean_insertion_auc']:.4f}")
                    
                except Exception as e:
                    print(f"      Error: {e}")
        
        # Run embedding analysis
        print(f"\n  Embedding Analysis...")
        try:
            emb_results = analyze_embedding_trajectory(
                model=model,
                data_loader=test_loader,
                saliency_extractor=extractor,
                saliency_method='integrated_grad',
                perturbation='mean',
                n_samples=config['embedding_analysis'].get('n_samples', 100),
                n_steps=config['embedding_analysis'].get('n_steps', 20),
                device=device
            )
            
            model_results['embedding'] = {
                'mean_smoothness': float(emb_results['mean_smoothness']),
                'mean_drift_curve': emb_results['mean_drift_curve'].tolist()
            }
            
            print(f"    Mean smoothness: {emb_results['mean_smoothness']:.4f}")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        all_results[model_name] = model_results
    
    # Save results
    results_path = results_dir / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate comparison plots
    if len(models) > 1:
        print("\nGenerating comparison plots...")
        
        # Faithfulness curves
        pf_results_for_plot = {}
        for model_name, results in all_results.items():
            if 'integrated_grad_mean' in results:
                pf_results_for_plot[model_name] = {
                    'mean_deletion_curve': np.array(results['integrated_grad_mean']['mean_deletion_curve']),
                    'mean_insertion_curve': np.array(results['integrated_grad_mean']['mean_insertion_curve'])
                }
        
        if pf_results_for_plot:
            plot_faithfulness_curves(
                pf_results_for_plot,
                output_path=str(figures_dir / 'faithfulness_comparison.png')
            )
        
        # Embedding trajectories
        emb_results_for_plot = {}
        for model_name, results in all_results.items():
            if 'embedding' in results:
                emb_results_for_plot[model_name] = {
                    'mean_drift_curve': np.array(results['embedding']['mean_drift_curve']),
                    'mean_smoothness': results['embedding']['mean_smoothness']
                }
        
        if emb_results_for_plot:
            plot_embedding_trajectory(
                emb_results_for_plot,
                output_path=str(figures_dir / 'embedding_comparison.png')
            )
    
    print("\nExperiment completed!")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run CLXAI evaluation experiments')
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml')
    parser.add_argument('--n_samples', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.n_samples:
        config['pixel_flipping']['n_samples'] = args.n_samples
        config['embedding_analysis']['n_samples'] = args.n_samples
    
    run_experiments(config)


if __name__ == "__main__":
    main()
