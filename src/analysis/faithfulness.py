"""
Faithfulness metrics computation for XAI evaluation.
"""

import numpy as np
from typing import Dict
from scipy import stats


class FaithfulnessEvaluator:
    """Compute and compare faithfulness metrics."""
    
    def compute_auc_deletion(self, curve: np.ndarray) -> float:
        x = np.linspace(0, 1, len(curve))
        return np.trapz(curve, x)
    
    def compute_auc_insertion(self, curve: np.ndarray) -> float:
        x = np.linspace(0, 1, len(curve))
        return np.trapz(curve, x)
    
    def compute_monotonicity(self, curve: np.ndarray, direction: str = 'decreasing') -> float:
        x = np.arange(len(curve))
        corr, _ = stats.spearmanr(x, curve)
        return -corr if direction == 'decreasing' else corr
    
    def evaluate(self, deletion_curves: np.ndarray, insertion_curves: np.ndarray) -> Dict:
        del_aucs = [self.compute_auc_deletion(c) for c in deletion_curves]
        ins_aucs = [self.compute_auc_insertion(c) for c in insertion_curves]
        
        return {
            'deletion_auc_mean': np.mean(del_aucs),
            'deletion_auc_std': np.std(del_aucs),
            'insertion_auc_mean': np.mean(ins_aucs),
            'insertion_auc_std': np.std(ins_aucs),
            'n_samples': len(deletion_curves)
        }


def compute_faithfulness_metrics(del_curves: np.ndarray, ins_curves: np.ndarray) -> Dict:
    return FaithfulnessEvaluator().evaluate(del_curves, ins_curves)


def compute_method_agreement(rankings: Dict[str, np.ndarray]):
    methods = list(rankings.keys())
    n = len(methods)
    tau_matrix = np.eye(n)
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                tau, _ = stats.kendalltau(rankings[m1].flatten(), rankings[m2].flatten())
                tau_matrix[i, j] = tau_matrix[j, i] = tau
    
    return tau_matrix, methods
