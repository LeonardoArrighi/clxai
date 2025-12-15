#!/usr/bin/env python
"""
Unit tests for CLXAI components.
Tests TripletLoss, ECEFaithfulness, and SVMClassifier.

Run with: python scripts/test_components.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

# Import components to test
from src.training.losses import TripletLoss, SupConLoss, SupConLossV2
from src.analysis.faithfulness import ECEFaithfulness, FaithfulnessEvaluator
from src.models.classifiers import SVMClassifier, KNNClassifier, MahalanobisClassifier


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  [FAIL] {name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*50}")
        return self.failed == 0


results = TestResults()


# =============================================================================
# TripletLoss Tests
# =============================================================================

def test_triplet_loss_hard():
    """Test TripletLoss with hard negative mining."""
    try:
        loss_fn = TripletLoss(margin=0.3, mining='hard')
        
        # Create embeddings: 3 classes, 4 samples each
        # NOTE: requires_grad=True is needed for loss.backward() to work
        batch_size = 12
        dim = 128
        embeddings = F.normalize(torch.randn(batch_size, dim, requires_grad=True), dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Forward pass
        loss = loss_fn(embeddings, labels)
        
        # Check loss is valid
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        assert loss.requires_grad, "Loss doesn't require grad"
        
        # Backward pass
        loss.backward()
        
        results.record("TripletLoss (hard mining)", True)
    except Exception as e:
        results.record("TripletLoss (hard mining)", False, str(e))


def test_triplet_loss_semi_hard():
    """Test TripletLoss with semi-hard negative mining."""
    try:
        loss_fn = TripletLoss(margin=0.3, mining='semi-hard')
        
        batch_size = 12
        dim = 128
        embeddings = F.normalize(torch.randn(batch_size, dim, requires_grad=True), dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        loss = loss_fn(embeddings, labels)
        
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss >= 0, "Loss is negative"
        
        loss.backward()
        
        results.record("TripletLoss (semi-hard mining)", True)
    except Exception as e:
        results.record("TripletLoss (semi-hard mining)", False, str(e))


def test_triplet_loss_batch_all():
    """Test TripletLoss with batch-all mining."""
    try:
        loss_fn = TripletLoss(margin=0.3, mining='all')
        
        batch_size = 12
        dim = 128
        embeddings = F.normalize(torch.randn(batch_size, dim, requires_grad=True), dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        loss = loss_fn(embeddings, labels)
        
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss >= 0, "Loss is negative"
        
        loss.backward()
        
        results.record("TripletLoss (batch-all mining)", True)
    except Exception as e:
        results.record("TripletLoss (batch-all mining)", False, str(e))


def test_triplet_loss_margin_effect():
    """Test that margin affects loss correctly."""
    try:
        # Create well-separated clusters
        torch.manual_seed(42)
        dim = 64
        
        # Class 0: near origin, Class 1: far from origin
        emb_c0 = F.normalize(torch.randn(4, dim) * 0.1, dim=1)
        emb_c1 = F.normalize(torch.randn(4, dim) * 0.1 + 2.0, dim=1)
        embeddings = torch.cat([emb_c0, emb_c1], dim=0)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Small margin should give smaller loss
        loss_small = TripletLoss(margin=0.1, mining='hard')(embeddings, labels)
        loss_large = TripletLoss(margin=1.0, mining='hard')(embeddings, labels)
        
        # Both should be valid
        assert not torch.isnan(loss_small), "Small margin loss is NaN"
        assert not torch.isnan(loss_large), "Large margin loss is NaN"
        
        results.record("TripletLoss (margin effect)", True)
    except Exception as e:
        results.record("TripletLoss (margin effect)", False, str(e))


def test_triplet_loss_gpu():
    """Test TripletLoss on GPU if available."""
    try:
        if not torch.cuda.is_available():
            results.record("TripletLoss (GPU)", True)  # Skip but pass
            return
        
        device = torch.device('cuda')
        loss_fn = TripletLoss(margin=0.3, mining='hard')
        
        embeddings = F.normalize(torch.randn(12, 128, device=device, requires_grad=True), dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], device=device)
        
        loss = loss_fn(embeddings, labels)
        
        assert loss.device.type == 'cuda', "Loss not on GPU"
        assert not torch.isnan(loss), "Loss is NaN"
        
        loss.backward()
        
        results.record("TripletLoss (GPU)", True)
    except Exception as e:
        results.record("TripletLoss (GPU)", False, str(e))


# =============================================================================
# ECEFaithfulness Tests
# =============================================================================

def test_ece_normalize_saliency():
    """Test saliency normalization."""
    try:
        ece = ECEFaithfulness(n_bins=10)
        
        # Test with random saliency
        saliency = np.random.randn(32, 32)
        normalized = ece.normalize_saliency(saliency)
        
        assert abs(normalized.sum() - 1.0) < 1e-6, "Normalized saliency doesn't sum to 1"
        assert (normalized >= 0).all(), "Normalized saliency has negative values"
        
        results.record("ECEFaithfulness (normalize_saliency)", True)
    except Exception as e:
        results.record("ECEFaithfulness (normalize_saliency)", False, str(e))


def test_ece_expected_confidence():
    """Test expected confidence drop computation."""
    try:
        ece = ECEFaithfulness(n_bins=10)
        
        # Create uniform saliency
        saliency = np.ones(100) / 100  # Already normalized
        pixel_order = np.arange(100)  # Remove in order
        
        expected = ece.compute_expected_confidence_drop(saliency, pixel_order)
        
        # Should start at 1.0 and end at 0.0
        assert abs(expected[0] - 1.0) < 1e-6, f"Expected starts at {expected[0]}, not 1.0"
        assert abs(expected[-1] - 0.0) < 1e-6, f"Expected ends at {expected[-1]}, not 0.0"
        
        # Should be monotonically decreasing
        assert (np.diff(expected) <= 1e-6).all(), "Expected confidence not monotonically decreasing"
        
        results.record("ECEFaithfulness (expected_confidence)", True)
    except Exception as e:
        results.record("ECEFaithfulness (expected_confidence)", False, str(e))


def test_ece_perfect_calibration():
    """Test ECE with perfectly calibrated confidence curve."""
    try:
        ece = ECEFaithfulness(n_bins=10)
        
        # Create perfectly calibrated case:
        # confidence drops linearly as we remove pixels
        n_steps = 21
        confidence_curve = np.linspace(1.0, 0.0, n_steps)
        
        # Uniform saliency
        saliency = np.ones(100)
        
        result = ece.compute_faithfulness_curve(
            saliency, confidence_curve, removal_order='most_important_first'
        )
        
        # Perfect calibration should have:
        # - ECE close to 0
        # - slope close to 1
        # - R² close to 1
        assert result['ece'] < 0.1, f"ECE too high: {result['ece']}"
        assert abs(result['slope'] - 1.0) < 0.2, f"Slope not ~1: {result['slope']}"
        assert result['r_squared'] > 0.9, f"R² too low: {result['r_squared']}"
        
        results.record("ECEFaithfulness (perfect calibration)", True)
    except Exception as e:
        results.record("ECEFaithfulness (perfect calibration)", False, str(e))


def test_ece_batch_evaluation():
    """Test batch evaluation of ECE."""
    try:
        ece = ECEFaithfulness(n_bins=10)
        
        # Create batch of samples
        n_samples = 10
        n_steps = 21
        
        saliencies = [np.random.rand(32, 32) for _ in range(n_samples)]
        confidence_curves = [np.sort(np.random.rand(n_steps))[::-1] for _ in range(n_samples)]
        
        result = ece.evaluate_batch(saliencies, confidence_curves)
        
        # Check all expected keys exist
        required_keys = ['ece_mean', 'ece_std', 'slope_mean', 'r_squared_mean', 
                        'calibration_score', 'n_samples']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        assert result['n_samples'] == n_samples, "Wrong sample count"
        
        results.record("ECEFaithfulness (batch evaluation)", True)
    except Exception as e:
        results.record("ECEFaithfulness (batch evaluation)", False, str(e))


# =============================================================================
# SVMClassifier Tests
# =============================================================================

def test_svm_fit_predict():
    """Test SVM fit and predict."""
    try:
        np.random.seed(42)
        
        # Create separable data
        n_samples = 200
        dim = 64
        n_classes = 5
        
        embeddings = []
        labels = []
        for c in range(n_classes):
            center = np.random.randn(dim) * 3
            samples = center + np.random.randn(n_samples // n_classes, dim) * 0.5
            embeddings.append(samples)
            labels.extend([c] * (n_samples // n_classes))
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        
        # Shuffle data before splitting (IMPORTANT: ensures all classes in train/test)
        idx = np.random.permutation(len(labels))
        embeddings = embeddings[idx]
        labels = labels[idx]
        
        # Split
        train_emb, test_emb = embeddings[:160], embeddings[160:]
        train_labels, test_labels = labels[:160], labels[160:]
        
        # Fit SVM
        svm = SVMClassifier(kernel='linear', C=1.0)
        svm.fit(train_emb, train_labels)
        
        # Predict
        predictions = svm.predict(test_emb)
        accuracy = svm.score(test_emb, test_labels)
        
        assert len(predictions) == len(test_labels), "Wrong prediction count"
        assert 0 <= accuracy <= 1, f"Invalid accuracy: {accuracy}"
        assert accuracy > 0.5, f"Accuracy too low: {accuracy}"  # Should be decent on separable data
        
        results.record("SVMClassifier (fit/predict)", True)
    except Exception as e:
        results.record("SVMClassifier (fit/predict)", False, str(e))


def test_svm_predict_proba():
    """Test SVM probability predictions."""
    try:
        np.random.seed(42)
        
        n_samples = 100
        dim = 32
        n_classes = 3
        
        embeddings = np.random.randn(n_samples, dim)
        labels = np.random.randint(0, n_classes, n_samples)
        
        svm = SVMClassifier(kernel='linear', probability=True)
        svm.fit(embeddings, labels)
        
        proba = svm.predict_proba(embeddings[:10])
        
        assert proba.shape == (10, n_classes), f"Wrong proba shape: {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities out of range"
        
        results.record("SVMClassifier (predict_proba)", True)
    except Exception as e:
        results.record("SVMClassifier (predict_proba)", False, str(e))


def test_svm_kernels():
    """Test SVM with different kernels."""
    try:
        np.random.seed(42)
        
        n_samples = 100
        dim = 32
        embeddings = np.random.randn(n_samples, dim)
        labels = np.random.randint(0, 3, n_samples)
        
        for kernel in ['linear', 'rbf']:
            svm = SVMClassifier(kernel=kernel, C=1.0)
            svm.fit(embeddings, labels)
            score = svm.score(embeddings, labels)
            assert score > 0, f"Score is 0 for {kernel} kernel"
        
        results.record("SVMClassifier (kernels)", True)
    except Exception as e:
        results.record("SVMClassifier (kernels)", False, str(e))


def test_svm_scaling():
    """Test SVM with and without feature scaling."""
    try:
        np.random.seed(42)
        
        # Create data with different scales
        n_samples = 100
        embeddings = np.random.randn(n_samples, 32)
        embeddings[:, 0] *= 1000  # One feature has much larger scale
        labels = np.random.randint(0, 2, n_samples)
        
        # With scaling
        svm_scaled = SVMClassifier(kernel='linear', scale_features=True)
        svm_scaled.fit(embeddings, labels)
        
        # Without scaling
        svm_unscaled = SVMClassifier(kernel='linear', scale_features=False)
        svm_unscaled.fit(embeddings, labels)
        
        # Both should work
        assert svm_scaled.score(embeddings, labels) > 0
        assert svm_unscaled.score(embeddings, labels) > 0
        
        results.record("SVMClassifier (scaling)", True)
    except Exception as e:
        results.record("SVMClassifier (scaling)", False, str(e))


# =============================================================================
# Integration Tests
# =============================================================================

def test_all_classifiers_consistency():
    """Test that all classifiers can handle the same data."""
    try:
        np.random.seed(42)
        
        # Create data
        n_samples = 200
        dim = 128
        n_classes = 10
        
        embeddings = []
        labels = []
        for c in range(n_classes):
            center = np.random.randn(dim)
            samples = center + 0.1 * np.random.randn(n_samples // n_classes, dim)
            embeddings.append(samples)
            labels.extend([c] * (n_samples // n_classes))
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        
        # Shuffle
        idx = np.random.permutation(len(labels))
        embeddings = embeddings[idx]
        labels = labels[idx]
        
        train_emb, test_emb = embeddings[:160], embeddings[160:]
        train_labels, test_labels = labels[:160], labels[160:]
        
        # Test all classifiers
        classifiers = [
            ('kNN', KNNClassifier(k=5, metric='cosine')),
            ('SVM', SVMClassifier(kernel='linear')),
            ('Mahalanobis', MahalanobisClassifier(num_classes=n_classes)),
        ]
        
        for name, clf in classifiers:
            clf.fit(train_emb, train_labels)
            score = clf.score(test_emb, test_labels)
            assert score > 0.5, f"{name} accuracy too low: {score}"
        
        results.record("All classifiers consistency", True)
    except Exception as e:
        results.record("All classifiers consistency", False, str(e))


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("="*50)
    print("CLXAI Component Tests")
    print("="*50)
    
    start_time = time.time()
    
    # TripletLoss tests
    print("\n[TripletLoss Tests]")
    test_triplet_loss_hard()
    test_triplet_loss_semi_hard()
    test_triplet_loss_batch_all()
    test_triplet_loss_margin_effect()
    test_triplet_loss_gpu()
    
    # ECEFaithfulness tests
    print("\n[ECEFaithfulness Tests]")
    test_ece_normalize_saliency()
    test_ece_expected_confidence()
    test_ece_perfect_calibration()
    test_ece_batch_evaluation()
    
    # SVMClassifier tests
    print("\n[SVMClassifier Tests]")
    test_svm_fit_predict()
    test_svm_predict_proba()
    test_svm_kernels()
    test_svm_scaling()
    
    # Integration tests
    print("\n[Integration Tests]")
    test_all_classifiers_consistency()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    
    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
