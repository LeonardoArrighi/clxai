"""
Classification methods for SCL embeddings.
Implements k-NN, Linear Probe, and Mahalanobis distance classifiers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional, Dict


class KNNClassifier:
    """k-Nearest Neighbors classifier for SCL embeddings."""
    
    def __init__(self, k: int = 10, metric: str = 'cosine'):
        self.k = k
        self.metric = metric
        self.knn = None
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        self.knn = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.metric,
            n_jobs=-1
        )
        self.knn.fit(embeddings, labels)
        self.is_fitted = True
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        return self.knn.predict(embeddings)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        return self.knn.predict_proba(embeddings)
    
    def score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(embeddings)
        return (predictions == labels).mean()


class LinearClassifier(nn.Module):
    """Linear probe classifier for SCL embeddings."""
    
    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(embeddings)
            return logits.argmax(dim=1)
    
    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(embeddings)
            return F.softmax(logits, dim=1)


def train_linear_classifier(
    classifier: LinearClassifier,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: str = 'cuda'
) -> Dict[str, list]:
    """Train linear classifier on frozen embeddings."""
    classifier = classifier.to(device)
    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)
    
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        classifier.train()
        logits = classifier(train_embeddings)
        loss = criterion(logits, train_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc = (logits.argmax(dim=1) == train_labels).float().mean().item()
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        
        if val_embeddings is not None:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_embeddings.to(device))
                val_acc = (val_logits.argmax(dim=1) == val_labels.to(device)).float().mean().item()
                history['val_acc'].append(val_acc)
    
    return history


class MahalanobisClassifier:
    """Mahalanobis distance-based classifier for SCL embeddings."""
    
    def __init__(self, num_classes: int = 10, regularization: float = 1e-6):
        self.num_classes = num_classes
        self.reg = regularization
        self.centroids = None
        self.precision = None
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        dim = embeddings.shape[1]
        
        self.centroids = np.zeros((self.num_classes, dim))
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.centroids[c] = embeddings[mask].mean(axis=0)
        
        centered = embeddings.copy()
        for c in range(self.num_classes):
            mask = labels == c
            centered[mask] -= self.centroids[c]
        
        cov = np.cov(centered.T)
        cov += self.reg * np.eye(dim)
        
        try:
            self.precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.precision = np.linalg.pinv(cov)
        
        self.is_fitted = True
    
    def _mahalanobis_distance(self, x: np.ndarray, centroid: np.ndarray) -> float:
        diff = x - centroid
        return np.sqrt(diff @ self.precision @ diff).item()
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        
        predictions = []
        for x in embeddings:
            distances = [
                self._mahalanobis_distance(x, self.centroids[c])
                for c in range(self.num_classes)
            ]
            predictions.append(np.argmin(distances))
        
        return np.array(predictions)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        
        probas = []
        for x in embeddings:
            distances = np.array([
                self._mahalanobis_distance(x, self.centroids[c])
                for c in range(self.num_classes)
            ])
            neg_distances = -distances
            exp_neg = np.exp(neg_distances - neg_distances.max())
            proba = exp_neg / exp_neg.sum()
            probas.append(proba)
        
        return np.array(probas)
    
    def score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(embeddings)
        return (predictions == labels).mean()
    
    def get_distances(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        
        all_distances = []
        for x in embeddings:
            distances = [
                self._mahalanobis_distance(x, self.centroids[c])
                for c in range(self.num_classes)
            ]
            all_distances.append(distances)
        
        return np.array(all_distances)


if __name__ == "__main__":
    np.random.seed(42)
    
    n_samples = 1000
    n_classes = 10
    dim = 512
    
    embeddings = []
    labels = []
    for c in range(n_classes):
        center = np.random.randn(dim)
        samples = center + 0.1 * np.random.randn(n_samples // n_classes, dim)
        embeddings.append(samples)
        labels.extend([c] * (n_samples // n_classes))
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    idx = np.random.permutation(len(labels))
    embeddings = embeddings[idx]
    labels = labels[idx]
    
    train_emb, test_emb = embeddings[:800], embeddings[800:]
    train_labels, test_labels = labels[:800], labels[800:]
    
    knn = KNNClassifier(k=5, metric='cosine')
    knn.fit(train_emb, train_labels)
    knn_acc = knn.score(test_emb, test_labels)
    print(f"kNN accuracy: {knn_acc:.4f}")
    
    maha = MahalanobisClassifier(num_classes=n_classes)
    maha.fit(train_emb, train_labels)
    maha_acc = maha.score(test_emb, test_labels)
    print(f"Mahalanobis accuracy: {maha_acc:.4f}")
