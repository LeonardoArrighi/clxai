"""
Supervised Contrastive Learning training for ResNet-18 on CIFAR-10.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.resnet import ResNet18Encoder
from src.models.classifiers import KNNClassifier, LinearClassifier, train_linear_classifier
from src.training.losses import SupConLossV2
from src.utils.data import get_cifar10_loaders

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_epoch_scl(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch with contrastive loss."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        # images is a list of two augmented views
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings for both views
        embeddings = model(images)  # (2B, D)
        
        # Compute contrastive loss
        loss = criterion(embeddings, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'train_loss': total_loss / len(train_loader)}


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    data_loader,
    device: torch.device
) -> tuple:
    """Extract embeddings and labels from data loader."""
    model.eval()
    embeddings = []
    labels = []
    
    for images, targets in tqdm(data_loader, desc="Extracting embeddings"):
        # Handle both regular and contrastive loaders
        if isinstance(images, list):
            images = images[0]
        images = images.to(device)
        
        # Get features (before projection head)
        features = model.get_embedding(images, normalize=True)
        embeddings.append(features.cpu())
        labels.append(targets)
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return embeddings, labels


def evaluate_knn(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    k: int = 10
) -> float:
    """Evaluate model using k-NN classifier on embeddings."""
    # Extract embeddings
    train_emb, train_labels = extract_embeddings(model, train_loader, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    
    # Train k-NN
    knn = KNNClassifier(k=k, metric='cosine')
    knn.fit(train_emb, train_labels)
    
    # Evaluate
    accuracy = knn.score(test_emb, test_labels) * 100
    return accuracy


def train_scl_model(config: dict):
    """
    Main training function for SCL model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(config.get('output_dir', 'results/models/scl'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.init(
            project=config.get('wandb_project', 'clxai'),
            name=config.get('run_name', 'scl_supcon'),
            config=config
        )
    
    # Data loaders - contrastive mode
    train_loader_cl, test_loader = get_cifar10_loaders(
        data_dir=config.get('data_dir', './data'),
        batch_size=config.get('batch_size', 256),
        num_workers=config.get('num_workers', 4),
        contrastive=True
    )
    
    # Regular loader for k-NN evaluation
    train_loader_eval, _ = get_cifar10_loaders(
        data_dir=config.get('data_dir', './data'),
        batch_size=config.get('batch_size', 256),
        num_workers=config.get('num_workers', 4),
        contrastive=False,
        augment=False
    )
    
    # Model
    model = ResNet18Encoder(
        embedding_dim=config.get('embedding_dim', 128)
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = SupConLossV2(temperature=config.get('temperature', 0.07))
    
    # Optimizer - SupCon paper uses LARS, we use SGD for simplicity
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.5),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Scheduler
    epochs = config.get('epochs', 500)
    warmup_epochs = config.get('warmup_epochs', 10)
    
    # Warmup + Cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train with contrastive loss
        train_metrics = train_epoch_scl(
            model, train_loader_cl, criterion, optimizer, device, epoch
        )
        
        scheduler.step()
        
        # Evaluate with k-NN periodically
        if epoch % config.get('eval_freq', 10) == 0 or epoch == epochs:
            knn_acc = evaluate_knn(
                model, train_loader_eval, test_loader, device,
                k=config.get('knn_k', 10)
            )
            train_metrics['knn_acc'] = knn_acc
            
            print(f"Epoch {epoch}: loss={train_metrics['train_loss']:.4f}, "
                  f"kNN_acc={knn_acc:.2f}%")
            
            # Save best model
            if knn_acc > best_acc:
                best_acc = knn_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'knn_acc': best_acc,
                    'config': config
                }, output_dir / 'best_model.pt')
        else:
            print(f"Epoch {epoch}: loss={train_metrics['train_loss']:.4f}")
        
        train_metrics['lr'] = scheduler.get_last_lr()[0]
        
        if WANDB_AVAILABLE and config.get('use_wandb', True):
            wandb.log(train_metrics, step=epoch)
        
        # Save checkpoint periodically
        if epoch % config.get('save_freq', 100) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Final save
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, output_dir / 'final_model.pt')
    
    # Train and save linear classifier
    print("\nTraining linear classifier on frozen embeddings...")
    train_emb, train_labels = extract_embeddings(model, train_loader_eval, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    
    linear_clf = LinearClassifier(input_dim=512, num_classes=10)
    history = train_linear_classifier(
        linear_clf,
        torch.tensor(train_emb, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(test_emb, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
        epochs=100,
        device=str(device)
    )
    linear_acc = history['val_acc'][-1] * 100
    print(f"Linear probe accuracy: {linear_acc:.2f}%")
    
    torch.save(linear_clf.state_dict(), output_dir / 'linear_classifier.pt')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best kNN accuracy: {best_acc:.2f}%")
    print(f"Linear probe accuracy: {linear_acc:.2f}%")
    
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.log({'final_knn_acc': best_acc, 'linear_acc': linear_acc})
        wandb.finish()
    
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train SCL ResNet-18 on CIFAR-10')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='results/models/scl')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='scl_supcon')
    args = parser.parse_args()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'temperature': args.temperature,
            'embedding_dim': 128,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'use_wandb': not args.no_wandb,
            'run_name': args.run_name,
            'wandb_project': 'clxai',
            'num_workers': 4,
            'save_freq': 100,
            'eval_freq': 10,
            'warmup_epochs': 10,
            'knn_k': 10
        }
    
    train_scl_model(config)


if __name__ == "__main__":
    main()
