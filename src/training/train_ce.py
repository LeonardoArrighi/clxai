"""
Cross-Entropy training for ResNet-18 on CIFAR-10.
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
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.resnet import ResNet18
from src.utils.data import get_cifar10_loaders

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return {
        'train_loss': total_loss / len(train_loader),
        'train_acc': 100. * correct / total
    }


def evaluate(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return {
        'test_loss': total_loss / len(test_loader),
        'test_acc': 100. * correct / total
    }


def train_ce_model(config: dict):
    """
    Main training function for CE model.
    
    Args:
        config: Training configuration dictionary
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config.get('output_dir', 'results/models/ce_baseline'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if available
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.init(
            project=config.get('wandb_project', 'clxai'),
            name=config.get('run_name', 'ce_baseline'),
            config=config
        )
    
    # Data loaders
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=config.get('data_dir', './data'),
        batch_size=config.get('batch_size', 128),
        num_workers=config.get('num_workers', 4),
        augment=True
    )
    
    # Model
    model = ResNet18(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    # Scheduler
    epochs = config.get('epochs', 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {**train_metrics, **test_metrics, 'lr': scheduler.get_last_lr()[0]}
        print(f"Epoch {epoch}: train_acc={train_metrics['train_acc']:.2f}%, "
              f"test_acc={test_metrics['test_acc']:.2f}%")
        
        if WANDB_AVAILABLE and config.get('use_wandb', True):
            wandb.log(metrics, step=epoch)
        
        # Save best model
        if test_metrics['test_acc'] > best_acc:
            best_acc = test_metrics['test_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_acc,
                'config': config
            }, output_dir / 'best_model.pt')
        
        # Save checkpoint periodically
        if epoch % config.get('save_freq', 50) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_metrics['test_acc'],
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Final save
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_metrics['test_acc'],
        'config': config
    }, output_dir / 'final_model.pt')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.finish()
    
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train CE ResNet-18 on CIFAR-10')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='results/models/ce_baseline')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='ce_baseline')
    args = parser.parse_args()
    
    # Load config from file or use defaults
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'use_wandb': not args.no_wandb,
            'run_name': args.run_name,
            'wandb_project': 'clxai',
            'num_workers': 4,
            'save_freq': 50
        }
    
    train_ce_model(config)


if __name__ == "__main__":
    main()
