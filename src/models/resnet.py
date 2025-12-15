"""
ResNet-18 implementation for CIFAR-10.
Supports both Cross-Entropy and Contrastive Learning training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder (without classification head).
    Outputs embeddings suitable for contrastive learning.
    """
    
    def __init__(self, embedding_dim: int = 128, in_channels: int = 3):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial conv layer (adapted for CIFAR-10 32x32 input)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
        
        self._initialize_weights()
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before projection head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning projected embeddings."""
        features = self.forward_features(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get embedding (features before projection)."""
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features


class ResNet18(nn.Module):
    """
    ResNet-18 for classification (Cross-Entropy training).
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        
        self.encoder = ResNet18Encoder(embedding_dim=128, in_channels=in_channels)
        self.encoder.projection = nn.Identity()
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        features = self.encoder.forward_features(x)
        logits = self.fc(features)
        return logits
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get embedding (features before classification head)."""
        features = self.encoder.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features
    
    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both logits and embeddings."""
        features = self.encoder.forward_features(x)
        logits = self.fc(features)
        embeddings = F.normalize(features, dim=1)
        return logits, embeddings


def get_resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
    encoder_only: bool = False,
    embedding_dim: int = 128
) -> nn.Module:
    """
    Factory function to create ResNet-18.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        encoder_only: If True, return encoder for contrastive learning
        embedding_dim: Dimension of contrastive embeddings
    
    Returns:
        ResNet-18 model
    """
    if encoder_only:
        return ResNet18Encoder(embedding_dim=embedding_dim)
    else:
        return ResNet18(num_classes=num_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_ce = ResNet18(num_classes=10).to(device)
    x = torch.randn(4, 3, 32, 32).to(device)
    
    logits = model_ce(x)
    print(f"CE model output shape: {logits.shape}")
    
    embeddings = model_ce.get_embedding(x)
    print(f"CE model embedding shape: {embeddings.shape}")
    
    encoder = ResNet18Encoder(embedding_dim=128).to(device)
    emb = encoder(x)
    print(f"Encoder output shape: {emb.shape}")
    
    params_ce = sum(p.numel() for p in model_ce.parameters())
    params_enc = sum(p.numel() for p in encoder.parameters())
    print(f"CE model params: {params_ce:,}")
    print(f"Encoder params: {params_enc:,}")
