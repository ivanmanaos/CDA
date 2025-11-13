import torch
import torch.nn as nn
import torchvision.models as models

# Define a custom augmentation head
class AugmentHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.augment_head = nn.Linear(dim, 12)  # Output size is 12

    def forward(self, x):
        return self.augment_head(x)

# Load EfficientNet model and modify its classifier
def get_model(device, pretrained: bool = True, image_size: int | None = None):
    """
    Create EfficientNet-B7 backbone with 12-dim head.

    Parameters
    - device: torch.device or str
    - pretrained: whether to try loading pretrained ImageNet weights (default True). If offline or fails, falls back to random init.
    - image_size: expected input size for transforms (informational). EfficientNet-B7 commonly uses 600; for CPU a smaller size like 384 is practical.

    Returns
    - torch.nn.Module moved to the specified device
    """
    try:
        # Torchvision >= 0.13 uses enums for weights
        weights_enum = getattr(models, 'EfficientNet_B7_Weights', None)
        if pretrained and weights_enum is not None:
            model = models.efficientnet_b7(weights=weights_enum.DEFAULT)
        else:
            # Fallback for older API accepting string or when pretrained=False
            model = models.efficientnet_b7(weights=('DEFAULT' if pretrained else None))
    except Exception:
        # Offline or other failures -> no pretrained weights
        model = models.efficientnet_b7(weights=None)
    model.classifier[1] = AugmentHead(dim=model.classifier[1].in_features)
    return model.to(device)

# Function to calculate VHS from model outputs
def calc_vhs(x: torch.Tensor):
    A = x[..., 0:2]
    B = x[..., 2:4]
    C = x[..., 4:6]
    D = x[..., 6:8]
    E = x[..., 8:10]
    F = x[..., 10:12]
    
    AB = torch.norm(A - B, p=2, dim=-1)
    CD = torch.norm(C - D, p=2, dim=-1)
    EF = torch.norm(E - F, p=2, dim=-1)
    
    vhs = 6 * (AB + CD) / EF
    
    return vhs
