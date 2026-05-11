"""
ResNet-50 and Swin V2-T backbones with binary classification heads.

From paper (Section 2.6.2):
    - ImageNet-pretrained weights
    - Original classification heads replaced with 2-logit FC layer
    - All other weights initialised from ImageNet

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
    ResNet-50: He et al., CVPR 2016
    Swin V2-T: Liu et al., CVPR 2022
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# --------------------------------------------------------------------------- #
#  Binary classification head                                                  #
# --------------------------------------------------------------------------- #

class BinaryHead(nn.Module):
    """2-logit fully connected head for benign/malignant classification."""
    def __init__(self, in_features, dropout=0.0):
        super().__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, 2))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


# --------------------------------------------------------------------------- #
#  ResNet-50                                                                   #
# --------------------------------------------------------------------------- #

def build_resnet50(pretrained=True, dropout=0.0):
    """
    ImageNet-pretrained ResNet-50 with 2-logit binary classification head.
    (Section 2.6.2 — original head replaced with 2-logit FC)
    """
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = tv_models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc    = BinaryHead(in_features, dropout)

    return model


# --------------------------------------------------------------------------- #
#  Swin Transformer V2-T                                                       #
# --------------------------------------------------------------------------- #

def build_swin_v2_t(pretrained=True, dropout=0.0):
    """
    ImageNet-pretrained Swin Transformer V2-T with 2-logit binary head.
    (Section 2.6.2)
    """
    weights = tv_models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tv_models.swin_v2_t(weights=weights)

    in_features   = model.head.in_features
    model.head    = BinaryHead(in_features, dropout)

    return model


# --------------------------------------------------------------------------- #
#  Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_backbone(name="resnet50", pretrained=True, dropout=0.0):
    """
    Build model by name string.

    Args:
        name: 'resnet50' | 'swin_v2_t'
    """
    if name == "resnet50":
        return build_resnet50(pretrained, dropout)
    elif name in ("swin_v2_t", "swin"):
        return build_swin_v2_t(pretrained, dropout)
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose resnet50 or swin_v2_t")


# --------------------------------------------------------------------------- #
#  FedBN: separate BN parameters for client-specific statistics               #
# --------------------------------------------------------------------------- #

def get_non_bn_params(model):
    """
    Return parameters excluding Batch Normalization layers.
    Used by FedBN aggregation — only non-BN parameters are federated.
    BN statistics remain client-specific (Section 2.4, FedBN).
    """
    non_bn_params = {}
    for name, param in model.state_dict().items():
        if "bn" not in name.lower() and "batch_norm" not in name.lower() \
                and "norm" not in name.lower():
            non_bn_params[name] = param
    return non_bn_params


def get_bn_params(model):
    """Return only BN parameters (kept client-specific in FedBN)."""
    bn_params = {}
    for name, param in model.state_dict().items():
        if "bn" in name.lower() or "batch_norm" in name.lower() \
                or "norm" in name.lower():
            bn_params[name] = param
    return bn_params


if __name__ == "__main__":
    for name in ["resnet50", "swin_v2_t"]:
        model = build_backbone(name, pretrained=False)
        x     = torch.randn(2, 3, 224, 224)
        out   = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output={out.shape}  params={n_params:,}")
