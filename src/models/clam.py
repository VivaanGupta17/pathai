"""
CLAM: Clustering-Constrained Attention Multiple Instance Learning

Implements CLAM-SB (single branch) and CLAM-MB (multi-branch) from:
    Lu et al., "Data-efficient and weakly supervised computational pathology
    on whole-slide images", Nature Biomedical Engineering, 2021.
    https://www.nature.com/articles/s41551-020-00682-w

Original implementation: https://github.com/mahmoodlab/CLAM

Key innovation: Instance-level clustering pseudo-supervision through
SVM-based instance discrimination loss, enabling better instance-level
feature learning without tile-level annotations.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SVM Instance Loss
# ---------------------------------------------------------------------------


def svm_instance_loss(
    instance_features: torch.Tensor,
    instance_labels: torch.Tensor,
    smooth_svm: bool = True,
) -> torch.Tensor:
    """
    Smooth SVM (hinge) loss for instance-level pseudo-supervision.

    For positive bags: top-K instances → class label 1
    For negative bags: all instances → class label 0

    L_inst = max(0, 1 - y * s) or smooth variant: log(1 + exp(-y * s))

    Args:
        instance_features: Instance logits [K, 1] from instance classifier
        instance_labels:   Pseudo-labels {0, 1} [K]
        smooth_svm:        Use smooth SVM (cross-entropy-like) vs. hard hinge

    Returns:
        Instance loss scalar
    """
    instance_features = instance_features.squeeze(1)
    labels = instance_labels.float()

    if smooth_svm:
        # Smooth SVM: equivalent to cross-entropy with {-1, +1} labels
        # y ∈ {0, 1} → map to {-1, +1}
        y = 2 * labels - 1
        loss = F.softplus(-y * instance_features).mean()
    else:
        # Hard hinge loss
        y = 2 * labels - 1
        loss = torch.clamp(1 - y * instance_features, min=0).mean()

    return loss


# ---------------------------------------------------------------------------
# CLAM Attention Module
# ---------------------------------------------------------------------------


class CLAMAttention(nn.Module):
    """
    Gated attention module used inside CLAM (identical structure to ABMIL
    gated attention, reproduced here for CLAM-specific weight initialization
    and size configuration).

    Supports three size presets from the original paper:
        'small': 1024 → 512 (attention: 256)
        'big':   1024 → 512 (attention: 384)
    """

    SIZE_DICT = {
        "small": [1024, 512, 256],
        "big":   [1024, 512, 384],
    }

    def __init__(
        self,
        size_arg: str = "small",
        dropout: bool = True,
        gate: bool = True,
    ) -> None:
        super().__init__()
        size = self.SIZE_DICT[size_arg]

        self.fc1 = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
        )

        if gate:
            self.attention_V = nn.Sequential(
                nn.Linear(size[1], size[2]),
                nn.Tanh(),
            )
            self.attention_U = nn.Sequential(
                nn.Linear(size[1], size[2]),
                nn.Sigmoid(),
            )
            self.attention_weights = nn.Linear(size[2], 1)
        else:
            self.attention_V = nn.Sequential(
                nn.Linear(size[1], size[2]),
                nn.Tanh(),
                nn.Linear(size[2], 1),
            )
            self.attention_U = None
            self.attention_weights = None

        self.gate = gate
        self.feature_dim = size[1]

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Raw tile features [N, input_dim]

        Returns:
            h: Encoded features [N, feature_dim]
            A: Raw attention scores (pre-softmax) [N, 1]
        """
        h = self.fc1(x)

        if self.gate:
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_weights(A_V * A_U)  # [N, 1]
        else:
            A = self.attention_V(h)  # [N, 1]

        return h, A


# ---------------------------------------------------------------------------
# Instance Classifier
# ---------------------------------------------------------------------------


class InstanceClassifier(nn.Module):
    """
    Lightweight linear classifier for instance-level pseudo-supervision.

    Applied to the top-K (for positive bags) or random-K (for negative bags)
    instances selected by attention score.
    """

    def __init__(self, feature_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        # Two separate binary classifiers (one per class)
        self.classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(num_classes)
        ])

    def forward(
        self,
        h: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            h:          Instance features [K, feature_dim]
            class_idx:  Which class classifier to use

        Returns:
            logits: [K, 1]
        """
        return self.classifiers[class_idx](h)


# ---------------------------------------------------------------------------
# CLAM-SB: Single Branch
# ---------------------------------------------------------------------------


class CLAM_SB(nn.Module):
    """
    CLAM Single-Branch (CLAM-SB).

    Uses a single attention branch for all classes.
    The instance-level clustering loss provides auxiliary supervision.

    Architecture:
        x → fc → h
        h → attention_V × attention_U → A (scores)
        A → softmax → α
        z = Σ αᵢ hᵢ  (bag embedding)
        z → classifier → logits

    Instance loss:
        For positive bags: top-K instances → treated as class-positive
        For negative bags: top-K instances → treated as class-negative
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: bool = True,
        k_sample: int = 8,
        num_classes: int = 2,
        instance_loss_fn: str = "svm",
        subtyping: bool = False,
    ) -> None:
        """
        Args:
            gate:              Use gated attention
            size_arg:          'small' or 'big' (controls hidden dims)
            dropout:           Apply dropout in feature encoder
            k_sample:          Number of instances for clustering pseudo-labels
            num_classes:       Number of output classes
            instance_loss_fn:  'svm' or 'ce'
            subtyping:         If True, positive/negative framing changes for
                               multi-class subtyping tasks
        """
        super().__init__()
        self.k_sample = k_sample
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.instance_loss_fn = instance_loss_fn

        # Shared attention module
        self.attention_net = CLAMAttention(size_arg, dropout, gate)

        # Instance classifiers (one binary per class)
        feature_dim = self.attention_net.feature_dim
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(num_classes)
        ])

        # Bag-level classifier
        self.classifiers = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:               Bag of tile features [N, input_dim]
            label:           Ground-truth slide label (needed for instance loss)
            instance_eval:   Whether to compute instance-level loss
            return_features: Whether to return bag embedding
            attention_only:  If True, return only attention scores

        Returns:
            logits:        Slide-level logits [num_classes]
            instance_dict: Dict with 'instance_loss', 'inst_labels', 'inst_preds'
            A:             Attention probabilities [N, 1]
        """
        h, A = self.attention_net(x)  # [N, feat_dim], [N, 1]

        if attention_only:
            return F.softmax(A.transpose(0, 1), dim=1).transpose(0, 1)

        # Softmax normalization
        A_raw = A.transpose(0, 1)      # [1, N]
        A_norm = F.softmax(A_raw, dim=1)  # [1, N]

        # Bag embedding
        M = torch.mm(A_norm, h)  # [1, feature_dim]
        logits = self.classifiers(M)  # [1, num_classes]
        logits = logits.squeeze(0)    # [num_classes]

        instance_dict = {}

        if instance_eval and label is not None:
            instance_loss, inst_labels, inst_preds = self._compute_instance_loss(
                h, A_norm, label
            )
            instance_dict = {
                "instance_loss": instance_loss,
                "inst_labels": inst_labels,
                "inst_preds": inst_preds,
            }

        A_out = A_norm.transpose(0, 1)  # [N, 1]

        if return_features:
            instance_dict["features"] = M.squeeze(0)

        return logits, instance_dict, A_out

    def _compute_instance_loss(
        self,
        h: torch.Tensor,
        A_norm: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute instance-level clustering loss.

        Strategy (non-subtyping binary):
        - Positive bag (label=1): top-K instances → class 1, bottom-K → class 0
        - Negative bag (label=0): top-K instances → class 0

        Args:
            h:      Encoded features [N, feature_dim]
            A_norm: Attention probabilities [1, N]
            label:  Slide label scalar

        Returns:
            total_loss: Aggregated instance loss
            all_labels: Pseudo-instance labels
            all_preds:  Instance predictions
        """
        k = min(self.k_sample, h.shape[0] // 2)
        k = max(k, 1)

        total_loss = torch.zeros(1, device=h.device, dtype=h.dtype)
        all_labels = []
        all_preds = []

        slide_label = label.item() if torch.is_tensor(label) else int(label)

        for class_idx in range(self.num_classes):
            classifier = self.instance_classifiers[class_idx]

            if not self.subtyping:
                # Binary: positive or negative bag
                if slide_label == class_idx:
                    # Positive bag: top-K → positive, bottom-K → negative
                    top_idx = torch.topk(A_norm[0], k).indices
                    bot_idx = torch.topk(-A_norm[0], k).indices

                    top_feats = h[top_idx]
                    bot_feats = h[bot_idx]

                    top_logits = classifier(top_feats)  # [k, 1]
                    bot_logits = classifier(bot_feats)  # [k, 1]

                    inst_logits = torch.cat([top_logits, bot_logits], dim=0)
                    inst_labels = torch.cat([
                        torch.ones(k, dtype=torch.long),
                        torch.zeros(k, dtype=torch.long),
                    ]).to(h.device)

                    if self.instance_loss_fn == "svm":
                        loss = svm_instance_loss(inst_logits, inst_labels)
                    else:
                        loss = F.cross_entropy(
                            torch.cat([1 - torch.sigmoid(inst_logits),
                                       torch.sigmoid(inst_logits)], dim=1),
                            inst_labels,
                        )

                    total_loss += loss
                    all_labels.append(inst_labels)
                    all_preds.append((torch.sigmoid(inst_logits) > 0.5).long().squeeze(1))
                else:
                    # Negative bag: top-K → negative
                    top_idx = torch.topk(A_norm[0], k).indices
                    top_feats = h[top_idx]
                    top_logits = classifier(top_feats)

                    inst_labels = torch.zeros(k, dtype=torch.long, device=h.device)
                    if self.instance_loss_fn == "svm":
                        loss = svm_instance_loss(top_logits, inst_labels)
                    else:
                        loss = F.cross_entropy(
                            torch.cat([1 - torch.sigmoid(top_logits),
                                       torch.sigmoid(top_logits)], dim=1),
                            inst_labels,
                        )
                    total_loss += loss
                    all_labels.append(inst_labels)
                    all_preds.append((torch.sigmoid(top_logits) > 0.5).long().squeeze(1))

            else:
                # Subtyping: top-K of current class branch
                top_idx = torch.topk(A_norm[0], k).indices
                top_feats = h[top_idx]
                top_logits = classifier(top_feats)
                inst_labels = (torch.ones(k, dtype=torch.long) * (1 if slide_label == class_idx else 0)).to(h.device)

                if self.instance_loss_fn == "svm":
                    loss = svm_instance_loss(top_logits, inst_labels)
                else:
                    loss = F.cross_entropy(
                        torch.cat([1 - torch.sigmoid(top_logits),
                                   torch.sigmoid(top_logits)], dim=1),
                        inst_labels,
                    )
                total_loss += loss
                all_labels.append(inst_labels)
                all_preds.append((torch.sigmoid(top_logits) > 0.5).long().squeeze(1))

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        return total_loss / self.num_classes, all_labels, all_preds

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention probabilities [N] for heatmap generation."""
        with torch.no_grad():
            _, A = self.attention_net(x)
            A = F.softmax(A.transpose(0, 1), dim=1).squeeze(0)
        return A


# ---------------------------------------------------------------------------
# CLAM-MB: Multi-Branch
# ---------------------------------------------------------------------------


class CLAM_MB(nn.Module):
    """
    CLAM Multi-Branch (CLAM-MB).

    Uses a separate attention branch per class, allowing each branch
    to focus on class-discriminative regions. The final bag embedding
    for class c uses attention computed by branch c.

    Architecture:
        x → fc → h
        For each class c:
            h → attention_c → α_c
            z_c = Σ α_c,i h_i
            logit_c = classifier_c(z_c)
    """

    SIZE_DICT = {
        "small": [1024, 512, 256],
        "big":   [1024, 512, 384],
    }

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: bool = True,
        k_sample: int = 8,
        num_classes: int = 2,
        instance_loss_fn: str = "svm",
        subtyping: bool = False,
    ) -> None:
        super().__init__()
        self.k_sample = k_sample
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.instance_loss_fn = instance_loss_fn

        size = self.SIZE_DICT[size_arg]

        # Shared feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
        )

        # Per-class attention modules
        if gate:
            self.attention_nets = nn.ModuleList([
                nn.ModuleDict({
                    "attention_V": nn.Sequential(nn.Linear(size[1], size[2]), nn.Tanh()),
                    "attention_U": nn.Sequential(nn.Linear(size[1], size[2]), nn.Sigmoid()),
                    "attention_w": nn.Linear(size[2], 1),
                })
                for _ in range(num_classes)
            ])
        else:
            self.attention_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(size[1], size[2]),
                    nn.Tanh(),
                    nn.Linear(size[2], 1),
                )
                for _ in range(num_classes)
            ])

        self.gate = gate
        self.feature_dim = size[1]

        # Per-class instance classifiers
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(size[1], 1) for _ in range(num_classes)
        ])

        # Per-class bag classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(size[1], 1) for _ in range(num_classes)
        ])

    def _attention(self, h: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Compute attention scores for a specific class branch."""
        if self.gate:
            net = self.attention_nets[class_idx]
            A_V = net["attention_V"](h)
            A_U = net["attention_U"](h)
            A = net["attention_w"](A_V * A_U)
        else:
            A = self.attention_nets[class_idx](h)
        return A  # [N, 1]

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        instance_eval: bool = False,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:               Bag of tile features [N, input_dim]
            label:           Ground-truth slide label
            instance_eval:   Whether to compute instance-level loss
            return_features: Whether to include bag embeddings in output

        Returns:
            logits:        Slide-level logits [num_classes]
            instance_dict: Dict with instance supervision info
            A_all:         Stacked attention [num_classes, N]
        """
        h = self.feature_encoder(x)  # [N, feature_dim]

        all_logits = []
        all_attentions = []
        total_instance_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        all_inst_labels = []
        all_inst_preds = []

        for class_idx in range(self.num_classes):
            A = self._attention(h, class_idx)     # [N, 1]
            A_t = A.transpose(0, 1)               # [1, N]
            A_norm = F.softmax(A_t, dim=1)        # [1, N]

            M = torch.mm(A_norm, h)               # [1, feature_dim]
            logit = self.classifiers[class_idx](M)  # [1, 1]
            all_logits.append(logit.squeeze())
            all_attentions.append(A_norm.squeeze(0))  # [N]

            if instance_eval and label is not None:
                k = min(self.k_sample, h.shape[0] // 2)
                k = max(k, 1)
                slide_label = label.item() if torch.is_tensor(label) else int(label)

                top_idx = torch.topk(A_norm[0], k).indices
                top_feats = h[top_idx]
                top_logits = self.instance_classifiers[class_idx](top_feats)

                inst_labels = torch.full((k,), int(slide_label == class_idx),
                                         dtype=torch.long, device=h.device)
                if self.instance_loss_fn == "svm":
                    inst_loss = svm_instance_loss(top_logits, inst_labels)
                else:
                    inst_loss = F.cross_entropy(
                        torch.cat([1 - torch.sigmoid(top_logits),
                                   torch.sigmoid(top_logits)], dim=1),
                        inst_labels,
                    )
                total_instance_loss += inst_loss
                all_inst_labels.append(inst_labels)
                all_inst_preds.append((torch.sigmoid(top_logits) > 0.5).long().squeeze(1))

        logits = torch.stack(all_logits)    # [num_classes]
        A_all = torch.stack(all_attentions, dim=0)  # [num_classes, N]

        instance_dict = {}
        if instance_eval and label is not None:
            instance_dict = {
                "instance_loss": total_instance_loss / self.num_classes,
                "inst_labels": torch.cat(all_inst_labels),
                "inst_preds": torch.cat(all_inst_preds),
            }

        return logits, instance_dict, A_all

    def get_attention(self, x: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        """Return attention probabilities for a specific class branch."""
        with torch.no_grad():
            h = self.feature_encoder(x)
            A = self._attention(h, class_idx)
            A = F.softmax(A.transpose(0, 1), dim=1).squeeze(0)
        return A


# ---------------------------------------------------------------------------
# Combined Loss for CLAM Training
# ---------------------------------------------------------------------------


class CLAMLoss(nn.Module):
    """
    Combined loss for CLAM training:
        L = L_bag + λ * L_instance

    where L_bag is cross-entropy on bag-level prediction and
    L_instance is the clustering-based instance loss.
    """

    def __init__(
        self,
        bag_weight: float = 0.7,
        instance_weight: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.bag_weight = bag_weight
        self.instance_weight = instance_weight
        self.bag_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        instance_dict: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            logits:        [num_classes] or [batch, num_classes]
            label:         Ground-truth [1] or [batch]
            instance_dict: Contains 'instance_loss' from CLAM forward

        Returns:
            total_loss: Combined scalar loss
            loss_dict:  Breakdown of individual losses
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)

        bag_loss = self.bag_loss(logits, label)
        instance_loss = instance_dict.get("instance_loss", torch.zeros(1, device=logits.device))

        total_loss = (
            self.bag_weight * bag_loss
            + self.instance_weight * instance_loss.squeeze()
        )

        return total_loss, {
            "bag_loss": bag_loss.item(),
            "instance_loss": instance_loss.item() if torch.is_tensor(instance_loss) else instance_loss,
            "total_loss": total_loss.item(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_clam(config: dict) -> nn.Module:
    """
    Build CLAM model from configuration dictionary.

    Args:
        config: Dictionary with 'model_type' and hyperparameters

    Returns:
        CLAM_SB or CLAM_MB model
    """
    model_type = config.get("model_type", "clam_sb").lower()
    kwargs = {
        "gate": config.get("gate", True),
        "size_arg": config.get("size_arg", "small"),
        "dropout": config.get("dropout", True),
        "k_sample": config.get("k_sample", 8),
        "num_classes": config.get("num_classes", 2),
        "instance_loss_fn": config.get("instance_loss_fn", "svm"),
        "subtyping": config.get("subtyping", False),
    }

    if model_type == "clam_sb":
        return CLAM_SB(**kwargs)
    elif model_type == "clam_mb":
        return CLAM_MB(**kwargs)
    else:
        raise ValueError(f"Unknown CLAM model type: {model_type}. Choose 'clam_sb' or 'clam_mb'.")
