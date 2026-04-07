"""
Attention-Based Multiple Instance Learning (ABMIL)

Implements gated attention MIL from:
    Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018
    https://arxiv.org/abs/1802.04712

Also includes:
    - Multi-head attention variant
    - Top-K instance pooling
    - Instance-level scoring for interpretability
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------


class GatedAttention(nn.Module):
    """
    Gated attention mechanism (Ilse et al. 2018, Eq. 5).

    a_k = softmax( w^T (tanh(V h_k) ⊙ sigmoid(U h_k)) )

    The gating prevents uninformative instances from contributing via
    near-zero sigmoid outputs even when tanh is near ±1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Instance features [N, input_dim]

        Returns:
            attention_scores: Raw (pre-softmax) scores [N, 1]
        """
        h = self.dropout(h)
        A_V = self.attention_V(h)   # [N, hidden_dim]
        A_U = self.attention_U(h)   # [N, hidden_dim]
        A = self.attention_weights(A_V * A_U)  # [N, 1]
        return A


class StandardAttention(nn.Module):
    """
    Standard (non-gated) attention mechanism (Ilse et al. 2018, Eq. 4).

    a_k = softmax( w^T tanh(V h_k) )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Instance features [N, input_dim]

        Returns:
            attention_scores: Raw (pre-softmax) scores [N, 1]
        """
        return self.attention(h)


class MultiHeadAttention(nn.Module):
    """
    Multi-head gated attention for capturing diverse instance relationships.

    Each head independently computes attention weights, producing
    multi_heads × 1 attention distributions. Final aggregation
    uses the mean across heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            GatedAttention(input_dim, hidden_dim, dropout)
            for _ in range(num_heads)
        ])

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Instance features [N, input_dim]

        Returns:
            attention_probs: Mean attention probabilities [N, 1]
            head_attentions: Per-head attention probabilities [num_heads, N, 1]
        """
        head_scores = torch.stack(
            [head(h) for head in self.heads], dim=0
        )  # [num_heads, N, 1]
        head_attentions = torch.softmax(head_scores, dim=1)  # [num_heads, N, 1]
        attention_probs = head_attentions.mean(dim=0)  # [N, 1]
        return attention_probs, head_attentions


# ---------------------------------------------------------------------------
# Main ABMIL Model
# ---------------------------------------------------------------------------


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning for WSI classification.

    Supports:
        - Gated and standard attention
        - Multi-head attention
        - Top-K pooling as an alternative aggregation
        - Instance-level scoring for heatmap generation

    Reference:
        Ilse M, Tomczak JM, Welling M. Attention-based Deep Multiple
        Instance Learning. ICML 2018. https://arxiv.org/abs/1802.04712
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 1,
        gated: bool = True,
        dropout: float = 0.25,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Args:
            input_dim:      Dimension of input tile features
            hidden_dim:     Hidden dimension after feature encoder
            attention_dim:  Attention bottleneck dimension
            num_classes:    Number of output classes
            num_heads:      Number of attention heads (1 = standard ABMIL)
            gated:          Use gated attention (recommended)
            dropout:        Dropout probability
            top_k:          If set, use Top-K pooling instead of attention
        """
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k

        # --- Feature encoder (compresses input features) ---
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Attention module ---
        if top_k is None:
            if num_heads > 1:
                self.attention = MultiHeadAttention(
                    hidden_dim, attention_dim, num_heads, dropout
                )
                self.multi_head = True
            else:
                if gated:
                    self.attention = GatedAttention(hidden_dim, attention_dim, dropout)
                else:
                    self.attention = StandardAttention(hidden_dim, attention_dim, dropout)
                self.multi_head = False
        else:
            self.attention = None
            self.multi_head = False

        # --- Classifier head ---
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        h: torch.Tensor,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            h: Bag of tile features [N, input_dim]
            return_attention: Whether to return attention scores

        Returns:
            logits: Slide-level class logits [num_classes]
            attention_scores: Attention probabilities [N, 1] or None
        """
        # Encode features
        h = self.feature_encoder(h)  # [N, hidden_dim]

        if self.top_k is not None:
            # Top-K pooling: aggregate top-K most informative tiles
            slide_emb, attention_scores = self._topk_pooling(h)
        elif self.multi_head:
            # Multi-head attention
            attention_probs, head_attentions = self.attention(h)  # [N, 1]
            slide_emb = (attention_probs * h).sum(dim=0)  # [hidden_dim]
            attention_scores = attention_probs
        else:
            # Standard gated/non-gated attention
            A = self.attention(h)          # [N, 1]
            A = A.transpose(0, 1)         # [1, N]
            A = F.softmax(A, dim=1)       # [1, N]
            slide_emb = torch.mm(A, h)    # [1, hidden_dim]
            slide_emb = slide_emb.squeeze(0)  # [hidden_dim]
            attention_scores = A.transpose(0, 1)  # [N, 1]

        logits = self.classifier(slide_emb)  # [num_classes]

        if return_attention:
            return logits, attention_scores
        return logits, None

    def _topk_pooling(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K pooling: select K tiles with highest L2 norm as proxy for
        informativeness, then average-pool them.

        Args:
            h: Encoded features [N, hidden_dim]

        Returns:
            slide_emb: Aggregated slide embedding [hidden_dim]
            attention_scores: Binary mask [N, 1] with 1s at top-K positions
        """
        k = min(self.top_k, h.shape[0])
        norms = torch.norm(h, dim=1)  # [N]
        topk_indices = torch.topk(norms, k).indices  # [k]

        # Create attention mask
        attention_scores = torch.zeros(h.shape[0], 1, device=h.device)
        attention_scores[topk_indices] = 1.0 / k

        slide_emb = h[topk_indices].mean(dim=0)  # [hidden_dim]
        return slide_emb, attention_scores

    def get_attention_scores(self, h: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get raw attention scores for heatmap generation.

        Args:
            h: Tile features [N, input_dim]

        Returns:
            attention: Normalized attention probabilities [N]
        """
        with torch.no_grad():
            _, attention = self.forward(h, return_attention=True)
        return attention.squeeze(1)  # [N]

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities for a slide.

        Args:
            h: Tile features [N, input_dim]

        Returns:
            probs: Class probabilities [num_classes]
        """
        with torch.no_grad():
            logits, _ = self.forward(h)
            probs = F.softmax(logits, dim=0)
        return probs


# ---------------------------------------------------------------------------
# Multi-class ABMIL with per-class attention
# ---------------------------------------------------------------------------


class MultiClassABMIL(nn.Module):
    """
    Multi-class ABMIL with separate attention branch per class.

    Each class has its own attention mechanism, allowing the model to
    focus on different tiles for different classification decisions.
    Useful for multi-label pathology tasks (e.g., predicting multiple
    biomarkers from a single WSI).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # One attention module per class
        self.attention_modules = nn.ModuleList([
            GatedAttention(hidden_dim, attention_dim, dropout)
            for _ in range(num_classes)
        ])

        # One classifier per class (binary output)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])

    def forward(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Tile features [N, input_dim]

        Returns:
            logits: Per-class logits [num_classes]
            attention_matrix: Per-class attention [num_classes, N]
        """
        h_enc = self.feature_encoder(h)  # [N, hidden_dim]

        all_logits = []
        all_attentions = []

        for c in range(self.num_classes):
            A = self.attention_modules[c](h_enc)  # [N, 1]
            A = F.softmax(A, dim=0)               # [N, 1]
            slide_emb = (A * h_enc).sum(dim=0)    # [hidden_dim]
            logit = self.classifiers[c](slide_emb)  # [1]
            all_logits.append(logit)
            all_attentions.append(A.squeeze(1))  # [N]

        logits = torch.cat(all_logits)                        # [num_classes]
        attention_matrix = torch.stack(all_attentions, dim=0) # [num_classes, N]

        return logits, attention_matrix


# ---------------------------------------------------------------------------
# Attention regularization loss
# ---------------------------------------------------------------------------


def attention_entropy_loss(attention_scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Entropy regularization on attention scores to prevent degenerate
    distributions (all attention on one tile).

    Maximizes entropy → more spread-out attention.
    L = -sum(a_i * log(a_i))

    Args:
        attention_scores: Normalized attention probabilities [N]
        eps: Small constant for numerical stability

    Returns:
        neg_entropy: Negative entropy (minimize this to maximize entropy)
    """
    entropy = -(attention_scores * torch.log(attention_scores + eps)).sum()
    return -entropy  # Minimize negative entropy = maximize entropy


def attention_sparsity_loss(attention_scores: torch.Tensor, target_k: int = 20) -> torch.Tensor:
    """
    L1-based sparsity regularization to encourage the model to focus
    on a small number of informative tiles.

    Args:
        attention_scores: Attention probabilities [N]
        target_k: Target number of attended tiles

    Returns:
        sparsity_loss: L1 norm of attention beyond top-K
    """
    n = attention_scores.shape[0]
    k = min(target_k, n)

    sorted_scores = torch.sort(attention_scores, descending=True).values
    # Penalize non-top-K attention
    tail_scores = sorted_scores[k:]
    return tail_scores.sum()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_abmil(config: dict) -> ABMIL:
    """
    Build ABMIL from configuration dictionary.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        ABMIL model
    """
    return ABMIL(
        input_dim=config.get("input_dim", 1024),
        hidden_dim=config.get("hidden_dim", 512),
        attention_dim=config.get("attention_dim", 256),
        num_classes=config.get("num_classes", 2),
        num_heads=config.get("num_heads", 1),
        gated=config.get("gated", True),
        dropout=config.get("dropout", 0.25),
        top_k=config.get("top_k", None),
    )
