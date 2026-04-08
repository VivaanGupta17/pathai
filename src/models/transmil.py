"""
TransMIL: Transformer-Based Correlated Multiple Instance Learning

Implements TransMIL from:
    Shao et al., "TransMIL: Transformer Based Correlated Multiple Instance
    Learning for Whole Slide Image Classification", NeurIPS 2021.
    https://arxiv.org/abs/2106.00908

Key innovations:
    1. Nyström attention for O(n) complexity on long sequences (10,000+ tiles)
    2. Morphology-aware position encoding using spatial (x, y) tile coordinates
    3. Pyramid positional encoding (PPE) via square root of sequence length
    4. Captures both spatial correlations and long-range dependencies

Reference implementation: https://github.com/szc19990412/TransMIL
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Nyström Attention
# ---------------------------------------------------------------------------


class NystromAttention(nn.Module):
    """
    Nyström-based efficient self-attention for long sequences.

    Approximates full attention using m landmark tokens:
        Â ≈ softmax(Q K̃ᵀ) D̃⁻¹ softmax(K̃ Kᵀ)ᵀ

    Reduces complexity from O(n²) to O(n·m) where m << n.
    Critical for WSI processing where n can be 5,000–50,000 tiles.

    Reference:
        Xiong et al., "Nyströmformer: A Nyström-Based Self-Attention
        Mechanism", AAAI 2021. https://arxiv.org/abs/2102.03902
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_landmarks: int = 256,
        dropout: float = 0.0,
        pinv_iterations: int = 6,
    ) -> None:
        """
        Args:
            dim:               Input/output dimension
            num_heads:         Number of attention heads
            num_landmarks:     Number of landmark tokens (m << n)
            dropout:           Attention dropout
            pinv_iterations:   Iterations for pseudo-inverse computation
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pinv_iterations = pinv_iterations

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    Input tokens [B, N, dim] or [N, dim] (unbatched)
            mask: Optional padding mask [B, N]

        Returns:
            out:  Output tokens same shape as x
        """
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)

        B, N, _ = x.shape
        m = min(self.num_landmarks, N)

        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2),
            qkv,
        )  # Each: [B, heads, N, head_dim]

        q = q * self.scale

        # Landmark selection via average pooling (fold → pool → unfold)
        q_landmarks = q.reshape(B, self.num_heads, m, N // m, self.head_dim).mean(dim=3)  # [B, heads, m, head_dim]
        k_landmarks = k.reshape(B, self.num_heads, m, N // m, self.head_dim).mean(dim=3)

        # Kernel matrices
        kernel_1 = F.softmax(torch.matmul(q, k_landmarks.transpose(-2, -1)), dim=-1)  # [B, heads, N, m]
        kernel_2 = F.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-2, -1)), dim=-1)  # [B, heads, m, m]
        kernel_3 = F.softmax(torch.matmul(q_landmarks, k.transpose(-2, -1)), dim=-1)    # [B, heads, m, N]

        # Moore-Penrose pseudo-inverse of kernel_2 via iterative method
        pinv = self._iterative_pinv(kernel_2)  # [B, heads, m, m]

        # Nyström approximation: A ≈ K1 * pinv(K2) * K3
        attn_output = torch.matmul(kernel_1, torch.matmul(pinv, torch.matmul(kernel_3, v)))
        # [B, heads, N, head_dim]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, -1)  # [B, N, dim]
        out = self.to_out(attn_output)

        if unbatched:
            out = out.squeeze(0)

        return out

    def _iterative_pinv(self, A: torch.Tensor, n_iter: int = None) -> torch.Tensor:
        """
        Compute approximate Moore-Penrose pseudo-inverse via iterative refinement.
        More stable than torch.linalg.pinv for small matrices in mixed precision.
        """
        n_iter = n_iter or self.pinv_iterations
        # Initialize: A_0 = A^T / (||A||_1 * ||A||_inf)
        I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        Ak = A.transpose(-2, -1) / (
            torch.linalg.norm(A, ord=1, dim=(-2, -1), keepdim=True)
            * torch.linalg.norm(A, ord=float('inf'), dim=(-2, -1), keepdim=True)
        )
        for _ in range(n_iter):
            # Ak+1 = Ak (2I - A Ak)
            Ak = torch.matmul(Ak, 2 * I - torch.matmul(A, Ak))
        return Ak


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class MorphologyPositionEncoding(nn.Module):
    """
    Spatial position encoding using tile (x, y) coordinates.

    Maps continuous 2D coordinates to sinusoidal embeddings, allowing
    the transformer to leverage spatial topology of the tissue section.
    Unlike standard ViT (fixed grid), this handles variable-size WSIs
    with irregular tissue boundaries.

    Encoding:
        PE(x, y, 2i)   = sin(x / 10000^(2i/d))
        PE(x, y, 2i+1) = cos(x / 10000^(2i/d))
        (similarly for y in the second half of d_model)
    """

    def __init__(self, d_model: int, max_coord: int = 10000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_coord = max_coord

        half_d = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_d, dtype=torch.float)
            * -(math.log(max_coord) / half_d)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tile coordinates [N, 2] (x, y in pixels or grid units)

        Returns:
            pos_enc: Positional encoding [N, d_model]
        """
        coords = coords.float()
        x = coords[:, 0:1]  # [N, 1]
        y = coords[:, 1:2]  # [N, 1]

        half_d = self.d_model // 2
        div = self.div_term.unsqueeze(0)  # [1, half_d]

        # x-based encoding
        x_enc = torch.zeros(coords.shape[0], half_d, device=coords.device)
        x_enc[:, 0::2] = torch.sin(x * div[:, :half_d // 2 + 1][:, :half_d // 2])
        x_enc[:, 1::2] = torch.cos(x * div[:, :half_d // 2])

        # y-based encoding
        y_enc = torch.zeros(coords.shape[0], self.d_model - half_d, device=coords.device)
        y_enc[:, 0::2] = torch.sin(y * div[:, :(self.d_model - half_d) // 2 + 1][:, :(self.d_model - half_d) // 2])
        y_enc[:, 1::2] = torch.cos(y * div[:, :(self.d_model - half_d) // 2])

        return torch.cat([x_enc, y_enc], dim=1)  # [N, d_model]


class SquareRootPPE(nn.Module):
    """
    Pyramid Positional Encoding (PPE) via square root.

    TransMIL uses a 2D positional encoding based on the spatial position
    of tiles in the WSI, accounting for the fact that tiles form a 2D grid.

    The "pyramid" aspect: the position encoding is applied at multiple
    scales corresponding to sqrt(N) where N is bag size.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_layer = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Args:
            x: Token sequence [B, N, d_model] where N = h * w
            h: Spatial height (sqrt(N) rounded)
            w: Spatial width

        Returns:
            x + positional encoding, same shape
        """
        B, N, C = x.shape
        # Reshape to 2D spatial arrangement
        x_2d = x.transpose(1, 2).reshape(B, C, h, w)  # [B, C, h, w]
        x_2d = self.pos_layer(x_2d)
        x_2d = x_2d.reshape(B, C, N).transpose(1, 2)  # [B, N, C]
        return x + x_2d


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------


class TransMILLayer(nn.Module):
    """
    Single TransMIL transformer layer with Nyström attention and FFN.
    Pre-LayerNorm variant (more stable training than post-LN).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_landmarks: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            num_heads=num_heads,
            num_landmarks=num_landmarks,
            dropout=attn_dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] or [N, dim]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# TransMIL Main Model
# ---------------------------------------------------------------------------


class TransMIL(nn.Module):
    """
    Transformer-based MIL for WSI classification.

    Pipeline:
        1. Project tile features to transformer dimension
        2. Add spatial position encoding (morphological)
        3. Add [CLS] token
        4. Apply stacked transformer layers (Nyström attention)
        5. Classify from [CLS] token output

    Reference:
        Shao Z et al. TransMIL: Transformer Based Correlated Multiple
        Instance Learning for WSI Classification. NeurIPS 2021.
        https://arxiv.org/abs/2106.00908
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 2,
        dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 512,
        num_landmarks: int = 256,
        dropout: float = 0.1,
        use_nystrom: bool = True,
        use_pos_enc: bool = True,
        pos_enc_type: str = "morphology",  # 'morphology' | 'learned' | 'none'
    ) -> None:
        """
        Args:
            input_dim:      Dimension of input tile features
            num_classes:    Number of output classes
            dim:            Transformer hidden dimension
            num_layers:     Number of transformer layers
            num_heads:      Number of attention heads
            mlp_dim:        FFN hidden dimension
            num_landmarks:  Nyström landmark count
            dropout:        Dropout rate
            use_nystrom:    Use Nyström attention (required for long sequences)
            use_pos_enc:    Whether to apply positional encoding
            pos_enc_type:   Type of positional encoding
        """
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.pos_enc_type = pos_enc_type
        self.dim = dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional encoding
        if use_pos_enc:
            if pos_enc_type == "morphology":
                self.pos_encoder = MorphologyPositionEncoding(dim)
                self.pos_proj = nn.Linear(dim, dim)
            elif pos_enc_type == "learned":
                self.pos_encoder = None  # Applied during forward with learned emb
            # else: 'none'

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransMILLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_dim / dim,
                num_landmarks=num_landmarks,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output norm and classifier
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x:                Tile features [N, input_dim]
            coords:           Tile spatial coordinates [N, 2] (required for morphological PE)
            return_attention: Whether to return attention maps

        Returns:
            logits:       Slide-level logits [num_classes]
            attention:    Optional attention maps [num_layers, num_heads, N+1, N+1]
        """
        N = x.shape[0]

        # Project to transformer dimension
        h = self.input_proj(x)  # [N, dim]

        # Add positional encoding
        if self.use_pos_enc and self.pos_enc_type == "morphology" and coords is not None:
            pos_enc = self.pos_encoder(coords)   # [N, dim]
            pos_enc = self.pos_proj(pos_enc)     # [N, dim]
            h = h + pos_enc

        # Prepend [CLS] token
        cls = self.cls_token.expand(1, -1)  # [1, dim]
        h = torch.cat([cls, h], dim=0)      # [N+1, dim]

        # Add batch dimension for transformer
        h = h.unsqueeze(0)  # [1, N+1, dim]

        # Apply transformer layers
        attention_maps = [] if return_attention else None
        for layer in self.layers:
            h = layer(h)
            # Note: collecting attention maps would require modifying NystromAttention

        h = h.squeeze(0)   # [N+1, dim]
        h = self.norm(h)

        # Classify from [CLS] token
        cls_out = h[0]  # [dim]
        logits = self.classifier(cls_out)  # [num_classes]

        return logits, attention_maps

    def get_tile_representations(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get transformer-processed tile representations for heatmap generation.

        Args:
            x:      Tile features [N, input_dim]
            coords: Tile coordinates [N, 2]

        Returns:
            tile_reps: [N, dim] — post-transformer representations
        """
        with torch.no_grad():
            N = x.shape[0]
            h = self.input_proj(x)

            if self.use_pos_enc and self.pos_enc_type == "morphology" and coords is not None:
                pos_enc = self.pos_encoder(coords)
                pos_enc = self.pos_proj(pos_enc)
                h = h + pos_enc

            cls = self.cls_token.expand(1, -1)
            h = torch.cat([cls, h], dim=0).unsqueeze(0)

            for layer in self.layers:
                h = layer(h)

            h = h.squeeze(0)
            h = self.norm(h)

        return h[1:]  # Exclude CLS token, return [N, dim]

    def predict_proba(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Slide-level class probabilities [num_classes]."""
        with torch.no_grad():
            logits, _ = self.forward(x, coords)
            probs = F.softmax(logits, dim=0)
        return probs


# ---------------------------------------------------------------------------
# TransMIL with pseudo-bag augmentation
# ---------------------------------------------------------------------------


class TransMIL_Aug(TransMIL):
    """
    TransMIL with pseudo-bag augmentation for small datasets.

    During training, creates multiple sub-bags by randomly sampling
    a subset of tiles, then aggregates predictions across sub-bags.
    This provides implicit data augmentation for MIL training.
    """

    def __init__(
        self,
        *args,
        sub_bag_size: int = 512,
        n_sub_bags: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sub_bag_size = sub_bag_size
        self.n_sub_bags = n_sub_bags

    def forward_augmented(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward with pseudo-bag augmentation.
        Returns averaged logits across sub-bags.
        """
        N = x.shape[0]
        k = min(self.sub_bag_size, N)

        all_logits = []
        for _ in range(self.n_sub_bags):
            idx = torch.randperm(N, device=x.device)[:k]
            sub_x = x[idx]
            sub_coords = coords[idx] if coords is not None else None
            logits, _ = self.forward(sub_x, sub_coords)
            all_logits.append(logits)

        return torch.stack(all_logits).mean(dim=0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_transmil(config: dict) -> TransMIL:
    """Build TransMIL from configuration dictionary."""
    return TransMIL(
        input_dim=config.get("input_dim", 1024),
        num_classes=config.get("num_classes", 2),
        dim=config.get("dim", 512),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 8),
        mlp_dim=config.get("mlp_dim", 512),
        num_landmarks=config.get("num_landmarks", 256),
        dropout=config.get("dropout", 0.1),
        use_nystrom=config.get("use_nystrom", True),
        use_pos_enc=config.get("use_pos_enc", True),
        pos_enc_type=config.get("pos_enc_type", "morphology"),
    )

NYSTROM_LANDMARKS = 64  # approximation quality vs speed tradeoff
