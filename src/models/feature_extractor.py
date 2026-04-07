"""
Tile Feature Extraction for Computational Pathology

Supports multiple feature extractors:
    1. ResNet50 (ImageNet-pretrained) — baseline
    2. CTransPath — pathology-specific ViT/CNN hybrid, Wang et al. 2022
       https://github.com/Xiyue-Wang/TransPath
    3. UNI — pathology foundation model, Chen et al. 2024 (Mahmood Lab)
       https://github.com/mahmoodlab/UNI
    4. CONCH — vision-language pathology model, Lu et al. 2024 (Mahmood Lab)
       https://github.com/mahmoodlab/CONCH
    5. PLIP — pathology language-image pre-training, Huang et al. 2023
       https://github.com/PathologyFoundation/plip

Feature caching: Extract once → store as .pt files → reuse across MIL runs.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard Transforms
# ---------------------------------------------------------------------------


def get_imagenet_transform(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_pathology_transform(image_size: int = 224) -> transforms.Compose:
    """
    Transform for pathology-pretrained models (CTransPath, UNI, CONCH).
    Uses pathology-specific normalization statistics from large H&E datasets.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.70322989, 0.53606487, 0.66096631],
            std=[0.21716536, 0.26081574, 0.20723464],
        ),
    ])


# ---------------------------------------------------------------------------
# ResNet50 Feature Extractor
# ---------------------------------------------------------------------------


class ResNet50Extractor(nn.Module):
    """
    ImageNet-pretrained ResNet50 feature extractor.

    Removes the final classification head and outputs 1024-dim features
    from the penultimate layer (after global average pooling of layer3+4).

    Features: 2048-dim from global avg pool, optionally projected to 1024.
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1024,
        weights: str = "IMAGENET1K_V2",
    ) -> None:
        super().__init__()

        if pretrained:
            try:
                backbone = models.resnet50(weights=weights)
            except Exception:
                backbone = models.resnet50(pretrained=True)
        else:
            backbone = models.resnet50(pretrained=False)

        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Optional projection to target feature_dim
        in_dim = 2048
        if feature_dim != in_dim:
            self.projector = nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()

        self.feature_dim = feature_dim
        self.transform = get_imagenet_transform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of tiles [B, 3, H, W]

        Returns:
            features: [B, feature_dim]
        """
        h = self.features(x)       # [B, 2048, 1, 1]
        h = h.flatten(1)           # [B, 2048]
        h = self.projector(h)      # [B, feature_dim]
        return h


# ---------------------------------------------------------------------------
# CTransPath Feature Extractor
# ---------------------------------------------------------------------------


class CTransPathExtractor(nn.Module):
    """
    CTransPath: Pathology-specific feature extractor combining CNN local
    features with Transformer global context.

    Architecture: Swin Transformer modified with CNN local feature fusion.
    Pretrained on 15,197,067 pathology image patches from TCGA and PAIP.

    Paper: Wang et al., "Transformer-based Unsupervised Contrastive Learning
    for Histopathological Image Classification", Medical Image Analysis, 2022.

    Weights: https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX

    NOTE: Requires ctranspath package or manual weight loading.
    This implementation provides a wrapper that loads the weights if available,
    or falls back to a standard Swin-T as proxy.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.transform = get_pathology_transform()

        try:
            # Try loading ctranspath (requires the package)
            self._load_ctranspath(weights_path)
            self._loaded = True
            logger.info("Loaded CTransPath weights from %s", weights_path)
        except Exception as e:
            logger.warning(
                "Could not load CTransPath weights (%s). "
                "Using Swin-T proxy. For full CTransPath, install: "
                "pip install ctranspath and download weights from "
                "https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX",
                str(e),
            )
            self._load_swin_proxy()
            self._loaded = False

    def _load_ctranspath(self, weights_path: Optional[str]) -> None:
        """Load official CTransPath weights."""
        try:
            from ctran import ctranspath
            self.model = ctranspath()
            self.model.head = nn.Identity()
            if weights_path and os.path.exists(weights_path):
                td = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(td["model"], strict=True)
        except ImportError:
            raise ImportError("ctranspath package not installed")

    def _load_swin_proxy(self) -> None:
        """Swin Transformer proxy (same architecture, ImageNet weights)."""
        try:
            from torchvision.models import swin_t, Swin_T_Weights
            backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            backbone.head = nn.Identity()
            self.model = backbone
            self.feature_dim = 768
        except Exception:
            # Fallback to ResNet
            logger.warning("Swin-T unavailable, using ResNet50 proxy for CTransPath")
            backbone = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            features: [B, feature_dim]
        """
        h = self.model(x)
        if h.dim() > 2:
            h = h.flatten(1)
        return h


# ---------------------------------------------------------------------------
# UNI Feature Extractor
# ---------------------------------------------------------------------------


class UNIExtractor(nn.Module):
    """
    UNI: Universal pathology foundation model.

    DINOv2 ViT-L/16 pretrained on 100,000+ WSIs from Mass General Brigham
    (MGB) pathology archive — the largest pathology pretraining dataset.

    Paper: Chen et al., "Towards a General-Purpose Foundation Model for
    Computational Pathology", Nature Medicine, 2024.

    Model HuggingFace: MahmoodLab/UNI (requires gated access)
    GitHub: https://github.com/mahmoodlab/UNI

    Feature dimension: 1024 (ViT-L)
    Input size: 224×224 px

    NOTE: Requires HuggingFace authentication and approved access.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_hf: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = 1024
        self.transform = get_pathology_transform(image_size=224)

        loaded = False

        if weights_path and os.path.exists(weights_path):
            try:
                self._load_from_path(weights_path)
                loaded = True
                logger.info("Loaded UNI weights from %s", weights_path)
            except Exception as e:
                logger.warning("Failed to load UNI weights from path: %s", e)

        if not loaded and use_hf:
            try:
                self._load_from_hf()
                loaded = True
                logger.info("Loaded UNI from HuggingFace MahmoodLab/UNI")
            except Exception as e:
                logger.warning(
                    "Could not load UNI from HuggingFace (%s). "
                    "Using ViT-L/16 ImageNet proxy. "
                    "For UNI, request access at: https://huggingface.co/MahmoodLab/UNI",
                    str(e),
                )
                self._load_vit_proxy()

        if not loaded:
            self._load_vit_proxy()

    def _load_from_path(self, path: str) -> None:
        """Load UNI from local checkpoint."""
        import timm
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=True)

    def _load_from_hf(self) -> None:
        """Load UNI from HuggingFace Hub (requires gated access)."""
        import timm
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.model.head = nn.Identity()

    def _load_vit_proxy(self) -> None:
        """ViT-L/16 ImageNet proxy when UNI weights unavailable."""
        try:
            import timm
            self.model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=0)
            logger.info("Using timm ViT-L/16 (ImageNet) as UNI proxy")
        except ImportError:
            logger.warning("timm not installed; using ResNet50 as UNI proxy")
            backbone = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)
        if h.dim() > 2:
            h = h.flatten(1)
        return h


# ---------------------------------------------------------------------------
# CONCH Feature Extractor
# ---------------------------------------------------------------------------


class CONCHExtractor(nn.Module):
    """
    CONCH: Contrastive Learning from Captions for Histopathology.

    Vision-language model trained with caption-based contrastive learning
    on pathology images paired with diagnostic captions from pathology reports.

    Paper: Lu et al., "A Visual-Language Foundation Model for Computational
    Pathology", Nature Medicine, 2024.

    GitHub: https://github.com/mahmoodlab/CONCH
    HuggingFace: MahmoodLab/CONCH (requires gated access)

    Feature dimension: 512 (CLIP-style vision encoder)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = 512
        self.transform = get_pathology_transform(image_size=224)

        try:
            self._load_conch(weights_path)
            logger.info("Loaded CONCH model")
        except Exception as e:
            logger.warning(
                "CONCH unavailable (%s). Using CLIP ViT-B/32 proxy. "
                "For CONCH: https://github.com/mahmoodlab/CONCH",
                str(e),
            )
            self._load_clip_proxy()

    def _load_conch(self, weights_path: Optional[str]) -> None:
        """Load CONCH model."""
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            self.model, _ = create_model_from_pretrained(
                "conch_ViT-B-16",
                weights_path or "hf_hub:MahmoodLab/CONCH",
            )
            self.model.visual.output_tokens = True
        except ImportError:
            raise ImportError("conch package not installed")

    def _load_clip_proxy(self) -> None:
        """CLIP ViT-B/32 as proxy."""
        try:
            import clip
            model, _ = clip.load("ViT-B/32", device="cpu")
            self.model = model.visual
        except ImportError:
            logger.warning("clip not installed; using ResNet50 as CONCH proxy")
            backbone = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.model(x)
        if h.dim() > 2:
            h = h.flatten(1)
        return h


# ---------------------------------------------------------------------------
# Feature Extractor Registry
# ---------------------------------------------------------------------------


EXTRACTOR_REGISTRY = {
    "resnet50": ResNet50Extractor,
    "ctranspath": CTransPathExtractor,
    "uni": UNIExtractor,
    "conch": CONCHExtractor,
}


def build_feature_extractor(
    extractor_name: str,
    weights_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Build and return a feature extractor by name.

    Args:
        extractor_name: One of 'resnet50', 'ctranspath', 'uni', 'conch'
        weights_path:   Path to pretrained weights (if needed)
        device:         Compute device

    Returns:
        Feature extractor module in eval mode
    """
    name = extractor_name.lower()
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown extractor '{name}'. "
            f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
        )

    cls = EXTRACTOR_REGISTRY[name]

    if name in ("ctranspath", "uni", "conch"):
        extractor = cls(weights_path=weights_path)
    else:
        extractor = cls()

    extractor = extractor.to(device)
    extractor.eval()
    return extractor


# ---------------------------------------------------------------------------
# Tile Dataset for Batched Feature Extraction
# ---------------------------------------------------------------------------


class TileDataset(Dataset):
    """
    Dataset of pre-extracted tile images for batched feature extraction.
    Supports both PIL Image lists and paths to image files.
    """

    def __init__(
        self,
        tiles: Union[List[Image.Image], List[str]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.tiles = tiles
        self.transform = transform or get_imagenet_transform()
        self._is_paths = isinstance(tiles[0], (str, Path)) if tiles else False

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tile = self.tiles[idx]
        if self._is_paths:
            tile = Image.open(tile).convert("RGB")
        elif not isinstance(tile, Image.Image):
            tile = Image.fromarray(tile)
        return self.transform(tile)


# ---------------------------------------------------------------------------
# Main Feature Extractor Class (batched, cached)
# ---------------------------------------------------------------------------


class FeatureExtractorPipeline:
    """
    High-level pipeline for extracting and caching tile features.

    Workflow:
        1. Check cache: if slide_id.pt exists, load and return
        2. Create TileDataset from tiles
        3. Extract features in batches
        4. Save to cache
        5. Return [N, feature_dim] tensor

    Cache format: {cache_dir}/{slide_id}_{extractor_name}.pt
        Contains: {"features": Tensor[N, D], "coords": Tensor[N, 2]}
    """

    def __init__(
        self,
        extractor: nn.Module,
        extractor_name: str,
        batch_size: int = 256,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.extractor = extractor
        self.extractor_name = extractor_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, slide_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{slide_id}_{self.extractor_name}.pt"

    def extract(
        self,
        tiles: List[Image.Image],
        coords: Optional[np.ndarray] = None,
        slide_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features from a list of tiles.

        Args:
            tiles:    List of PIL Images
            coords:   Tile coordinates [N, 2] (x, y)
            slide_id: Slide identifier for caching

        Returns:
            features: [N, feature_dim] float32 tensor
            coords_t: [N, 2] tensor or None
        """
        # Check cache
        cache_path = self._cache_path(slide_id) if slide_id else None
        if cache_path and cache_path.exists():
            logger.info("Loading cached features: %s", cache_path)
            cached = torch.load(cache_path, map_location="cpu")
            return cached["features"], cached.get("coords")

        # Extract features
        transform = getattr(self.extractor, "transform", get_imagenet_transform())
        dataset = TileDataset(tiles, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_features = []
        self.extractor.eval()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                features = self.extractor(batch)
                all_features.append(features.cpu())

        features_tensor = torch.cat(all_features, dim=0)  # [N, D]
        coords_tensor = torch.from_numpy(coords).float() if coords is not None else None

        # Save to cache
        if cache_path:
            save_dict = {"features": features_tensor}
            if coords_tensor is not None:
                save_dict["coords"] = coords_tensor
            torch.save(save_dict, cache_path)
            logger.info("Cached features to %s (shape: %s)", cache_path, features_tensor.shape)

        return features_tensor, coords_tensor

    def extract_from_wsi(
        self,
        tile_generator,  # Generator yielding (tile, coord) tuples
        slide_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features directly from a WSI tile generator.

        Args:
            tile_generator: Iterable of (PIL.Image, (x, y)) tuples
            slide_id:       Slide identifier

        Returns:
            features: [N, feature_dim]
            coords:   [N, 2]
        """
        cache_path = self._cache_path(slide_id)
        if cache_path and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            return cached["features"], cached.get("coords", torch.zeros(0, 2))

        tiles = []
        all_coords = []
        for tile, coord in tile_generator:
            tiles.append(tile)
            all_coords.append(coord)

        coords_array = np.array(all_coords, dtype=np.float32)
        return self.extract(tiles, coords=coords_array, slide_id=slide_id)


def compute_cache_id(wsi_path: str, extractor_name: str, magnification: float) -> str:
    """Generate a deterministic cache ID for a WSI + extractor combination."""
    key = f"{wsi_path}_{extractor_name}_{magnification}"
    return hashlib.md5(key.encode()).hexdigest()[:16]
