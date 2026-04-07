"""
Attention Heatmap Generation for Whole Slide Images

Converts tile-level attention scores from MIL models into
spatially coherent heatmaps overlaid on the original WSI.

Workflow:
    1. Extract tile coordinates and attention weights from MIL model
    2. Map tile coordinates → spatial grid
    3. Gaussian smoothing for visual continuity
    4. Normalize to [0, 1] and apply colormap (jet, viridis, etc.)
    5. Blend with original WSI thumbnail
    6. Optionally threshold for tumor region localization

Applications:
    - Model interpretability and quality control
    - Comparison with pathologist annotations
    - Clinical report generation (highlight top tiles)
    - Active learning: identify uncertain regions
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate → Spatial Map
# ---------------------------------------------------------------------------


def coords_to_spatial_map(
    coords: np.ndarray,
    scores: np.ndarray,
    tile_size_l0: int,
    slide_dims_l0: Tuple[int, int],
    map_resolution: Tuple[int, int] = (1000, 1000),
) -> np.ndarray:
    """
    Convert tile coordinates and attention scores to a spatial heatmap.

    Args:
        coords:          [N, 2] tile (x, y) coordinates in level-0 pixels
        scores:          [N] attention scores (any non-negative values)
        tile_size_l0:    Tile size in level-0 pixel space
        slide_dims_l0:   (width, height) of slide at level 0
        map_resolution:  Output heatmap resolution (width, height)

    Returns:
        heatmap: [map_H, map_W] float32 array with attention values,
                 0 where no tile was scored
    """
    slide_w, slide_h = slide_dims_l0
    map_w, map_h = map_resolution

    # Scale factor from level-0 to map space
    scale_x = map_w / slide_w
    scale_y = map_h / slide_h

    heatmap = np.zeros((map_h, map_w), dtype=np.float32)
    count_map = np.zeros((map_h, map_w), dtype=np.float32)

    tile_w_map = max(1, int(tile_size_l0 * scale_x))
    tile_h_map = max(1, int(tile_size_l0 * scale_y))

    for (x_l0, y_l0), score in zip(coords, scores):
        # Map to heatmap coordinates
        x_map = int(x_l0 * scale_x)
        y_map = int(y_l0 * scale_y)

        # Clamp to bounds
        x1 = max(0, x_map)
        y1 = max(0, y_map)
        x2 = min(map_w, x_map + tile_w_map)
        y2 = min(map_h, y_map + tile_h_map)

        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += float(score)
            count_map[y1:y2, x1:x2] += 1.0

    # Average overlapping scores
    valid = count_map > 0
    heatmap[valid] = heatmap[valid] / count_map[valid]

    return heatmap


def apply_gaussian_smoothing(
    heatmap: np.ndarray,
    kernel_size: int = 51,
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to produce visually smooth heatmap.

    Args:
        heatmap:     Raw spatial attention map [H, W]
        kernel_size: Gaussian kernel size (odd number)
        sigma:       Gaussian sigma in pixels

    Returns:
        smoothed: [H, W] smoothed heatmap
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed = cv2.GaussianBlur(
        heatmap.astype(np.float32),
        (kernel_size, kernel_size),
        sigma,
    )
    return smoothed


def normalize_attention_map(
    heatmap: np.ndarray,
    percentile_clip: float = 99.0,
    min_nonzero: bool = True,
) -> np.ndarray:
    """
    Normalize attention map to [0, 1] range.

    Args:
        heatmap:        Raw attention scores [H, W]
        percentile_clip: Clip to this percentile to avoid outlier influence
        min_nonzero:    Set minimum to min of nonzero values (not global min)

    Returns:
        Normalized heatmap in [0, 1]
    """
    valid = heatmap > 0

    if not valid.any():
        return np.zeros_like(heatmap)

    if min_nonzero:
        vmin = float(heatmap[valid].min())
    else:
        vmin = float(heatmap.min())

    vmax = float(np.percentile(heatmap[valid], percentile_clip))

    if vmax <= vmin:
        return np.zeros_like(heatmap)

    normalized = np.clip((heatmap - vmin) / (vmax - vmin), 0.0, 1.0)

    # Zero out non-tissue areas
    normalized[~valid] = 0.0

    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Colormap Application
# ---------------------------------------------------------------------------


COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "coolwarm": cv2.COLORMAP_COOL,
    "rdylgn": cv2.COLORMAP_RdYlGn if hasattr(cv2, "COLORMAP_RdYlGn") else cv2.COLORMAP_JET,
}


def apply_colormap(
    normalized_map: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Apply a colormap to a normalized [0, 1] heatmap.

    Args:
        normalized_map: [H, W] float32 in [0, 1]
        colormap:       Colormap name ('jet', 'viridis', 'plasma', etc.)

    Returns:
        colored: [H, W, 3] uint8 RGB heatmap
    """
    # Convert to uint8 [0, 255]
    uint8_map = (normalized_map * 255).astype(np.uint8)

    # Apply colormap via OpenCV
    cmap_id = COLORMAPS.get(colormap.lower(), cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(uint8_map, cmap_id)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)  # → RGB
    return colored


# ---------------------------------------------------------------------------
# WSI Overlay
# ---------------------------------------------------------------------------


def overlay_heatmap_on_thumbnail(
    thumbnail: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Blend attention heatmap with WSI thumbnail.

    Args:
        thumbnail:  [H, W, 3] uint8 RGB WSI thumbnail
        heatmap:    [H, W] normalized attention map (must match thumbnail dims)
        alpha:      Heatmap opacity (0 = original image, 1 = pure heatmap)
        colormap:   Colormap for attention visualization
        mask:       Optional tissue mask [H, W]; heatmap only shown in tissue

    Returns:
        overlay: [H, W, 3] uint8 blended image
    """
    # Resize heatmap to match thumbnail if needed
    if heatmap.shape[:2] != thumbnail.shape[:2]:
        heatmap = cv2.resize(heatmap, (thumbnail.shape[1], thumbnail.shape[0]))

    colored = apply_colormap(heatmap, colormap)  # [H, W, 3]

    # Apply tissue mask if provided
    if mask is not None:
        mask_resized = cv2.resize(mask.astype(np.uint8), (thumbnail.shape[1], thumbnail.shape[0]))
        tissue = mask_resized > 0
        # Only blend where there's tissue
        overlay = thumbnail.copy()
        overlay[tissue] = (
            (1 - alpha) * thumbnail[tissue] + alpha * colored[tissue]
        ).astype(np.uint8)
    else:
        overlay = (
            (1 - alpha) * thumbnail.astype(np.float32)
            + alpha * colored.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    return overlay


# ---------------------------------------------------------------------------
# Top-K Tile Visualization
# ---------------------------------------------------------------------------


def get_top_k_tiles(
    coords: np.ndarray,
    scores: np.ndarray,
    k: int = 20,
    min_distance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-K tiles by attention score, with optional spatial diversity.

    Args:
        coords:       [N, 2] tile coordinates
        scores:       [N] attention scores
        k:            Number of top tiles to select
        min_distance: Minimum pixel distance between selected tiles
                      (None = no diversity constraint)

    Returns:
        top_coords: [K, 2] coordinates of top tiles
        top_scores: [K] attention scores of top tiles
    """
    k = min(k, len(scores))
    sorted_idx = np.argsort(scores)[::-1]

    if min_distance is None:
        top_idx = sorted_idx[:k]
    else:
        # Greedy selection with spatial diversity
        selected = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if len(selected) >= k:
                break
            # Check distance to all selected tiles
            dists = np.sqrt(
                ((coords[selected] - coords[idx]) ** 2).sum(axis=1)
            )
            if dists.min() >= min_distance:
                selected.append(idx)
        top_idx = np.array(selected)

    return coords[top_idx], scores[top_idx]


def visualize_top_tiles(
    wsi_reader,  # WSIReader instance
    coords: np.ndarray,
    scores: np.ndarray,
    k: int = 12,
    tile_size: int = 224,
    grid_cols: int = 4,
) -> Image.Image:
    """
    Create a grid visualization of top-K attention tiles.

    Args:
        wsi_reader: WSIReader instance for reading tiles
        coords:     [N, 2] tile coordinates
        scores:     [N] attention scores
        k:          Number of top tiles to show
        tile_size:  Tile display size in pixels
        grid_cols:  Number of columns in grid

    Returns:
        Grid image of top-K tiles with attention scores
    """
    top_coords, top_scores = get_top_k_tiles(coords, scores, k=k)
    k_actual = len(top_coords)
    grid_rows = (k_actual + grid_cols - 1) // grid_cols

    # Create grid canvas
    border = 4
    grid_w = grid_cols * (tile_size + border) + border
    grid_h = grid_rows * (tile_size + border + 20) + border  # +20 for score text
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(50, 50, 50))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for i, (coord, score) in enumerate(zip(top_coords, top_scores)):
        row = i // grid_cols
        col = i % grid_cols

        x_offset = border + col * (tile_size + border)
        y_offset = border + row * (tile_size + border + 20)

        try:
            tile = wsi_reader.get_tile_at_coord(int(coord[0]), int(coord[1]), tile_size)
            grid_img.paste(tile, (x_offset, y_offset))

            if draw:
                draw.text(
                    (x_offset + 2, y_offset + tile_size + 2),
                    f"a={score:.3f}",
                    fill=(255, 255, 255),
                    font=font,
                )
        except Exception as e:
            logger.warning("Could not read tile at %s: %s", coord, e)

    return grid_img


# ---------------------------------------------------------------------------
# Main Heatmap Generator
# ---------------------------------------------------------------------------


class HeatmapGenerator:
    """
    High-level interface for generating attention heatmaps from MIL models.

    Usage:
        generator = HeatmapGenerator(model, feature_extractor, device)
        overlay = generator.generate(wsi_path, output_dir)
    """

    def __init__(
        self,
        model,
        feature_extractor,
        device: Union[str, "torch.device"] = "cpu",
        alpha: float = 0.4,
        colormap: str = "jet",
        smooth_sigma: float = 20.0,
        top_k_tiles: int = 20,
        map_resolution: Tuple[int, int] = (1000, 1000),
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.alpha = alpha
        self.colormap = colormap
        self.smooth_sigma = smooth_sigma
        self.top_k_tiles = top_k_tiles
        self.map_resolution = map_resolution

    def generate(
        self,
        wsi_path: Union[str, Path],
        output_dir: Union[str, Path],
        magnification: float = 20.0,
        tile_size: int = 224,
        save_intermediate: bool = True,
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        Generate attention heatmap for a single WSI.

        Args:
            wsi_path:           Path to WSI file
            output_dir:         Directory to save outputs
            magnification:      Tile extraction magnification
            tile_size:          Tile size in pixels
            save_intermediate:  Save intermediate arrays

        Returns:
            Dict with:
                'overlay_path':    Path to saved overlay image
                'attention_map':   Raw attention map [H, W]
                'prediction':      Predicted class probabilities
                'top_tile_paths':  Paths to top-K tile images
        """
        import torch
        import torch.nn.functional as F

        from src.data.wsi_dataset import WSIReader
        from src.data.tile_processing import TileProcessor

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wsi_path = Path(wsi_path)
        slide_id = wsi_path.stem

        logger.info("Generating heatmap for: %s", slide_id)

        # --- Step 1: Extract tiles ---
        reader = WSIReader(wsi_path, target_magnification=magnification, tile_size=tile_size)
        processor = TileProcessor.default(normalize="macenko")

        tiles = []
        coords_list = []

        for tile, coord in reader.iter_tiles():
            processed = processor.process(tile, augment=False)
            if processed is not None:
                tiles.append(processed)
                coords_list.append(coord)

        if not tiles:
            logger.warning("No tissue tiles found in %s", slide_id)
            return {}

        coords_arr = np.array(coords_list, dtype=np.float32)
        logger.info("Extracted %d tiles from %s", len(tiles), slide_id)

        # --- Step 2: Extract features ---
        self.feature_extractor.eval()
        transform = getattr(self.feature_extractor, "transform", None)
        if transform is None:
            from src.models.feature_extractor import get_imagenet_transform
            transform = get_imagenet_transform()

        import torch
        from torch.utils.data import DataLoader
        from src.models.feature_extractor import TileDataset

        dataset = TileDataset(tiles, transform=transform)
        loader = DataLoader(dataset, batch_size=256, num_workers=0)

        features_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                feats = self.feature_extractor(batch)
                features_list.append(feats.cpu())

        features = torch.cat(features_list, dim=0)  # [N, D]
        coords_tensor = torch.from_numpy(coords_arr)

        # --- Step 3: MIL forward for attention scores ---
        self.model.eval()
        features_dev = features.to(self.device)

        with torch.no_grad():
            model_name = type(self.model).__name__.lower()

            if "clam" in model_name:
                _, _, attention = self.model(features_dev, label=None, instance_eval=False)
                if attention.dim() > 2:
                    attention = attention[1]  # Use class-1 branch for CLAM-MB
                attention_scores = attention.squeeze().cpu().numpy()

            elif "transmil" in model_name:
                logits, _ = self.model(features_dev, coords=coords_tensor.to(self.device))
                # For TransMIL, use L2 norm of tile representations as proxy
                tile_reps = self.model.get_tile_representations(
                    features_dev, coords=coords_tensor.to(self.device)
                )
                attention_scores = torch.norm(tile_reps, dim=1).cpu().numpy()

            else:
                # ABMIL
                _, attention = self.model(features_dev, return_attention=True)
                attention_scores = attention.squeeze().cpu().numpy()

            # Get prediction
            if "clam" in model_name:
                logits, _, _ = self.model(features_dev, label=None, instance_eval=False)
            elif "transmil" in model_name:
                logits, _ = self.model(features_dev)
            else:
                logits, _ = self.model(features_dev)

            probs = F.softmax(logits.cpu(), dim=-1).numpy()

        # --- Step 4: Build spatial heatmap ---
        tile_size_l0 = int(tile_size * reader.level_downsample)

        heatmap = coords_to_spatial_map(
            coords=coords_arr,
            scores=attention_scores,
            tile_size_l0=tile_size_l0,
            slide_dims_l0=reader.slide.dimensions,
            map_resolution=self.map_resolution,
        )

        # Smooth and normalize
        heatmap = apply_gaussian_smoothing(heatmap, sigma=self.smooth_sigma)
        heatmap_norm = normalize_attention_map(heatmap)

        # --- Step 5: Overlay on thumbnail ---
        thumbnail = np.array(reader.get_thumbnail(size=self.map_resolution[0]))
        thumbnail_resized = cv2.resize(thumbnail, (self.map_resolution[1], self.map_resolution[0]))

        overlay = overlay_heatmap_on_thumbnail(
            thumbnail_resized,
            heatmap_norm,
            alpha=self.alpha,
            colormap=self.colormap,
            mask=cv2.resize(reader.tissue_mask, (self.map_resolution[1], self.map_resolution[0])),
        )

        # --- Step 6: Save outputs ---
        overlay_path = output_dir / f"{slide_id}_heatmap.png"
        Image.fromarray(overlay).save(str(overlay_path))
        logger.info("Saved heatmap overlay to %s", overlay_path)

        if save_intermediate:
            np.save(str(output_dir / f"{slide_id}_attention_map.npy"), heatmap_norm)
            np.save(str(output_dir / f"{slide_id}_coords.npy"), coords_arr)
            np.save(str(output_dir / f"{slide_id}_attention_scores.npy"), attention_scores)

        pred_class = int(probs.argmax())
        pred_prob = float(probs.max())

        reader.close()

        return {
            "slide_id": slide_id,
            "overlay_path": str(overlay_path),
            "attention_map": heatmap_norm,
            "attention_scores": attention_scores,
            "coords": coords_arr,
            "prediction": probs.tolist(),
            "predicted_class": pred_class,
            "predicted_prob": pred_prob,
        }

    def generate_batch(
        self,
        wsi_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        **kwargs,
    ) -> List[Dict]:
        """
        Generate heatmaps for multiple slides.

        Args:
            wsi_paths:  List of WSI file paths
            output_dir: Output directory
            **kwargs:   Passed to generate()

        Returns:
            List of result dicts
        """
        results = []
        for i, wsi_path in enumerate(wsi_paths, 1):
            logger.info("[%d/%d] Processing %s", i, len(wsi_paths), Path(wsi_path).name)
            try:
                result = self.generate(wsi_path, output_dir, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error("Failed to generate heatmap for %s: %s", wsi_path, e)
                results.append({"slide_id": Path(wsi_path).stem, "error": str(e)})
        return results


# ---------------------------------------------------------------------------
# Multi-resolution Blending
# ---------------------------------------------------------------------------


def multi_resolution_heatmap(
    attention_maps: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Blend attention heatmaps from multiple magnifications or models.

    Args:
        attention_maps: List of [H, W] normalized attention maps
        weights:        Blend weights (default: equal weighting)

    Returns:
        blended: [H, W] blended heatmap
    """
    if weights is None:
        weights = [1.0 / len(attention_maps)] * len(attention_maps)

    assert len(attention_maps) == len(weights), "Must have same number of maps and weights"

    # Resize all maps to first map's resolution
    H, W = attention_maps[0].shape[:2]
    blended = np.zeros((H, W), dtype=np.float32)

    for hm, w in zip(attention_maps, weights):
        hm_resized = cv2.resize(hm.astype(np.float32), (W, H))
        blended += w * hm_resized

    return normalize_attention_map(blended)
