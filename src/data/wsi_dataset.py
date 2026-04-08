"""
Whole Slide Image (WSI) Dataset Handling

Provides memory-efficient reading, tissue detection, and tile extraction
from gigapixel WSI files using OpenSlide.

Supported formats: SVS, TIFF, NDPI, MRXS, SCN, BIF, VMS, VMU

Tissue detection approach:
    1. Read thumbnail at low magnification (~1.25x)
    2. Convert to grayscale
    3. Otsu thresholding to separate tissue from glass
    4. Morphological operations (dilation) to fill holes
    5. Extract tiles only from tissue regions

Tile extraction:
    - Multiple magnification support: 5x, 10x, 20x, 40x
    - Track (x, y) pixel coordinates for spatial attention maps
    - Lazy loading: read tile images only when requested
"""

import logging
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Optional openslide import with helpful error message
try:
    import openslide
    from openslide import OpenSlide, OpenSlideError
    from openslide.deepzoom import DeepZoomGenerator
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logger.warning(
        "openslide-python not installed. WSI functionality will be limited. "
        "Install with: pip install openslide-python "
        "(also requires system library: sudo apt-get install openslide-tools)"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard magnification → approximate microns-per-pixel (MPP) mapping
MAG_TO_MPP = {
    40: 0.25,
    20: 0.50,
    10: 1.00,
    5:  2.00,
    2.5: 4.00,
    1.25: 8.00,
}

# Standard tile sizes
DEFAULT_TILE_SIZE = 224
TISSUE_THUMBNAIL_SIZE = 512  # Size for tissue detection thumbnail


# ---------------------------------------------------------------------------
# Tissue Segmentation
# ---------------------------------------------------------------------------


class TissueSegmenter:
    """
    Detects tissue regions in a WSI thumbnail using Otsu thresholding.

    Returns a binary tissue mask that guides tile extraction, avoiding
    empty glass, pen marks, and artifact regions.
    """

    def __init__(
        self,
        otsu_threshold: float = 0.0,  # 0 = auto Otsu
        min_tissue_area: int = 1000,  # pixels in thumbnail space
        kernel_size: int = 7,
        use_saturation: bool = True,
    ) -> None:
        self.otsu_threshold = otsu_threshold
        self.min_tissue_area = min_tissue_area
        self.kernel_size = kernel_size
        self.use_saturation = use_saturation

    def segment(
        self,
        thumbnail: np.ndarray,
    ) -> np.ndarray:
        """
        Compute tissue mask from WSI thumbnail.

        Args:
            thumbnail: RGB image [H, W, 3], uint8

        Returns:
            tissue_mask: Binary mask [H, W], uint8 (255=tissue, 0=background)
        """
        if self.use_saturation:
            # HSV saturation channel: better than grayscale for H&E
            hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
            channel = hsv[:, :, 1]  # Saturation
            _, mask = cv2.threshold(
                channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            # Standard grayscale Otsu
            gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
            if self.otsu_threshold > 0:
                _, mask = cv2.threshold(
                    gray, int(self.otsu_threshold * 255), 255, cv2.THRESH_BINARY_INV
                )
            else:
                _, mask = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < self.min_tissue_area:
                mask[labels == label] = 0

        return mask

    def get_tissue_coordinates(
        self,
        tissue_mask: np.ndarray,
        tile_size_thumb: int,
        overlap: float = 0.0,
    ) -> List[Tuple[int, int]]:
        """
        Get (row, col) coordinates of tissue tiles in thumbnail space.

        Args:
            tissue_mask:     Binary mask [H, W]
            tile_size_thumb: Tile size in thumbnail pixels
            overlap:         Fractional overlap between tiles (0.0 = no overlap)

        Returns:
            List of (row, col) tuples for tiles with tissue content
        """
        stride = max(1, int(tile_size_thumb * (1 - overlap)))
        H, W = tissue_mask.shape
        coords = []

        for r in range(0, H - tile_size_thumb + 1, stride):
            for c in range(0, W - tile_size_thumb + 1, stride):
                tile_mask = tissue_mask[r:r + tile_size_thumb, c:c + tile_size_thumb]
                tissue_pct = (tile_mask > 0).mean()
                if tissue_pct > 0.5:  # >50% tissue in tile
                    coords.append((r, c))

        return coords


# ---------------------------------------------------------------------------
# WSI Reader
# ---------------------------------------------------------------------------


class WSIReader:
    """
    OpenSlide-based WSI reader with tissue detection and tile extraction.

    Handles the fundamental challenge of gigapixel images: a 40x WSI can
    be 100,000+ × 100,000+ pixels (>10 GB uncompressed). This class reads
    tiles lazily without loading the full image into memory.

    Usage:
        reader = WSIReader("slide.svs", target_magnification=20)
        for tile, coord in reader.iter_tiles(tile_size=224):
            features = extractor(tile)  # Process tile
    """

    def __init__(
        self,
        wsi_path: Union[str, Path],
        target_magnification: float = 20.0,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = 0,
        tissue_threshold: float = 0.5,
        min_tile_tissue_pct: float = 0.5,
        segmenter: Optional[TissueSegmenter] = None,
    ) -> None:
        """
        Args:
            wsi_path:              Path to WSI file (.svs, .tiff, .ndpi, etc.)
            target_magnification:  Target magnification for tile extraction
            tile_size:             Tile size in pixels
            tile_overlap:          Overlap in pixels between adjacent tiles
            tissue_threshold:      Minimum tissue fraction to include tile
            min_tile_tissue_pct:   Minimum tissue % within tile
            segmenter:             Tissue segmenter (default: TissueSegmenter())
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "openslide-python is required. "
                "Install: pip install openslide-python"
            )

        self.wsi_path = Path(wsi_path)
        self.target_magnification = target_magnification
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tissue_threshold = tissue_threshold
        self.min_tile_tissue_pct = min_tile_tissue_pct
        self.segmenter = segmenter or TissueSegmenter()

        # Open slide
        try:
            self.slide = OpenSlide(str(self.wsi_path))
        except OpenSlideError as e:
            raise ValueError(f"Cannot open WSI: {self.wsi_path}. Error: {e}")

        # Parse metadata
        self._parse_metadata()
        self._compute_extraction_level()
        self._compute_tissue_mask()

    def _parse_metadata(self) -> None:
        """Extract slide metadata from OpenSlide properties."""
        props = self.slide.properties

        # Scan magnification
        self.scan_magnification = None
        for key in ("openslide.objective-power", "aperio.AppMag", "hamamatsu.SourceLens"):
            if key in props:
                try:
                    self.scan_magnification = float(props[key])
                    break
                except (ValueError, TypeError):
                    pass

        if self.scan_magnification is None:
            self.scan_magnification = 40.0  # Assume 40x if not found
            logger.warning(
                "Could not determine scan magnification for %s. Assuming 40x.",
                self.wsi_path.name,
            )

        # MPP (microns per pixel)
        self.mpp_x = float(props.get("openslide.mpp-x", MAG_TO_MPP.get(self.scan_magnification, 0.25)))
        self.mpp_y = float(props.get("openslide.mpp-y", MAG_TO_MPP.get(self.scan_magnification, 0.25)))

        # Slide dimensions at level 0 (highest resolution)
        self.width_l0, self.height_l0 = self.slide.dimensions

        logger.info(
            "WSI: %s | Dims: %dx%d | Scan mag: %.0fx | MPP: %.3f",
            self.wsi_path.name,
            self.width_l0,
            self.height_l0,
            self.scan_magnification,
            self.mpp_x,
        )

    def _compute_extraction_level(self) -> None:
        """
        Find the OpenSlide level closest to the target magnification.
        """
        # Downsample factor from scan mag to target mag
        target_downsample = self.scan_magnification / self.target_magnification
        level_downsamples = self.slide.level_downsamples

        # Find best level
        best_level = 0
        best_diff = float("inf")
        for level, ds in enumerate(level_downsamples):
            diff = abs(ds - target_downsample)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        self.extraction_level = best_level
        self.level_downsample = level_downsamples[best_level]
        self.effective_magnification = self.scan_magnification / self.level_downsample

        # Tile size in level-0 coordinates (for read_region)
        self.tile_size_l0 = int(self.tile_size * self.level_downsample)
        self.stride_l0 = self.tile_size_l0 - int(self.tile_overlap * self.level_downsample)

        logger.info(
            "Extraction level: %d (downsample: %.2fx, effective mag: %.1fx)",
            self.extraction_level,
            self.level_downsample,
            self.effective_magnification,
        )

    def _compute_tissue_mask(self) -> None:
        """Generate tissue mask from low-resolution thumbnail."""
        # Use the lowest resolution level for tissue detection
        thumb_level = self.slide.level_count - 1
        thumb_dim = self.slide.level_dimensions[thumb_level]
        thumbnail = self.slide.read_region((0, 0), thumb_level, thumb_dim).convert("RGB")
        self.thumbnail = np.array(thumbnail)

        # Scale factor: thumbnail → level0
        self.thumb_to_l0_x = self.width_l0 / thumb_dim[0]
        self.thumb_to_l0_y = self.height_l0 / thumb_dim[1]

        # Compute tissue mask
        self.tissue_mask = self.segmenter.segment(self.thumbnail)

        tissue_pct = (self.tissue_mask > 0).mean() * 100
        logger.info(
            "Tissue segmentation: %.1f%% tissue detected", tissue_pct
        )

    def get_thumbnail(self, size: int = 512) -> Image.Image:
        """Get a thumbnail of the WSI."""
        aspect = self.height_l0 / self.width_l0
        thumb = self.slide.get_thumbnail((size, int(size * aspect)))
        return thumb

    def _is_tissue_tile(
        self,
        x_l0: int,
        y_l0: int,
    ) -> bool:
        """
        Check if a tile at level-0 coordinates contains sufficient tissue.
        """
        # Map to thumbnail coordinates
        x_thumb = int(x_l0 / self.thumb_to_l0_x)
        y_thumb = int(y_l0 / self.thumb_to_l0_y)

        tile_w_thumb = max(1, int(self.tile_size_l0 / self.thumb_to_l0_x))
        tile_h_thumb = max(1, int(self.tile_size_l0 / self.thumb_to_l0_y))

        # Clip to mask bounds
        x1 = max(0, x_thumb)
        y1 = max(0, y_thumb)
        x2 = min(self.tissue_mask.shape[1], x_thumb + tile_w_thumb)
        y2 = min(self.tissue_mask.shape[0], y_thumb + tile_h_thumb)

        if x2 <= x1 or y2 <= y1:
            return False

        tile_mask = self.tissue_mask[y1:y2, x1:x2]
        return (tile_mask > 0).mean() >= self.min_tile_tissue_pct

    def iter_tiles(
        self,
        return_coords: bool = True,
    ) -> Generator[Tuple[Image.Image, Tuple[int, int]], None, None]:
        """
        Iterate over tissue tiles in the WSI.

        Yields:
            tile:  PIL Image of size (tile_size, tile_size)
            coord: (x, y) in level-0 pixel coordinates
        """
        width_l, height_l = self.slide.level_dimensions[self.extraction_level]
        stride = max(1, self.tile_size - self.tile_overlap)

        n_x = max(1, (width_l - self.tile_size) // stride + 1)
        n_y = max(1, (height_l - self.tile_size) // stride + 1)

        total = n_x * n_y
        logger.info("Scanning %d candidate tiles (%.0fx magnification)...", total, self.effective_magnification)

        for yi in range(n_y):
            for xi in range(n_x):
                # Coordinates in level-0 space
                x_l0 = int(xi * stride * self.level_downsample)
                y_l0 = int(yi * stride * self.level_downsample)

                # Bounds check
                if x_l0 + self.tile_size_l0 > self.width_l0:
                    continue
                if y_l0 + self.tile_size_l0 > self.height_l0:
                    continue

                # Tissue check
                if not self._is_tissue_tile(x_l0, y_l0):
                    continue

                # Read tile
                tile = self.slide.read_region(
                    (x_l0, y_l0),
                    self.extraction_level,
                    (self.tile_size, self.tile_size),
                ).convert("RGB")

                if return_coords:
                    yield tile, (x_l0, y_l0)
                else:
                    yield tile

    def get_all_tiles(self) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Extract all tissue tiles into memory.

        Returns:
            tiles:  List of PIL Images
            coords: Array [N, 2] of (x, y) level-0 coordinates
        """
        tiles = []
        coords = []
        for tile, coord in self.iter_tiles():
            tiles.append(tile)
            coords.append(coord)

        coords_array = np.array(coords, dtype=np.float32) if coords else np.zeros((0, 2))
        logger.info("Extracted %d tissue tiles from %s", len(tiles), self.wsi_path.name)
        return tiles, coords_array

    def get_tile_at_coord(
        self,
        x_l0: int,
        y_l0: int,
        size: Optional[int] = None,
    ) -> Image.Image:
        """Read a specific tile by level-0 coordinates."""
        size = size or self.tile_size
        tile = self.slide.read_region(
            (x_l0, y_l0),
            self.extraction_level,
            (size, size),
        ).convert("RGB")
        return tile

    @property
    def slide_id(self) -> str:
        """Slide identifier from filename (without extension)."""
        return self.wsi_path.stem

    @property
    def n_tiles_estimate(self) -> int:
        """Approximate number of tissue tiles."""
        tissue_area = (self.tissue_mask > 0).sum()
        total_area = self.tissue_mask.size
        tissue_fraction = tissue_area / total_area

        width_l, height_l = self.slide.level_dimensions[self.extraction_level]
        stride = max(1, self.tile_size - self.tile_overlap)
        total_tiles = (width_l // stride) * (height_l // stride)
        return int(total_tiles * tissue_fraction)

    def close(self) -> None:
        """Close the OpenSlide handle."""
        if hasattr(self, "slide") and self.slide:
            self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"WSIReader({self.wsi_path.name}, "
            f"mag={self.effective_magnification:.0f}x, "
            f"~{self.n_tiles_estimate} tiles)"
        )


# ---------------------------------------------------------------------------
# WSI Bag Dataset (for MIL training/evaluation)
# ---------------------------------------------------------------------------


class WSIBagDataset:
    """
    Dataset of pre-extracted feature bags for MIL training.

    Each "bag" is a (features, coords, label) tuple where:
        - features: [N, D] tensor of tile features
        - coords:   [N, 2] tensor of tile coordinates
        - label:    integer class label

    Features are loaded from a cache directory containing .pt files
    generated by FeatureExtractorPipeline.
    """

    def __init__(
        self,
        feature_dir: Union[str, Path],
        label_dict: Dict[str, int],
        max_bag_size: Optional[int] = None,
        min_bag_size: int = 10,
        shuffle: bool = True,
    ) -> None:
        """
        Args:
            feature_dir:  Directory containing {slide_id}_{extractor}.pt files
            label_dict:   Mapping from slide_id to integer label
            max_bag_size: If set, randomly sample at most N tiles per slide
            min_bag_size: Skip slides with fewer than this many tiles
            shuffle:      Shuffle tiles within each bag (for training)
        """
        self.feature_dir = Path(feature_dir)
        self.label_dict = label_dict
        self.max_bag_size = max_bag_size
        self.min_bag_size = min_bag_size
        self.shuffle = shuffle

        # Discover available feature files
        self.slide_ids = [
            sid for sid in label_dict
            if self._has_features(sid)
        ]

        missing = len(label_dict) - len(self.slide_ids)
        if missing > 0:
            logger.warning("%d slides missing feature files", missing)
        logger.info("WSIBagDataset: %d slides loaded", len(self.slide_ids))

    def _has_features(self, slide_id: str) -> bool:
        """Check if feature file exists for a slide."""
        pattern = list(self.feature_dir.glob(f"{slide_id}_*.pt"))
        return len(pattern) > 0

    def _get_feature_path(self, slide_id: str) -> Path:
        """Find feature file for a slide."""
        matches = list(self.feature_dir.glob(f"{slide_id}_*.pt"))
        if not matches:
            raise FileNotFoundError(f"No features for slide: {slide_id}")
        return matches[0]  # Use first match

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Load a slide bag.

        Returns:
            features: [N, D]
            coords:   [N, 2] or None
            label:    integer
        """
        slide_id = self.slide_ids[idx]
        label = self.label_dict[slide_id]

        data = torch.load(self._get_feature_path(slide_id), map_location="cpu")
        features = data["features"]  # [N, D]
        coords = data.get("coords")   # [N, 2] or None

        N = features.shape[0]

        # Filter minimum bag size
        if N < self.min_bag_size:
            logger.warning("Slide %s has only %d tiles (< min %d)", slide_id, N, self.min_bag_size)

        # Shuffle tiles
        if self.shuffle:
            perm = torch.randperm(N)
            features = features[perm]
            if coords is not None:
                coords = coords[perm]

        # Subsample bag
        if self.max_bag_size and N > self.max_bag_size:
            idx_sample = torch.randperm(N)[:self.max_bag_size]
            features = features[idx_sample]
            if coords is not None:
                coords = coords[idx_sample]

        return features, coords, label

    def get_slide_ids(self) -> List[str]:
        return self.slide_ids.copy()

    def class_counts(self) -> Dict[int, int]:
        """Return count per class for class-weighted sampling."""
        counts: Dict[int, int] = {}
        for sid in self.slide_ids:
            label = self.label_dict[sid]
            counts[label] = counts.get(label, 0) + 1
        return counts

DEFAULT_TILE_SIZE = 256  # pixels at 20x magnification

# fix magnification level selection in OpenSlide
# was using level 0 always; now selects the best level for target magnification
def get_best_level_for_magnification(slide, target_mpp=0.5):
    """select the best pyramid level for a given target resolution (MPP)
    
    previously always defaulted to level 0 regardless of requested magnification
    causing unnecessarily large image reads and OOM errors on high-res slides
    """
    try:
        native_mpp = float(slide.properties.get('openslide.mpp-x', 0.25))
    except (KeyError, ValueError):
        native_mpp = 0.25  # assume 40x if metadata missing

    # find level with downsample closest to target_mpp / native_mpp
    target_ds = target_mpp / native_mpp
    best_level = 0
    best_diff = abs(slide.level_downsamples[0] - target_ds)
    for level, ds in enumerate(slide.level_downsamples):
        diff = abs(ds - target_ds)
        if diff < best_diff:
            best_diff = diff
            best_level = level
    return best_level
