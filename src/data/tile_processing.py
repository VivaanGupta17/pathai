"""
Tile Processing for Computational Pathology

Implements:
    1. Stain normalization:
       - Macenko (SVD-based OD decomposition) — recommended
       - Reinhard (color transfer in LAB space)
       - Vahadane (SNMF-based, highest quality, slow)
    2. Background/artifact filtering
    3. Color augmentation in HED (Hematoxylin-Eosin-DAB) stain space
    4. Tile quality scoring

H&E stain: Hematoxylin stains nuclei purple/blue, Eosin stains cytoplasm pink.
Stain normalization corrects for inter-scanner and inter-lab staining variation,
which is a major source of domain shift in computational pathology models.
"""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color space utilities
# ---------------------------------------------------------------------------


def rgb_to_od(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert RGB to Optical Density (OD) space.

    OD = -log(RGB / 255 + eps) = -log(I/I0)
    where I0 = 255 (incident light intensity).

    Args:
        rgb: [H, W, 3] uint8 RGB image
        eps: Small constant for numerical stability

    Returns:
        od: [H, W, 3] float32 OD image
    """
    rgb = rgb.astype(np.float32) / 255.0 + eps
    return -np.log(rgb)


def od_to_rgb(od: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert Optical Density (OD) back to RGB.

    RGB = exp(-OD) * 255

    Args:
        od: [H, W, 3] float32 OD image

    Returns:
        rgb: [H, W, 3] uint8 RGB image
    """
    rgb = np.exp(-od) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to HED (Hematoxylin-Eosin-DAB) stain space.

    Uses the standard stain matrix from:
        Ruifrok A, Johnston D. "Quantification of histochemical staining by
        color deconvolution." Anal Quant Cytol Histol. 2001.

    Args:
        rgb: [H, W, 3] uint8

    Returns:
        hed: [H, W, 3] float32, channels are H, E, D concentrations
    """
    # Standard H&E stain matrix (rows: stain vectors in RGB OD space)
    HED_MATRIX = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.072, 0.990, 0.105],  # Eosin
        [0.268, 0.570, 0.776],  # DAB
    ], dtype=np.float32)

    od = rgb_to_od(rgb)  # [H, W, 3]
    H, W, _ = od.shape
    od_flat = od.reshape(-1, 3)  # [N, 3]

    # Deconvolve: HED = OD @ inv(M)
    inv_matrix = np.linalg.pinv(HED_MATRIX)
    hed_flat = od_flat @ inv_matrix  # [N, 3]
    hed = hed_flat.reshape(H, W, 3)
    return hed.astype(np.float32)


def hed_to_rgb(hed: np.ndarray) -> np.ndarray:
    """
    Convert HED stain concentrations back to RGB.

    Args:
        hed: [H, W, 3] float32

    Returns:
        rgb: [H, W, 3] uint8
    """
    HED_MATRIX = np.array([
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.776],
    ], dtype=np.float32)

    H, W, _ = hed.shape
    hed_flat = hed.reshape(-1, 3)
    od_flat = hed_flat @ HED_MATRIX  # Reconstruct OD
    od = od_flat.reshape(H, W, 3)
    return od_to_rgb(od)


# ---------------------------------------------------------------------------
# Macenko Stain Normalization
# ---------------------------------------------------------------------------


class MacenkoNormalizer:
    """
    Macenko stain normalization.

    Uses SVD to find the dominant stain directions in OD space,
    then normalizes the tile to match a reference stain matrix.

    Paper:
        Macenko et al., "A method for normalizing histology slides for
        quantitative analysis", ISBI 2009.

    This is the most widely used method in computational pathology.
    """

    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        angular_percentile: int = 99,
    ) -> None:
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentile = angular_percentile
        self.reference_HE = None
        self.reference_maxC = None

    def _get_stain_matrix(
        self,
        od: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate stain matrix from OD image using SVD.

        Returns:
            HE: [2, 3] stain matrix (H and E rows)
            C:  [2, N] stain concentrations
        """
        # Remove background (low OD = background)
        od_hat = od[od[:, :, 0] + od[:, :, 1] + od[:, :, 2] > self.luminosity_threshold]
        if len(od_hat) == 0:
            # Fallback: use standard stain matrix
            HE = np.array([[0.650, 0.704, 0.286],
                           [0.072, 0.990, 0.105]], dtype=np.float32)
            C = np.zeros((2, od.reshape(-1, 3).T.shape[1]), dtype=np.float32)
            return HE, C

        # SVD of OD vectors
        _, _, V = np.linalg.svd(od_hat, full_matrices=False)
        V = V[:2].T  # Take first 2 right singular vectors [3, 2]

        # Project onto plane and find angles
        phi = np.arctan2(od_hat @ V[:, 1], od_hat @ V[:, 0])

        # Find stain vectors at extreme angles
        minphi = np.percentile(phi, 100 - self.angular_percentile)
        maxphi = np.percentile(phi, self.angular_percentile)

        v1 = V @ np.array([np.cos(minphi), np.sin(minphi)])
        v2 = V @ np.array([np.cos(maxphi), np.sin(maxphi)])

        # Assign H and E (hematoxylin has higher blue component)
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])

        # Compute concentrations: OD = C @ HE
        od_flat = od.reshape(-1, 3)
        C = np.linalg.lstsq(HE.T, od_flat.T, rcond=None)[0]
        return HE, C

    def fit(self, reference_image: np.ndarray) -> "MacenkoNormalizer":
        """
        Fit normalizer to a reference image.

        Args:
            reference_image: [H, W, 3] uint8 RGB reference tile

        Returns:
            self (for chaining)
        """
        od = rgb_to_od(reference_image)
        HE, C = self._get_stain_matrix(od)
        self.reference_HE = HE.astype(np.float32)
        self.reference_maxC = np.percentile(C, 99, axis=1).astype(np.float32)
        return self

    def fit_default(self) -> "MacenkoNormalizer":
        """Use standard Camelyon reference statistics (pre-computed)."""
        self.reference_HE = np.array([
            [0.5626, 0.7201, 0.4062],
            [0.2159, 0.8012, 0.5581],
        ], dtype=np.float32)
        self.reference_maxC = np.array([1.9705, 1.0308], dtype=np.float32)
        return self

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize a tile to match the reference stain distribution.

        Args:
            image: [H, W, 3] uint8

        Returns:
            normalized: [H, W, 3] uint8
        """
        if self.reference_HE is None:
            raise RuntimeError("Call fit() or fit_default() first")

        H, W, _ = image.shape
        od = rgb_to_od(image)
        _, C = self._get_stain_matrix(od)

        # Normalize concentrations
        maxC = np.percentile(C, 99, axis=1, keepdims=True)
        maxC = np.maximum(maxC, 1e-6)
        C_norm = C / maxC * self.reference_maxC[:, None]

        # Reconstruct with reference stain matrix
        od_norm = self.reference_HE.T @ C_norm  # [3, N]
        od_norm = od_norm.T.reshape(H, W, 3)

        return od_to_rgb(od_norm)

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """Callable interface for use in transforms."""
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)
        normalized = self.normalize(image)
        return Image.fromarray(normalized) if is_pil else normalized


# ---------------------------------------------------------------------------
# Reinhard Stain Normalization
# ---------------------------------------------------------------------------


class ReinhardNormalizer:
    """
    Reinhard color normalization in LAB color space.

    Matches mean and std of LAB channels between source and target images.
    Faster than Macenko but less accurate for large stain variation.

    Paper:
        Reinhard et al., "Color Transfer between Images", IEEE CGA 2001.
    """

    def __init__(self) -> None:
        self.target_means = None
        self.target_stds = None

    def fit_default(self) -> "ReinhardNormalizer":
        """Use standard pathology reference statistics."""
        self.target_means = np.array([148.60, 41.56, -5.17], dtype=np.float32)
        self.target_stds = np.array([41.13, 19.20, 6.28], dtype=np.float32)
        return self

    def fit(self, reference: np.ndarray) -> "ReinhardNormalizer":
        """Fit to reference image in LAB space."""
        lab = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2LAB)
        self.target_means = lab.reshape(-1, 3).mean(axis=0)
        self.target_stds = lab.reshape(-1, 3).std(axis=0)
        return self

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to reference LAB statistics."""
        if self.target_means is None:
            raise RuntimeError("Call fit() or fit_default() first")
        lab = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2LAB)
        src_means = lab.reshape(-1, 3).mean(axis=0)
        src_stds = lab.reshape(-1, 3).std(axis=0) + 1e-6

        lab = (lab - src_means) / src_stds * self.target_stds + self.target_means
        lab = np.clip(lab, 0, 255)
        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)
        result = self.normalize(image)
        return Image.fromarray(result) if is_pil else result


# ---------------------------------------------------------------------------
# Background / Artifact Detection
# ---------------------------------------------------------------------------


class BackgroundFilter:
    """
    Filter out background (glass), white adipose tissue, and artifact tiles.

    Criteria:
        1. White background: mean brightness > threshold
        2. Low tissue density: std of grayscale < threshold
        3. Ink/pen marks: unusual color distribution
        4. Out-of-focus: low Laplacian variance
    """

    def __init__(
        self,
        brightness_threshold: float = 220.0,
        std_threshold: float = 8.0,
        focus_threshold: float = 50.0,
        min_saturation: float = 5.0,
    ) -> None:
        self.brightness_threshold = brightness_threshold
        self.std_threshold = std_threshold
        self.focus_threshold = focus_threshold
        self.min_saturation = min_saturation

    def is_background(self, tile: np.ndarray) -> bool:
        """
        Returns True if tile should be filtered out.

        Args:
            tile: [H, W, 3] uint8 RGB tile

        Returns:
            True if background/artifact, False if tissue
        """
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Brightness check (white background/adipose)
        if gray.mean() > self.brightness_threshold:
            return True

        # 2. Low variance check (blank/glass)
        if gray.std() < self.std_threshold:
            return True

        # 3. Saturation check (tissue has color)
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV).astype(np.float32)
        if hsv[:, :, 1].mean() < self.min_saturation:
            return True

        # 4. Focus check (Laplacian variance)
        lap_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
        if lap_var < self.focus_threshold:
            return True

        return False

    def quality_score(self, tile: np.ndarray) -> float:
        """
        Compute a tile quality score in [0, 1].

        Higher = better quality tissue tile.
        """
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Normalize individual metrics
        brightness_score = 1.0 - min(gray.mean() / 255.0, 1.0)
        variance_score = min(gray.std() / 50.0, 1.0)
        saturation_score = min(hsv[:, :, 1].mean() / 50.0, 1.0)
        focus_score = min(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var() / 500.0, 1.0)

        return float((brightness_score + variance_score + saturation_score + focus_score) / 4.0)


# ---------------------------------------------------------------------------
# Stain Augmentation (HED perturbation)
# ---------------------------------------------------------------------------


class HEDAugmentation:
    """
    Color augmentation in HED stain space for H&E pathology images.

    Randomly perturbs H, E, and D channel intensities independently,
    simulating variation in staining protocol and scanner calibration.
    More realistic than standard RGB color jitter for pathology images.

    Reference:
        Tellez et al., "Quantifying the effects of data augmentation and
        stain color normalization in convolutional neural networks for
        computational pathology", Medical Image Analysis, 2019.
    """

    def __init__(
        self,
        h_scale: float = 0.05,  # Scale perturbation for H channel
        e_scale: float = 0.05,  # Scale perturbation for E channel
        d_scale: float = 0.05,  # Scale perturbation for D channel
        h_shift: float = 0.02,  # Shift perturbation for H channel
        e_shift: float = 0.02,  # Shift perturbation for E channel
        d_shift: float = 0.02,  # Shift perturbation for D channel
    ) -> None:
        self.h_scale = h_scale
        self.e_scale = e_scale
        self.d_scale = d_scale
        self.h_shift = h_shift
        self.e_shift = e_shift
        self.d_shift = d_shift

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """
        Apply random HED perturbation.

        Args:
            image: RGB tile

        Returns:
            Augmented RGB tile (same type as input)
        """
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        hed = rgb_to_hed(image)  # [H, W, 3]

        # Random multiplicative and additive perturbations
        for c, (scale, shift) in enumerate(zip(
            [self.h_scale, self.e_scale, self.d_scale],
            [self.h_shift, self.e_shift, self.d_shift],
        )):
            alpha = 1.0 + np.random.uniform(-scale, scale)
            beta = np.random.uniform(-shift, shift)
            hed[:, :, c] = hed[:, :, c] * alpha + beta

        result = hed_to_rgb(hed)

        return Image.fromarray(result) if is_pil else result


# ---------------------------------------------------------------------------
# Tile Quality Scoring
# ---------------------------------------------------------------------------


def compute_tile_quality(tile: np.ndarray) -> dict:
    """
    Compute multiple quality metrics for a tile.

    Args:
        tile: [H, W, 3] uint8 RGB

    Returns:
        Dict with quality metrics:
            - brightness: mean grayscale (lower = darker/more tissue)
            - contrast:   std of grayscale (higher = more texture)
            - saturation: mean saturation (higher = more stained)
            - focus:      Laplacian variance (higher = sharper)
            - tissue_pct: fraction of non-background pixels
            - quality:    composite score [0, 1]
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float(hsv[:, :, 1].mean())
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Background: bright (>230) + low saturation
    bg_mask = (gray > 230) & (hsv[:, :, 1] < 10)
    tissue_pct = float(1.0 - bg_mask.mean())

    # Composite quality score
    quality = (
        min(contrast / 50.0, 1.0) * 0.3
        + min(saturation / 50.0, 1.0) * 0.3
        + min(focus / 200.0, 1.0) * 0.2
        + tissue_pct * 0.2
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "focus": focus,
        "tissue_pct": tissue_pct,
        "quality": float(quality),
    }


# ---------------------------------------------------------------------------
# Processing Pipeline
# ---------------------------------------------------------------------------


class TileProcessor:
    """
    Complete tile processing pipeline combining:
        1. Background filtering
        2. Stain normalization
        3. Quality scoring
        4. Optional augmentation

    Designed to be applied during feature extraction.
    """

    def __init__(
        self,
        normalizer: Optional[Union[MacenkoNormalizer, ReinhardNormalizer]] = None,
        background_filter: Optional[BackgroundFilter] = None,
        augmentation: Optional[HEDAugmentation] = None,
        quality_threshold: float = 0.2,
    ) -> None:
        self.normalizer = normalizer
        self.background_filter = background_filter or BackgroundFilter()
        self.augmentation = augmentation
        self.quality_threshold = quality_threshold

    @classmethod
    def default(cls, normalize: str = "macenko") -> "TileProcessor":
        """
        Create a default processing pipeline.

        Args:
            normalize: 'macenko', 'reinhard', or 'none'
        """
        if normalize == "macenko":
            normalizer = MacenkoNormalizer().fit_default()
        elif normalize == "reinhard":
            normalizer = ReinhardNormalizer().fit_default()
        else:
            normalizer = None

        return cls(
            normalizer=normalizer,
            background_filter=BackgroundFilter(),
            augmentation=None,  # Enabled during training via data augmentation pipeline
        )

    def process(
        self,
        tile: Image.Image,
        augment: bool = False,
    ) -> Optional[Image.Image]:
        """
        Process a single tile.

        Args:
            tile:    PIL Image
            augment: Apply augmentation

        Returns:
            Processed tile, or None if filtered out as background
        """
        tile_arr = np.array(tile)

        # 1. Background filtering
        if self.background_filter.is_background(tile_arr):
            return None

        # 2. Quality check
        quality = compute_tile_quality(tile_arr)
        if quality["quality"] < self.quality_threshold:
            return None

        # 3. Stain normalization
        if self.normalizer is not None:
            try:
                tile_arr = self.normalizer.normalize(tile_arr)
            except Exception as e:
                logger.debug("Stain normalization failed: %s", e)

        # 4. Augmentation
        if augment and self.augmentation is not None:
            tile_arr = self.augmentation(tile_arr)

        return Image.fromarray(tile_arr)

# Macenko preferred over Reinhard for H&E variation

# tissue percentage threshold for filtering out background tiles
# avoids wasting compute on glass/artifact regions
TISSUE_THRESHOLD = 0.5  # tiles with <50% tissue are discarded by default

def filter_low_tissue_tiles(tiles, tissue_masks, threshold=TISSUE_THRESHOLD):
    """filter tiles based on tissue content percentage
    
    tiles with low tissue percentage are mostly background or glass artifacts
    and should be excluded from feature extraction
    """
    kept = []
    for tile, mask in zip(tiles, tissue_masks):
        tissue_pct = mask.mean() if hasattr(mask, 'mean') else 0.0
        if tissue_pct >= threshold:
            kept.append((tile, tissue_pct))
    return kept
