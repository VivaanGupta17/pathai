#!/usr/bin/env python3
"""
Feature Extraction Script

Extracts tile features from WSI files and caches them as .pt files
for efficient MIL training (extract once, train many times).

Usage:
    python scripts/extract_features.py \
        --data_dir /data/camelyon16 \
        --output_dir data/features/ctranspath_20x \
        --feature_extractor ctranspath \
        --magnification 20 \
        --tile_size 224 \
        --batch_size 256 \
        --gpu 0

    # Extract with stain normalization
    python scripts/extract_features.py \
        --data_dir /data/camelyon16 \
        --output_dir data/features/ctranspath_20x_macenko \
        --feature_extractor ctranspath \
        --stain_normalize macenko \
        --gpu 0

    # Process only specific slides (e.g., test set)
    python scripts/extract_features.py \
        --data_dir /data/camelyon16 \
        --slide_list data/test_slides.txt \
        --output_dir data/features/test_ctranspath

Performance:
    - CTransPath: ~2-5 min/slide (20x, V100)
    - ResNet50: ~30-60 sec/slide (20x, V100)
    - Expected output: ~2 GB for all 400 Camelyon16 training slides
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tile features from WSI files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing WSI files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for feature .pt files")
    parser.add_argument("--slide_list", type=str, default=None,
                        help="Text file with one slide filename per line (optional)")
    parser.add_argument("--extension", type=str, default=None,
                        help="WSI file extension (auto-detect if not specified)")

    # Feature extractor
    parser.add_argument("--feature_extractor", type=str, default="ctranspath",
                        choices=["resnet50", "ctranspath", "uni", "conch"],
                        help="Feature extractor model")
    parser.add_argument("--extractor_weights", type=str, default=None,
                        help="Path to pretrained feature extractor weights")

    # Tile extraction
    parser.add_argument("--magnification", type=float, default=20.0,
                        help="Target magnification for tile extraction")
    parser.add_argument("--tile_size", type=int, default=224,
                        help="Tile size in pixels")
    parser.add_argument("--tile_overlap", type=int, default=0,
                        help="Tile overlap in pixels")
    parser.add_argument("--stain_normalize", type=str, default="macenko",
                        choices=["macenko", "reinhard", "none"],
                        help="Stain normalization method")

    # Compute
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (-1 for CPU)")

    # Options
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip slides with existing feature files")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    return parser.parse_args()


def find_wsi_files(data_dir: Path, extension: str = None) -> list:
    """Find all WSI files in data_dir."""
    extensions = [extension] if extension else [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn"]
    wsi_files = []
    for ext in extensions:
        wsi_files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(wsi_files)


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(args.gpu))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Build feature extractor
    from src.models.feature_extractor import build_feature_extractor, FeatureExtractorPipeline

    logger.info("Loading feature extractor: %s", args.feature_extractor)
    extractor = build_feature_extractor(
        args.feature_extractor,
        weights_path=args.extractor_weights,
        device=device,
    )

    # Tile processor
    from src.data.tile_processing import TileProcessor
    processor = TileProcessor.default(normalize=args.stain_normalize)

    # Setup extraction pipeline
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FeatureExtractorPipeline(
        extractor=extractor,
        extractor_name=args.feature_extractor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=str(output_dir),
        device=device,
    )

    # Find WSI files
    data_dir = Path(args.data_dir)
    if args.slide_list:
        with open(args.slide_list) as f:
            slide_names = [line.strip() for line in f if line.strip()]
        wsi_files = [data_dir / name for name in slide_names if (data_dir / name).exists()]
    else:
        wsi_files = find_wsi_files(data_dir, args.extension)

    logger.info("Found %d WSI files", len(wsi_files))

    # Filter already-processed slides
    if args.skip_existing:
        remaining = []
        for wsi_path in wsi_files:
            cache_path = output_dir / f"{wsi_path.stem}_{args.feature_extractor}.pt"
            if cache_path.exists():
                logger.debug("Skipping %s (already processed)", wsi_path.name)
            else:
                remaining.append(wsi_path)
        logger.info("Processing %d slides (%d already done)", len(remaining), len(wsi_files) - len(remaining))
        wsi_files = remaining

    if not wsi_files:
        logger.info("All slides already processed. Done.")
        return

    # Process slides
    from src.data.wsi_dataset import WSIReader

    success_count = 0
    fail_count = 0
    total_tiles = 0

    for i, wsi_path in enumerate(tqdm(wsi_files, desc="Extracting features"), 1):
        t0 = time.time()
        slide_id = wsi_path.stem
        try:
            reader = WSIReader(
                wsi_path,
                target_magnification=args.magnification,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )

            tiles = []
            coords_list = []

            for tile, coord in reader.iter_tiles():
                processed = processor.process(tile, augment=False)
                if processed is not None:
                    tiles.append(processed)
                    coords_list.append(coord)

            reader.close()

            if not tiles:
                logger.warning("No tissue tiles in %s — skipping", slide_id)
                continue

            import numpy as np
            coords_arr = np.array(coords_list, dtype=np.float32)

            features, coords_tensor = pipeline.extract(
                tiles, coords=coords_arr, slide_id=slide_id
            )

            t_elapsed = time.time() - t0
            total_tiles += len(tiles)
            success_count += 1

            tqdm.write(
                f"[{i:4d}/{len(wsi_files)}] {slide_id}: "
                f"{len(tiles)} tiles, {features.shape} features, {t_elapsed:.1f}s"
            )

        except Exception as e:
            fail_count += 1
            logger.error("Failed to process %s: %s", slide_id, e)

    logger.info(
        "\nExtraction complete: %d succeeded, %d failed, %d total tiles",
        success_count, fail_count, total_tiles,
    )
    logger.info("Features saved to: %s", output_dir)


if __name__ == "__main__":
    main()
