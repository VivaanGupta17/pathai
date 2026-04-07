#!/usr/bin/env python3
"""
Attention Heatmap Generation Script

Generates attention heatmap overlays for WSIs using a trained MIL model.

Usage:
    # Single slide
    python scripts/generate_heatmap.py \
        --wsi_path /data/camelyon16/testing/images/test_001.tif \
        --checkpoint results/clam_sb_camelyon16/best_model.pt \
        --feature_extractor ctranspath \
        --output_dir results/heatmaps \
        --alpha 0.4

    # Batch processing from directory
    python scripts/generate_heatmap.py \
        --wsi_dir /data/camelyon16/testing/images \
        --checkpoint results/clam_sb_camelyon16/best_model.pt \
        --feature_extractor ctranspath \
        --output_dir results/heatmaps \
        --batch

    # Compare with pathologist annotations (compute IoU)
    python scripts/generate_heatmap.py \
        --wsi_path /data/camelyon16/testing/images/test_001.tif \
        --checkpoint results/clam_sb_camelyon16/best_model.pt \
        --annotation_dir /data/camelyon16/testing/masks \
        --output_dir results/heatmaps \
        --compute_iou

Output files per slide:
    {slide_id}_heatmap.png           - Attention overlay on thumbnail
    {slide_id}_attention_map.npy     - Raw attention map
    {slide_id}_top_tiles.png         - Grid of top-K tiles
    {slide_id}_result.json           - Prediction and metadata
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps for WSIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--wsi_path", type=str, help="Single WSI path")
    input_group.add_argument("--wsi_dir", type=str, help="Directory of WSIs (batch mode)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature_extractor", type=str, default="ctranspath",
                        choices=["resnet50", "ctranspath", "uni", "conch"])
    parser.add_argument("--extractor_weights", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None,
                        help="Directory with .xml annotation files")

    # Tile extraction
    parser.add_argument("--magnification", type=float, default=20.0)
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--stain_normalize", type=str, default="macenko",
                        choices=["macenko", "reinhard", "none"])

    # Heatmap settings
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Heatmap opacity (0=original, 1=heatmap only)")
    parser.add_argument("--colormap", type=str, default="jet",
                        choices=["jet", "hot", "viridis", "plasma", "inferno"])
    parser.add_argument("--map_resolution", type=int, default=1000,
                        help="Heatmap output resolution in pixels")
    parser.add_argument("--top_k_tiles", type=int, default=20,
                        help="Number of top attention tiles to visualize")

    # Metrics
    parser.add_argument("--compute_iou", action="store_true",
                        help="Compute IoU with pathologist annotations")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Feature cache directory")

    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def process_single_slide(
    wsi_path: Path,
    model,
    feature_extractor,
    args,
    output_dir: Path,
    device: torch.device,
    annotation_parser=None,
) -> dict:
    """Process a single slide and return results."""
    from src.data.wsi_dataset import WSIReader
    from src.data.tile_processing import TileProcessor
    from src.models.feature_extractor import TileDataset, get_imagenet_transform
    from src.evaluation.heatmap_generator import (
        coords_to_spatial_map,
        apply_gaussian_smoothing,
        normalize_attention_map,
        overlay_heatmap_on_thumbnail,
        visualize_top_tiles,
    )
    from src.evaluation.pathology_metrics import compute_attention_iou_at_thresholds
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import cv2
    from PIL import Image

    slide_id = wsi_path.stem
    logger.info("Processing: %s", slide_id)

    # Extract tiles
    reader = WSIReader(wsi_path, target_magnification=args.magnification, tile_size=args.tile_size)
    processor = TileProcessor.default(normalize=args.stain_normalize)

    tiles = []
    coords_list = []
    raw_tiles = []  # Keep originals for visualization

    for tile, coord in reader.iter_tiles():
        raw_tiles.append(tile)
        processed = processor.process(tile, augment=False)
        if processed is not None:
            tiles.append(processed)
            coords_list.append(coord)

    if not tiles:
        logger.warning("No tiles in %s", slide_id)
        reader.close()
        return {"slide_id": slide_id, "error": "No tissue tiles"}

    coords_arr = np.array(coords_list, dtype=np.float32)

    # Feature extraction
    transform = getattr(feature_extractor, "transform", get_imagenet_transform())
    dataset = TileDataset(tiles, transform=transform)
    loader = DataLoader(dataset, batch_size=256, num_workers=0)

    features_list = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = feature_extractor(batch)
            features_list.append(feats.cpu())

    features = torch.cat(features_list, dim=0).to(device)
    coords_tensor = torch.from_numpy(coords_arr).to(device)

    # MIL forward
    model.eval()
    with torch.no_grad():
        model_name = type(model).__name__.lower()

        if "clam" in model_name:
            logits, _, attention = model(features, label=None, instance_eval=False)
            if attention.dim() > 2:
                attention = attention[1]
            attention_scores = attention.squeeze().cpu().numpy()
        elif "transmil" in model_name:
            logits, _ = model(features, coords=coords_tensor)
            tile_reps = model.get_tile_representations(features, coords=coords_tensor)
            attention_scores = torch.norm(tile_reps, dim=1).cpu().numpy()
            logits, _ = model(features, coords=coords_tensor)
        else:
            logits, attn = model(features, return_attention=True)
            attention_scores = attn.squeeze().cpu().numpy() if attn is not None else np.ones(len(tiles))

        probs = F.softmax(logits.cpu(), dim=-1).numpy()

    pred_class = int(probs.argmax())
    pred_prob = float(probs.max())

    # Build heatmap
    tile_size_l0 = int(args.tile_size * reader.level_downsample)
    map_res = (args.map_resolution, args.map_resolution)

    heatmap = coords_to_spatial_map(
        coords=coords_arr,
        scores=attention_scores,
        tile_size_l0=tile_size_l0,
        slide_dims_l0=reader.slide.dimensions,
        map_resolution=map_res,
    )
    heatmap = apply_gaussian_smoothing(heatmap, sigma=20.0)
    heatmap_norm = normalize_attention_map(heatmap)

    # Thumbnail overlay
    thumbnail = np.array(reader.get_thumbnail(size=args.map_resolution))
    thumbnail_resized = cv2.resize(thumbnail, (map_res[1], map_res[0]))
    tissue_mask = cv2.resize(reader.tissue_mask, (map_res[1], map_res[0]))

    overlay = overlay_heatmap_on_thumbnail(
        thumbnail_resized, heatmap_norm,
        alpha=args.alpha, colormap=args.colormap,
        mask=tissue_mask,
    )

    # Save outputs
    overlay_path = output_dir / f"{slide_id}_heatmap.png"
    Image.fromarray(overlay).save(str(overlay_path))

    np.save(str(output_dir / f"{slide_id}_attention_map.npy"), heatmap_norm)

    # Top-K tile visualization
    try:
        top_grid = visualize_top_tiles(
            reader, coords_arr, attention_scores, k=args.top_k_tiles
        )
        top_tiles_path = output_dir / f"{slide_id}_top_tiles.png"
        top_grid.save(str(top_tiles_path))
    except Exception as e:
        logger.warning("Top tiles visualization failed: %s", e)

    result = {
        "slide_id": slide_id,
        "predicted_class": pred_class,
        "probabilities": probs.tolist(),
        "confidence": pred_prob,
        "n_tiles": len(tiles),
        "heatmap_path": str(overlay_path),
    }

    # Compute IoU with annotations if requested
    if args.compute_iou and annotation_parser is not None:
        try:
            annotation_mask = annotation_parser.get_annotation_mask(
                slide_id, reader.slide.dimensions, 1.0
            )
            # Resize heatmap to level-0 scale for comparison
            mask_resized = cv2.resize(
                annotation_mask,
                (args.map_resolution, args.map_resolution)
            )
            iou_metrics = compute_attention_iou_at_thresholds(heatmap_norm, mask_resized)
            result["iou_metrics"] = iou_metrics
            logger.info("  IoU@0.5: %.4f, Max IoU: %.4f",
                        iou_metrics.get("iou@0.5", 0), iou_metrics.get("max_iou", 0))
        except Exception as e:
            logger.warning("IoU computation failed: %s", e)

    # Save result JSON
    with open(output_dir / f"{slide_id}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    reader.close()
    logger.info("  Saved: %s | Class %d (p=%.3f)", overlay_path.name, pred_class, pred_prob)
    return result


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Load model
    from src.inference.slide_classifier import load_model_from_checkpoint
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Load feature extractor
    from src.models.feature_extractor import build_feature_extractor
    feature_extractor = build_feature_extractor(
        args.feature_extractor,
        weights_path=args.extractor_weights,
        device=device,
    )

    # Annotation parser
    annotation_parser = None
    if args.annotation_dir:
        from src.data.camelyon_dataset import TumorAnnotationParser
        annotation_parser = TumorAnnotationParser(args.annotation_dir)

    # Gather WSI paths
    if args.wsi_path:
        wsi_paths = [Path(args.wsi_path)]
    else:
        wsi_dir = Path(args.wsi_dir)
        wsi_paths = []
        for ext in [".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]:
            wsi_paths.extend(wsi_dir.glob(f"*{ext}"))
        wsi_paths = sorted(wsi_paths)
        logger.info("Found %d WSI files in %s", len(wsi_paths), wsi_dir)

    # Process slides
    all_results = []
    for i, wsi_path in enumerate(wsi_paths, 1):
        logger.info("[%d/%d] %s", i, len(wsi_paths), wsi_path.name)
        try:
            result = process_single_slide(
                wsi_path, model, feature_extractor,
                args, output_dir, device, annotation_parser
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Failed: %s — %s", wsi_path.name, e)
            all_results.append({"slide_id": wsi_path.stem, "error": str(e)})

    # Summary
    with open(output_dir / "batch_heatmap_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_success = sum(1 for r in all_results if "error" not in r)
    logger.info("Done: %d/%d slides processed. Output: %s", n_success, len(all_results), output_dir)

    # Print IoU summary if computed
    if args.compute_iou:
        iou_values = [
            r["iou_metrics"]["max_iou"]
            for r in all_results
            if "iou_metrics" in r and "max_iou" in r["iou_metrics"]
        ]
        if iou_values:
            logger.info("Mean Max IoU: %.4f ± %.4f", np.mean(iou_values), np.std(iou_values))


if __name__ == "__main__":
    main()
