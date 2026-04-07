"""
Whole Slide Image Inference Pipeline

Provides end-to-end inference for single WSIs or batches:
    1. Load trained MIL model from checkpoint
    2. Extract tiles with tissue detection
    3. Apply stain normalization
    4. Compute tile features (cached)
    5. MIL forward pass → slide-level prediction
    6. Generate attention heatmap
    7. Extract top-K representative tiles
    8. Generate JSON/PDF report

Usage (CLI):
    python src/inference/slide_classifier.py \
        --wsi path/to/slide.svs \
        --checkpoint results/best_model.pt \
        --output results/inference/

Usage (Python API):
    classifier = SlideClassifier.from_checkpoint("best_model.pt")
    result = classifier.classify("slide.svs")
    print(result["predicted_class"], result["probabilities"])
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> tuple:
    """
    Load MIL model from checkpoint.

    The checkpoint contains:
        - 'model_state': Model state dict
        - 'config':      Training configuration (model architecture params)

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device:          Inference device

    Returns:
        (model, config): Loaded model in eval mode and config dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = config.get("model", {}).get("name", "clam_sb").lower()

    # Build model from config
    if "clam_sb" in model_name:
        from src.models.clam import CLAM_SB
        model = CLAM_SB(
            gate=config.get("model", {}).get("gate", True),
            size_arg=config.get("model", {}).get("size_arg", "small"),
            dropout=config.get("model", {}).get("dropout", True),
            num_classes=config.get("model", {}).get("num_classes", 2),
        )
    elif "clam_mb" in model_name:
        from src.models.clam import CLAM_MB
        model = CLAM_MB(
            gate=config.get("model", {}).get("gate", True),
            size_arg=config.get("model", {}).get("size_arg", "small"),
            dropout=config.get("model", {}).get("dropout", True),
            num_classes=config.get("model", {}).get("num_classes", 2),
        )
    elif "transmil" in model_name:
        from src.models.transmil import TransMIL
        m_cfg = config.get("model", {})
        model = TransMIL(
            input_dim=m_cfg.get("input_dim", 1024),
            num_classes=m_cfg.get("num_classes", 2),
            dim=m_cfg.get("dim", 512),
            num_layers=m_cfg.get("num_layers", 2),
            num_heads=m_cfg.get("num_heads", 8),
        )
    elif "abmil" in model_name:
        from src.models.attention_mil import ABMIL
        m_cfg = config.get("model", {})
        model = ABMIL(
            input_dim=m_cfg.get("input_dim", 1024),
            hidden_dim=m_cfg.get("hidden_dim", 512),
            num_classes=m_cfg.get("num_classes", 2),
            gated=m_cfg.get("gated", True),
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_name}")

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    metrics = checkpoint.get("metrics", {})
    auroc = metrics.get("auroc", "unknown")

    logger.info(
        "Loaded %s (epoch %s, val AUROC: %s) from %s",
        type(model).__name__, epoch, auroc, checkpoint_path,
    )
    return model, config


# ---------------------------------------------------------------------------
# Slide Classifier
# ---------------------------------------------------------------------------


class SlideClassifier:
    """
    End-to-end WSI classifier for production inference.

    Handles the complete pipeline from raw WSI to prediction + heatmap.
    Designed for single-slide and batch inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        config: dict,
        class_names: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu",
        magnification: float = 20.0,
        tile_size: int = 224,
        stain_normalize: str = "macenko",
        generate_heatmaps: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            model:             Trained MIL model
            feature_extractor: Tile feature extractor
            config:            Configuration dict
            class_names:       Class labels ['Normal', 'Tumor']
            device:            Inference device
            magnification:     WSI extraction magnification
            tile_size:         Tile size in pixels
            stain_normalize:   Normalization method ('macenko', 'reinhard', 'none')
            generate_heatmaps: Whether to generate attention heatmaps
            cache_dir:         Directory for feature caching
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        self.class_names = class_names or [f"Class_{i}" for i in range(
            config.get("model", {}).get("num_classes", 2)
        )]
        self.device = torch.device(device)
        self.magnification = magnification
        self.tile_size = tile_size
        self.stain_normalize = stain_normalize
        self.generate_heatmaps = generate_heatmaps
        self.cache_dir = cache_dir

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        feature_extractor_name: str = "ctranspath",
        feature_extractor_weights: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> "SlideClassifier":
        """
        Convenience constructor: load everything from a checkpoint path.

        Args:
            checkpoint_path:           Path to model checkpoint
            feature_extractor_name:    'resnet50', 'ctranspath', 'uni', 'conch'
            feature_extractor_weights: Path to feature extractor weights
            class_names:               Class labels
            device:                    Inference device

        Returns:
            Configured SlideClassifier
        """
        from src.models.feature_extractor import build_feature_extractor

        model, config = load_model_from_checkpoint(checkpoint_path, device)
        extractor_name = feature_extractor_name or config.get("data", {}).get("feature_extractor", "ctranspath")

        feature_extractor = build_feature_extractor(
            extractor_name,
            weights_path=feature_extractor_weights,
            device=device,
        )

        # Read class names from config if not provided
        if class_names is None:
            class_names = config.get("data", {}).get("class_names", None)

        return cls(
            model=model,
            feature_extractor=feature_extractor,
            config=config,
            class_names=class_names,
            device=device,
            **kwargs,
        )

    def classify(
        self,
        wsi_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        slide_id: Optional[str] = None,
    ) -> Dict:
        """
        Classify a single WSI.

        Args:
            wsi_path:   Path to WSI file
            output_dir: If set, save heatmap and report here
            slide_id:   Slide identifier override

        Returns:
            Dict with:
                slide_id, predicted_class, class_name,
                probabilities, n_tiles, processing_time_sec,
                heatmap_path (if output_dir set),
                top_tiles (coords of top-K attention tiles)
        """
        from src.data.wsi_dataset import WSIReader
        from src.data.tile_processing import TileProcessor

        t0 = time.time()
        wsi_path = Path(wsi_path)
        slide_id = slide_id or wsi_path.stem

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Classifying: %s", slide_id)

        # --- Tile extraction ---
        reader = WSIReader(
            wsi_path,
            target_magnification=self.magnification,
            tile_size=self.tile_size,
        )
        processor = TileProcessor.default(normalize=self.stain_normalize)

        tiles = []
        coords_list = []

        for tile, coord in reader.iter_tiles():
            processed = processor.process(tile, augment=False)
            if processed is not None:
                tiles.append(processed)
                coords_list.append(coord)

        if not tiles:
            reader.close()
            return {
                "slide_id": slide_id,
                "error": "No tissue tiles found",
                "predicted_class": -1,
                "probabilities": [],
            }

        coords_arr = np.array(coords_list, dtype=np.float32)
        logger.info("  Tiles: %d", len(tiles))

        # --- Feature extraction ---
        features, coords_tensor = self._extract_features(tiles, coords_arr, slide_id)

        # --- MIL inference ---
        probs, attention_scores = self._mil_forward(features, coords_tensor)

        # --- Build result ---
        pred_class = int(np.argmax(probs))
        pred_class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else str(pred_class)

        t_elapsed = time.time() - t0

        result = {
            "slide_id": slide_id,
            "predicted_class": pred_class,
            "class_name": pred_class_name,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
            "confidence": float(probs[pred_class]),
            "n_tiles": len(tiles),
            "processing_time_sec": round(t_elapsed, 2),
        }

        # --- Top-K tiles ---
        if attention_scores is not None:
            top_k = min(20, len(attention_scores))
            top_idx = np.argsort(attention_scores)[-top_k:][::-1]
            result["top_tile_coords"] = coords_arr[top_idx].tolist()
            result["top_tile_scores"] = attention_scores[top_idx].tolist()

        # --- Heatmap generation ---
        if self.generate_heatmaps and output_dir and attention_scores is not None:
            heatmap_path = self._generate_and_save_heatmap(
                reader, coords_arr, attention_scores, output_dir, slide_id
            )
            result["heatmap_path"] = str(heatmap_path)

        # --- Save report ---
        if output_dir:
            report_path = output_dir / f"{slide_id}_result.json"
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("  Report saved to %s", report_path)

        reader.close()

        logger.info(
            "  Result: %s (prob=%.3f) in %.1fs",
            pred_class_name,
            probs[pred_class],
            t_elapsed,
        )
        return result

    def _extract_features(
        self,
        tiles: list,
        coords_arr: np.ndarray,
        slide_id: str,
    ) -> tuple:
        """Extract or load cached features."""
        from src.models.feature_extractor import FeatureExtractorPipeline

        extractor_name = getattr(self.feature_extractor, "extractor_name", "unknown")
        pipeline = FeatureExtractorPipeline(
            extractor=self.feature_extractor,
            extractor_name=extractor_name,
            batch_size=256,
            num_workers=0,
            cache_dir=self.cache_dir,
            device=self.device,
        )
        features, coords_t = pipeline.extract(tiles, coords=coords_arr, slide_id=slide_id)
        return features.to(self.device), (coords_t.to(self.device) if coords_t is not None else None)

    def _mil_forward(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Run MIL model forward pass and return (probs, attention_scores)."""
        self.model.eval()

        with torch.no_grad():
            model_name = type(self.model).__name__.lower()

            if "clam" in model_name:
                logits, _, attention = self.model(features, label=None, instance_eval=False)
                if attention.dim() > 2:
                    attention = attention[1]  # class-1 branch (tumor) for CLAM-MB
                attention_scores = attention.squeeze().cpu().numpy()
            elif "transmil" in model_name:
                logits, _ = self.model(features, coords=coords)
                tile_reps = self.model.get_tile_representations(features, coords=coords)
                attention_scores = torch.norm(tile_reps, dim=1).cpu().numpy()
            else:
                logits, attention = self.model(features, return_attention=True)
                attention_scores = attention.squeeze().cpu().numpy() if attention is not None else None

            probs = F.softmax(logits.cpu(), dim=-1).numpy()

        return probs, attention_scores

    def _generate_and_save_heatmap(
        self,
        reader,
        coords_arr: np.ndarray,
        attention_scores: np.ndarray,
        output_dir: Path,
        slide_id: str,
    ) -> Path:
        """Generate and save attention heatmap."""
        from src.evaluation.heatmap_generator import (
            coords_to_spatial_map,
            apply_gaussian_smoothing,
            normalize_attention_map,
            overlay_heatmap_on_thumbnail,
        )
        import cv2

        tile_size_l0 = int(self.tile_size * reader.level_downsample)
        map_res = (1000, 1000)

        heatmap = coords_to_spatial_map(
            coords=coords_arr,
            scores=attention_scores,
            tile_size_l0=tile_size_l0,
            slide_dims_l0=reader.slide.dimensions,
            map_resolution=map_res,
        )
        heatmap = apply_gaussian_smoothing(heatmap, sigma=20.0)
        heatmap_norm = normalize_attention_map(heatmap)

        thumbnail = np.array(reader.get_thumbnail(size=1000))
        thumbnail_resized = cv2.resize(thumbnail, (map_res[1], map_res[0]))
        tissue_mask = cv2.resize(reader.tissue_mask, (map_res[1], map_res[0]))

        overlay = overlay_heatmap_on_thumbnail(
            thumbnail_resized, heatmap_norm, alpha=0.4, mask=tissue_mask
        )

        heatmap_path = output_dir / f"{slide_id}_heatmap.png"
        from PIL import Image as PILImage
        PILImage.fromarray(overlay).save(str(heatmap_path))

        return heatmap_path

    def classify_batch(
        self,
        wsi_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        n_workers: int = 1,
    ) -> List[Dict]:
        """
        Classify a batch of slides.

        Args:
            wsi_paths:  List of WSI file paths
            output_dir: Directory for saving results
            n_workers:  Number of parallel processes (1 = sequential)

        Returns:
            List of result dicts
        """
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Batch inference: %d slides", len(wsi_paths))

        for i, wsi_path in enumerate(wsi_paths, 1):
            logger.info("[%d/%d] %s", i, len(wsi_paths), Path(wsi_path).name)
            try:
                result = self.classify(wsi_path, output_dir=output_dir)
                results.append(result)
            except Exception as e:
                logger.error("Failed to classify %s: %s", wsi_path, e)
                results.append({
                    "slide_id": Path(wsi_path).stem,
                    "error": str(e),
                    "predicted_class": -1,
                })

        # Save batch summary
        summary_path = output_dir / "batch_results.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Batch summary saved to %s", summary_path)

        return results

    def generate_report(
        self,
        result: Dict,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate a human-readable inference report.

        Args:
            result:      Result dict from classify()
            output_path: If set, save report to this file

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "PathAI Inference Report",
            "=" * 60,
            f"Slide ID:          {result.get('slide_id', 'N/A')}",
            f"Predicted Class:   {result.get('class_name', 'N/A')}",
            f"Confidence:        {result.get('confidence', 0):.4f}",
            "",
            "--- Class Probabilities ---",
        ]

        probs = result.get("probabilities", {})
        for cls_name, prob in probs.items():
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {cls_name:15} {bar} {prob:.4f}")

        lines += [
            "",
            f"Tiles processed:   {result.get('n_tiles', 0)}",
            f"Processing time:   {result.get('processing_time_sec', 0):.1f}s",
        ]

        if "heatmap_path" in result:
            lines.append(f"Heatmap:           {result['heatmap_path']}")

        if "error" in result:
            lines += ["", f"ERROR: {result['error']}"]

        lines.append("=" * 60)
        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report
