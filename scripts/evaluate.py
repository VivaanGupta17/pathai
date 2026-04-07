#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates a trained MIL model on a test set.

Usage:
    # Evaluate on Camelyon16 test set
    python scripts/evaluate.py \
        --config configs/camelyon_config.yaml \
        --checkpoint results/clam_sb_camelyon16/best_model.pt \
        --feature_dir data/features/ctranspath_20x \
        --output_dir results/clam_sb_eval \
        --compute_froc

    # Evaluate and save attention heatmaps for all slides
    python scripts/evaluate.py \
        --config configs/camelyon_config.yaml \
        --checkpoint results/clam_sb_camelyon16/best_model.pt \
        --feature_dir data/features/ctranspath_20x \
        --output_dir results/clam_sb_eval \
        --save_attention \
        --compute_froc

    # Evaluate on arbitrary slide list
    python scripts/evaluate.py \
        --checkpoint results/clam_sb_camelyon16/best_model.pt \
        --feature_dir data/features/ctranspath_20x \
        --label_csv data/test_labels.csv \
        --output_dir results/eval
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MIL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/camelyon_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Directory with .pt feature files")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Camelyon16 data directory (for test labels)")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="CSV file with slide_id,label columns")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--compute_froc", action="store_true",
                        help="Compute FROC metrics (requires annotation masks)")
    parser.add_argument("--save_attention", action="store_true",
                        help="Save per-slide attention scores as .npy files")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--class_names", nargs="+", default=["Normal", "Tumor"])
    return parser.parse_args()


def load_labels_from_csv(csv_path: str) -> dict:
    """Load slide_id → label mapping from CSV."""
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support various column naming conventions
            slide_id = row.get("slide_id") or row.get("name") or row.get("slide", "")
            label = int(row.get("label") or row.get("tumor") or row.get("class", 0))
            if slide_id:
                labels[slide_id] = label
    return labels


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.feature_dir:
        config["data"]["feature_dir"] = args.feature_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Load model
    from src.inference.slide_classifier import load_model_from_checkpoint
    model, ckpt_config = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()

    # Load labels
    if args.label_csv:
        label_dict = load_labels_from_csv(args.label_csv)
    elif args.data_dir:
        from src.data.camelyon_dataset import parse_camelyon16_labels, get_camelyon16_train_test_split
        data_dir = Path(args.data_dir)
        label_dict = parse_camelyon16_labels(data_dir)
        _, test_ids = get_camelyon16_train_test_split(label_dict, data_dir)
        label_dict = {sid: label_dict[sid] for sid in test_ids}
    else:
        logger.warning("No labels provided. Using all slides in feature_dir with label 0.")
        feature_dir = Path(config["data"]["feature_dir"])
        extractor_name = config.get("feature_extractor", {}).get("name", "ctranspath")
        label_dict = {}
        for pt_file in feature_dir.glob(f"*_{extractor_name}.pt"):
            slide_id = pt_file.stem.rsplit(f"_{extractor_name}", 1)[0]
            label_dict[slide_id] = 0

    # Build evaluation dataset
    from src.data.wsi_dataset import WSIBagDataset
    from torch.utils.data import DataLoader

    eval_dataset = WSIBagDataset(
        feature_dir=config["data"]["feature_dir"],
        label_dict=label_dict,
        max_bag_size=None,
        shuffle=False,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info("Evaluating on %d slides", len(eval_dataset))

    # Run inference
    all_labels = []
    all_probs = []
    all_preds = []
    slide_results = []

    with torch.no_grad():
        for batch in eval_loader:
            features, coords, labels = batch
            features = features.squeeze(0).to(device)
            label_val = int(labels.squeeze().item())

            if isinstance(coords, torch.Tensor) and coords.numel() > 0:
                coords_dev = coords.squeeze(0).to(device)
            else:
                coords_dev = None

            model_name = type(model).__name__.lower()

            if "clam" in model_name:
                logits, _, attention = model(features, label=None, instance_eval=False)
                if attention.dim() > 2:
                    attention = attention[1]
                attn_scores = attention.squeeze().cpu().numpy()
            elif "transmil" in model_name:
                logits, _ = model(features, coords=coords_dev)
                attn_scores = None
            else:
                logits, attn = model(features, return_attention=True)
                attn_scores = attn.squeeze().cpu().numpy() if attn is not None else None

            probs = F.softmax(logits.cpu(), dim=-1).numpy()
            pred = int(probs.argmax())
            prob_pos = float(probs[1]) if len(probs) > 1 else float(probs[0])

            all_labels.append(label_val)
            all_probs.append(prob_pos)
            all_preds.append(pred)

            # Get slide_id
            slide_idx = len(slide_results)
            slide_id = eval_dataset.slide_ids[slide_idx] if slide_idx < len(eval_dataset.slide_ids) else str(slide_idx)

            result = {
                "slide_id": slide_id,
                "label": label_val,
                "pred": pred,
                "prob_tumor": round(prob_pos, 6),
                "correct": int(pred == label_val),
            }
            slide_results.append(result)

            # Save attention scores
            if args.save_attention and attn_scores is not None:
                attn_path = output_dir / f"{slide_id}_attention.npy"
                np.save(str(attn_path), attn_scores)

    # Compute metrics
    from src.evaluation.pathology_metrics import (
        compute_slide_metrics,
        compute_confusion_matrix,
        generate_evaluation_report,
        compute_ece,
    )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_slide_metrics(y_true, y_pred, y_prob, class_names=args.class_names)
    ece = compute_ece(y_true, y_prob)
    metrics["ece"] = ece

    # Log metrics
    logger.info("\n--- Evaluation Results ---")
    logger.info("  AUROC:      %.4f", metrics.get("auroc", 0))
    logger.info("  Accuracy:   %.4f", metrics.get("accuracy", 0))
    logger.info("  F1:         %.4f", metrics.get("f1", 0))
    logger.info("  Sensitivity:%.4f", metrics.get("sensitivity", 0))
    logger.info("  Specificity:%.4f", metrics.get("specificity", 0))
    logger.info("  ECE:        %.4f", metrics.get("ece", 0))

    # Generate report
    y_prob_2d = np.stack([1 - y_prob, y_prob], axis=1)
    report = generate_evaluation_report(
        y_true, y_pred, y_prob_2d,
        class_names=args.class_names,
        task=f"Camelyon16 Evaluation ({type(model).__name__})",
    )
    print(report)

    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(report)

    # Save detailed metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-slide predictions
    pred_csv = output_dir / "predictions.csv"
    with open(pred_csv, "w", newline="") as f:
        if slide_results:
            writer = csv.DictWriter(f, fieldnames=slide_results[0].keys())
            writer.writeheader()
            writer.writerows(slide_results)
    logger.info("Predictions saved to %s", pred_csv)

    # Save confusion matrix
    cm, cm_names = compute_confusion_matrix(y_true, y_pred, args.class_names)
    try:
        from src.evaluation.pathology_metrics import plot_confusion_matrix
        plot_confusion_matrix(
            cm, cm_names,
            output_path=str(output_dir / "confusion_matrix.png"),
            title=f"Confusion Matrix ({type(model).__name__})",
        )
    except Exception as e:
        logger.warning("Could not plot confusion matrix: %s", e)

    logger.info("Evaluation complete. Results in: %s", output_dir)
    return metrics


if __name__ == "__main__":
    main()
