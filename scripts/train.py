#!/usr/bin/env python3
"""
MIL Model Training Script

Trains CLAM-SB, CLAM-MB, TransMIL, or ABMIL on pre-extracted WSI features.

Usage:
    # Train CLAM-SB on Camelyon16 with CTransPath features
    python scripts/train.py \
        --config configs/camelyon_config.yaml \
        --model clam_sb \
        --feature_dir data/features/ctranspath_20x \
        --output_dir results/clam_sb_camelyon16 \
        --gpu 0

    # Train TransMIL
    python scripts/train.py \
        --config configs/camelyon_config.yaml \
        --model transmil \
        --feature_dir data/features/ctranspath_20x \
        --output_dir results/transmil_camelyon16 \
        --lr 1e-4 \
        --n_epochs 30 \
        --gpu 0

    # Cross-validation
    python scripts/train.py \
        --config configs/camelyon_config.yaml \
        --feature_dir data/features/ctranspath_20x \
        --output_dir results/cv_clam_sb \
        --cross_val 5 \
        --gpu 0

    # Resume from checkpoint
    python scripts/train.py \
        --config configs/camelyon_config.yaml \
        --feature_dir data/features/ctranspath_20x \
        --output_dir results/clam_sb_camelyon16 \
        --resume results/clam_sb_camelyon16/checkpoint_epoch010.pt \
        --gpu 0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MIL model on WSI features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/camelyon_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None,
                        choices=["clam_sb", "clam_mb", "transmil", "abmil"],
                        help="Override model from config")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Override feature directory from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--bag_loss", type=str, default=None, choices=["ce", "focal"])
    parser.add_argument("--cross_val", type=int, default=None,
                        help="Number of CV folds (None = use fixed train/test split)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (-1 for CPU)")
    return parser.parse_args()


def build_model(config: dict) -> torch.nn.Module:
    """Build MIL model from config."""
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "clam_sb").lower()

    if model_name == "clam_sb":
        from src.models.clam import CLAM_SB
        return CLAM_SB(
            gate=model_cfg.get("gate", True),
            size_arg=model_cfg.get("size_arg", "small"),
            dropout=model_cfg.get("dropout", True),
            k_sample=model_cfg.get("k_sample", 8),
            num_classes=model_cfg.get("num_classes", 2),
            instance_loss_fn=model_cfg.get("instance_loss_fn", "svm"),
            subtyping=model_cfg.get("subtyping", False),
        )
    elif model_name == "clam_mb":
        from src.models.clam import CLAM_MB
        return CLAM_MB(
            gate=model_cfg.get("gate", True),
            size_arg=model_cfg.get("size_arg", "small"),
            dropout=model_cfg.get("dropout", True),
            k_sample=model_cfg.get("k_sample", 8),
            num_classes=model_cfg.get("num_classes", 2),
            instance_loss_fn=model_cfg.get("instance_loss_fn", "svm"),
        )
    elif model_name == "transmil":
        from src.models.transmil import TransMIL
        t_cfg = model_cfg.get("transmil", {})
        return TransMIL(
            input_dim=t_cfg.get("input_dim", model_cfg.get("input_dim", 1024)),
            num_classes=model_cfg.get("num_classes", 2),
            dim=t_cfg.get("dim", 512),
            num_layers=t_cfg.get("num_layers", 2),
            num_heads=t_cfg.get("num_heads", 8),
            mlp_dim=t_cfg.get("mlp_dim", 512),
            num_landmarks=t_cfg.get("num_landmarks", 256),
            use_nystrom=t_cfg.get("use_nystrom", True),
            use_pos_enc=t_cfg.get("use_pos_enc", True),
            pos_enc_type=t_cfg.get("pos_enc_type", "morphology"),
        )
    elif model_name == "abmil":
        from src.models.attention_mil import ABMIL
        a_cfg = model_cfg.get("abmil", {})
        return ABMIL(
            input_dim=a_cfg.get("input_dim", 1024),
            hidden_dim=a_cfg.get("hidden_dim", 512),
            attention_dim=a_cfg.get("attention_dim", 256),
            num_classes=model_cfg.get("num_classes", 2),
            gated=a_cfg.get("gated", True),
            dropout=model_cfg.get("dropout", 0.25),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_training(
    config: dict,
    train_ids: list,
    val_ids: list,
    label_dict: dict,
    output_dir: Path,
    device: torch.device,
    resume_path: str = None,
) -> dict:
    """Run a single training run."""
    from torch.utils.data import DataLoader
    from src.data.wsi_dataset import WSIBagDataset
    from src.training.mil_trainer import MILTrainer, build_weighted_sampler, set_seed

    set_seed(config.get("experiment", {}).get("seed", 42))

    feature_dir = config["data"]["feature_dir"]

    # Build datasets
    train_dataset = WSIBagDataset(
        feature_dir=feature_dir,
        label_dict={sid: label_dict[sid] for sid in train_ids if sid in label_dict},
        max_bag_size=config["data"].get("max_bag_size", None),
        min_bag_size=config["data"].get("min_bag_size", 16),
        shuffle=True,
    )
    val_dataset = WSIBagDataset(
        feature_dir=feature_dir,
        label_dict={sid: label_dict[sid] for sid in val_ids if sid in label_dict},
        max_bag_size=None,
        min_bag_size=config["data"].get("min_bag_size", 16),
        shuffle=False,
    )

    logger.info("Train: %d slides | Val: %d slides", len(train_dataset), len(val_dataset))

    # Sampler
    train_labels = [train_dataset.label_dict[sid] for sid in train_dataset.slide_ids]
    sampler = build_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    # Build model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s (%d trainable params)", type(model).__name__, n_params)

    # Build trainer
    train_config = config.get("training", {})
    trainer = MILTrainer(
        model=model,
        config=train_config,
        output_dir=str(output_dir),
        device=str(device),
    )

    # Resume if requested
    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path)
        logger.info("Resuming from epoch %d", start_epoch)

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=train_config.get("n_epochs", 20),
    )

    return history


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found at %s; using defaults", config_path)
        config = {}

    # Apply CLI overrides
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.feature_dir:
        config.setdefault("data", {})["feature_dir"] = args.feature_dir
    if args.n_epochs:
        config.setdefault("training", {})["n_epochs"] = args.n_epochs
    if args.lr:
        config.setdefault("training", {})["lr"] = args.lr
    if args.bag_loss:
        config.setdefault("training", {})["bag_loss"] = args.bag_loss

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_name = config.get("experiment", {}).get("name", "experiment")
        output_dir = Path("results") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load labels
    from src.data.camelyon_dataset import parse_camelyon16_labels, get_camelyon16_train_test_split

    data_dir = Path(config["data"].get("data_dir", "/data/camelyon16"))
    if not data_dir.exists():
        logger.warning("Data dir %s not found; using placeholder labels", data_dir)
        # Create dummy labels for demo
        feature_dir = Path(config["data"].get("feature_dir", "data/features"))
        label_dict = {}
        if feature_dir.exists():
            for pt_file in feature_dir.glob("*.pt"):
                slide_id = pt_file.stem.rsplit("_", 1)[0]
                label_dict[slide_id] = 0  # Placeholder
        if not label_dict:
            logger.error("No feature files found and no data directory. Exiting.")
            sys.exit(1)
        train_ids = list(label_dict.keys())
        n_val = max(1, len(train_ids) // 5)
        val_ids = train_ids[-n_val:]
        train_ids = train_ids[:-n_val]
    else:
        label_dict = parse_camelyon16_labels(data_dir)
        train_ids, test_ids = get_camelyon16_train_test_split(label_dict, data_dir)
        # Use 20% of training as validation
        n_val = max(1, len(train_ids) // 5)
        val_ids = train_ids[-n_val:]
        train_ids = train_ids[:-n_val]

    # Cross-validation or single run
    if args.cross_val:
        from src.data.camelyon_dataset import create_cv_splits
        all_ids = train_ids + val_ids
        splits = create_cv_splits(all_ids, label_dict, n_splits=args.cross_val, seed=args.seed)

        all_histories = []
        for fold, (fold_train, fold_val) in enumerate(splits, 1):
            logger.info("=" * 50)
            logger.info("FOLD %d/%d", fold, args.cross_val)
            fold_dir = output_dir / f"fold_{fold}"
            history = run_training(config, fold_train, fold_val, label_dict, fold_dir, device)
            all_histories.append(history)

        # Save CV summary
        with open(output_dir / "cv_history.json", "w") as f:
            json.dump(all_histories, f, indent=2)
        logger.info("Cross-validation complete. Results in %s", output_dir)
    else:
        history = run_training(
            config, train_ids, val_ids, label_dict,
            output_dir, device, resume_path=args.resume
        )
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Training complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
