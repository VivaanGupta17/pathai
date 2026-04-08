"""
MIL Training Engine for Computational Pathology

Features:
    - Cross-entropy + CLAM instance clustering loss
    - Class-weighted sampling for imbalanced datasets
    - Mixed precision training (torch.cuda.amp)
    - Gradient accumulation for large effective batch sizes
    - Learning rate scheduling (cosine, step, OneCycleLR)
    - Early stopping with patience
    - Checkpoint management (save best + periodic)
    - TensorBoard logging
    - Reproducible training with seed management

MIL training note: Each "batch" is a single WSI bag (N tiles × D features).
True batch size = 1 slide, but gradient accumulation simulates larger batches.
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance in pathology datasets.
    Down-weights easy negatives and focuses on hard cases.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] or [C] — raw class scores
            targets: [B] or [] — integer class labels
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        p_t = torch.exp(-ce_loss)
        focal = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(logits.device)
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class SurvivalLoss(nn.Module):
    """
    Cox proportional hazards loss for survival prediction tasks.
    Used for overall survival (OS) prediction from WSIs.
    """

    def forward(
        self,
        hazards: torch.Tensor,
        survival_time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """
        Partial likelihood loss for censored survival data.

        Args:
            hazards:       Predicted hazard scores [B]
            survival_time: Follow-up time [B]
            event:         Event indicator (1=death, 0=censored) [B]
        """
        # Sort by descending survival time
        sort_idx = torch.argsort(survival_time, descending=True)
        h = hazards[sort_idx]
        e = event[sort_idx].float()

        # Cumulative hazard log-sum-exp trick
        log_risk = torch.logcumsumexp(h, dim=0)
        log_likelihood = h - log_risk
        loss = -torch.mean(e * log_likelihood)
        return loss


# ---------------------------------------------------------------------------
# Class-Balanced Sampler
# ---------------------------------------------------------------------------


def build_weighted_sampler(
    labels: List[int],
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler for class-balanced batch construction.

    For pathology datasets that are often imbalanced (e.g., 75% negative),
    this ensures each training batch sees equal class representation.

    Args:
        labels:      List of integer class labels for all samples
        replacement: Sample with replacement

    Returns:
        WeightedRandomSampler
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor([class_weights[l] for l in labels])

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=replacement,
    )


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    """Tracks running average of a metric."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",  # 'max' for AUC, 'min' for loss
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "max":
            improved = metric > self.best_score + self.min_delta
        else:
            improved = metric < self.best_score - self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


# ---------------------------------------------------------------------------
# MIL Trainer
# ---------------------------------------------------------------------------


class MILTrainer:
    """
    Training engine for Multiple Instance Learning models.

    Handles the full training loop including:
        - Mixed precision forward/backward pass
        - CLAM instance loss integration
        - Gradient accumulation
        - Validation and checkpointing
        - Learning rate scheduling

    Typical usage:
        trainer = MILTrainer(model, config, output_dir)
        trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        output_dir: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        """
        Args:
            model:      MIL model (ABMIL, CLAM_SB, CLAM_MB, TransMIL)
            config:     Training configuration dictionary
            output_dir: Directory for checkpoints and logs
            device:     Training device
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Loss function
        self.bag_loss_fn = self._build_loss()

        # Mixed precision
        self.use_amp = config.get("amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = -float("inf")
        self.best_epoch = 0

        # Early stopping
        patience = config.get("early_stopping_patience", 20)
        self.early_stopper = EarlyStopping(patience=patience, mode="max")

        # Gradient accumulation
        self.grad_accum_steps = config.get("grad_accum_steps", 1)

        # TensorBoard
        self.writer = None
        if config.get("use_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.output_dir / "logs"))
            except ImportError:
                logger.warning("TensorBoard not available")

        logger.info(
            "MILTrainer initialized | Model: %s | AMP: %s | GradAccum: %d",
            type(model).__name__,
            self.use_amp,
            self.grad_accum_steps,
        )

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        lr = self.config.get("lr", 2e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)
        opt_name = self.config.get("optimizer", "adam").lower()

        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            return optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=self.config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_loss(self) -> nn.Module:
        """Build bag-level loss from config."""
        loss_name = self.config.get("bag_loss", "ce").lower()
        class_weights = self.config.get("class_weights", None)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if loss_name == "ce":
            return nn.CrossEntropyLoss(weight=class_weights)
        elif loss_name == "focal":
            return FocalLoss(
                gamma=self.config.get("focal_gamma", 2.0),
                alpha=class_weights,
            )
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def _get_scheduler(self, n_steps_per_epoch: int) -> optim.lr_scheduler._LRScheduler:
        """Build LR scheduler from config."""
        sched_name = self.config.get("scheduler", "cosine").lower()
        n_epochs = self.config.get("n_epochs", 20)

        if sched_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6
            )
        elif sched_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("lr_step_size", 10),
                gamma=self.config.get("lr_gamma", 0.1),
            )
        elif sched_name == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get("lr", 2e-4),
                epochs=n_epochs,
                steps_per_epoch=n_steps_per_epoch,
            )
        elif sched_name == "none":
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda e: 1.0)
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def _forward_step(
        self,
        features: torch.Tensor,
        label: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        is_clam: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Unified forward step for all MIL model types.

        Returns:
            loss:          Scalar loss value
            logits:        [num_classes] predictions
            info:          Additional info dict
        """
        features = features.squeeze(0).to(self.device)  # [N, D]
        label = label.squeeze(0).to(self.device)
        if coords is not None:
            coords = coords.squeeze(0).to(self.device)

        model_name = type(self.model).__name__.lower()

        if "clam" in model_name:
            # CLAM forward with instance evaluation
            logits, instance_dict, _ = self.model(
                features,
                label=label,
                instance_eval=True,
            )

            # Bag loss
            if logits.dim() == 1:
                logits_2d = logits.unsqueeze(0)
                label_1d = label.unsqueeze(0) if label.dim() == 0 else label
            else:
                logits_2d = logits
                label_1d = label

            bag_loss = self.bag_loss_fn(logits_2d, label_1d)

            # Instance loss
            inst_loss_weight = self.config.get("inst_loss_weight", 0.3)
            inst_loss = instance_dict.get("instance_loss", torch.zeros(1, device=self.device))
            total_loss = (1 - inst_loss_weight) * bag_loss + inst_loss_weight * inst_loss.squeeze()

            info = {
                "bag_loss": bag_loss.item(),
                "instance_loss": inst_loss.item() if torch.is_tensor(inst_loss) else inst_loss,
            }

        elif "transmil" in model_name:
            logits, _ = self.model(features, coords=coords)
            if logits.dim() == 1:
                logits_2d = logits.unsqueeze(0)
            else:
                logits_2d = logits
            label_1d = label.unsqueeze(0) if label.dim() == 0 else label
            total_loss = self.bag_loss_fn(logits_2d, label_1d)
            info = {"bag_loss": total_loss.item()}

        else:
            # ABMIL and others
            logits, _ = self.model(features)
            if logits.dim() == 1:
                logits_2d = logits.unsqueeze(0)
            else:
                logits_2d = logits
            label_1d = label.unsqueeze(0) if label.dim() == 0 else label
            total_loss = self.bag_loss_fn(logits_2d, label_1d)
            info = {"bag_loss": total_loss.item()}

        return total_loss, logits, info

    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with epoch metrics: loss, accuracy
        """
        self.model.train()
        loss_meter = AverageMeter("train_loss")
        acc_meter = AverageMeter("train_acc")

        self.optimizer.zero_grad()
        step_count = 0

        for batch_idx, batch in enumerate(train_loader):
            features, coords, labels = batch

            # Normalize coords type
            if isinstance(coords, torch.Tensor) and coords.numel() == 0:
                coords = None

            with autocast(enabled=self.use_amp):
                loss, logits, info = self._forward_step(features, labels, coords)
                loss = loss / self.grad_accum_steps

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            step_count += 1

            if step_count % self.grad_accum_steps == 0:
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Compute accuracy
            label_val = labels.squeeze().item() if torch.is_tensor(labels) else int(labels)
            if logits.dim() > 0:
                pred = logits.detach().cpu().argmax().item()
            else:
                pred = int(logits.detach().cpu().item() > 0)
            correct = int(pred == label_val)

            loss_meter.update(loss.item() * self.grad_accum_steps)
            acc_meter.update(correct)

            if self.writer and self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss", loss_meter.val, self.global_step)

        if scheduler and self.config.get("scheduler", "cosine") not in ("onecycle",):
            scheduler.step()

        return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate on a held-out split.

        Returns:
            Dict with AUROC, accuracy, F1
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        self.model.eval()
        all_probs = []
        all_labels = []
        all_preds = []
        loss_meter = AverageMeter("val_loss")

        for batch in val_loader:
            features, coords, labels = batch
            if isinstance(coords, torch.Tensor) and coords.numel() == 0:
                coords = None

            with autocast(enabled=self.use_amp):
                loss, logits, _ = self._forward_step(features, labels, coords)

            loss_meter.update(loss.item())

            # Collect predictions
            label_val = int(labels.squeeze().item())
            if logits.dim() >= 1 and logits.shape[-1] > 1:
                import torch.nn.functional as F
                probs = F.softmax(logits.cpu(), dim=-1)
                pred = probs.argmax().item()
                prob_pos = probs[1].item() if probs.shape[-1] > 1 else probs[0].item()
            else:
                prob_pos = torch.sigmoid(logits.cpu()).item()
                pred = int(prob_pos > 0.5)

            all_probs.append(prob_pos)
            all_labels.append(label_val)
            all_preds.append(pred)

        # Compute metrics
        metrics = {"loss": loss_meter.avg}

        try:
            if len(set(all_labels)) > 1:
                metrics["auroc"] = roc_auc_score(all_labels, all_probs)
            else:
                metrics["auroc"] = 0.5
            metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["f1"] = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        except Exception as e:
            logger.warning("Metric computation failed: %s", e)
            metrics.update({"auroc": 0.0, "accuracy": 0.0, "f1": 0.0})

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        filename: str = "checkpoint.pt",
    ) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        if self.scaler:
            state["scaler_state"] = self.scaler.state_dict()
        torch.save(state, self.output_dir / filename)

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Load checkpoint. Returns epoch number."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scaler and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        logger.info("Loaded checkpoint from epoch %d", state.get("epoch", 0))
        return state.get("epoch", 0)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader (WSIBagDataset)
            val_loader:   Validation DataLoader
            n_epochs:     Override config epoch count

        Returns:
            history: Dict with lists of per-epoch metrics
        """
        n_epochs = n_epochs or self.config.get("n_epochs", 20)
        scheduler = self._get_scheduler(len(train_loader))

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_auroc": [], "val_accuracy": [],
        }

        for epoch in range(1, n_epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, scheduler)

            # Validate
            val_metrics = self.validate(val_loader)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | Train loss: %.4f | "
                "Val AUROC: %.4f | Val Acc: %.4f | LR: %.2e | Time: %.1fs",
                epoch, n_epochs,
                train_metrics["loss"],
                val_metrics.get("auroc", 0.0),
                val_metrics.get("accuracy", 0.0),
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_auroc"].append(val_metrics.get("auroc", 0.0))
            history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

            # TensorBoard
            if self.writer:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Save checkpoint
            val_auroc = val_metrics.get("auroc", 0.0)
            if val_auroc > self.best_val_metric:
                self.best_val_metric = val_auroc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, "best_model.pt")
                logger.info("  ↑ New best AUROC: %.4f (epoch %d)", val_auroc, epoch)

            # Periodic checkpoint
            if epoch % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(epoch, val_metrics, f"checkpoint_epoch{epoch:03d}.pt")

            # Early stopping
            if self.early_stopper(val_auroc):
                logger.info(
                    "Early stopping at epoch %d (best: %.4f at epoch %d)",
                    epoch, self.best_val_metric, self.best_epoch,
                )
                break

        if self.writer:
            self.writer.close()

        logger.info(
            "Training complete. Best val AUROC: %.4f at epoch %d",
            self.best_val_metric, self.best_epoch,
        )
        return history

# distributed feature extraction for large WSI datasets
# uses DataParallel for multi-GPU when available
def setup_distributed_feature_extraction(model, device_ids=None):
    """wrap feature extractor with DataParallel for multi-GPU extraction"""
    import torch
    import torch.nn as nn

    if device_ids is None:
        n_gpus = torch.cuda.device_count()
        device_ids = list(range(n_gpus)) if n_gpus > 1 else None

    if device_ids and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f'using {len(device_ids)} GPUs for feature extraction: {device_ids}')
    else:
        print('single GPU or CPU for feature extraction')
    return model
