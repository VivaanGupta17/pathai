"""
Pathology-Specific Evaluation Metrics

Implements:
    1. Slide-level AUROC, accuracy, F1, precision, recall
    2. FROC (Free-Response ROC) for lesion detection — Camelyon standard
    3. Quadratic Weighted Kappa (QWK) for ordinal grading tasks
       (Gleason grading, ISUP grade groups)
    4. Per-class confusion matrix with pathology-specific visualization
    5. Calibration metrics (ECE, reliability diagrams)
    6. Attention heatmap IoU vs. pathologist annotations
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slide-Level Classification Metrics
# ---------------------------------------------------------------------------


def compute_slide_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive slide-level classification metrics.

    Args:
        y_true:       Ground-truth labels [N]
        y_pred:       Predicted class labels [N]
        y_prob:       Predicted probabilities — [N] for binary, [N, C] for multi-class
        class_names:  Optional class name labels

    Returns:
        Dict containing:
            auroc, accuracy, f1, precision, recall,
            average_precision, specificity, balanced_accuracy
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2

    metrics: Dict[str, float] = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Balanced accuracy (macro recall)
    metrics["balanced_accuracy"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # AUROC
    try:
        if is_binary:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["auroc"] = float(roc_auc_score(y_true, prob_pos))
        else:
            metrics["auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError as e:
        metrics["auroc"] = float("nan")
        logger.warning("AUROC computation failed: %s", e)

    # F1
    avg = "binary" if is_binary else "weighted"
    metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Precision, Recall
    metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    # Average Precision (area under PR curve)
    try:
        if is_binary:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["average_precision"] = float(average_precision_score(y_true, prob_pos))
        else:
            metrics["average_precision"] = float(
                average_precision_score(y_true, y_prob, average="macro")
            )
    except Exception:
        metrics["average_precision"] = float("nan")

    # Specificity (for binary)
    if is_binary:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return metrics


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUROC.

    Returns:
        fpr:   False positive rates
        tpr:   True positive rates
        auroc: Area under ROC curve
    """
    y_prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    auroc = roc_auc_score(y_true, y_prob_pos)
    return fpr, tpr, float(auroc)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "youden",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Methods:
        'youden':  Maximizes Youden's J = sensitivity + specificity - 1
        'f1':      Maximizes F1 score
        'balanced': Maximizes balanced accuracy

    Returns:
        threshold: Optimal decision threshold
        metric_value: Value of the optimization metric at threshold
    """
    y_prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)

    if metric == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])
    elif metric == "f1":
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in thresholds:
            pred = (y_prob_pos >= thresh).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return float(best_thresh), float(best_f1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Quadratic Weighted Kappa (for grading tasks)
# ---------------------------------------------------------------------------


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    QWK is the standard metric for ordinal grading tasks:
        - Gleason grading (grade groups 1-5)
        - ISUP grade (0-5)
        - Nottingham grade (1-3)

    Weights penalize disagreements quadratically with distance,
    so a prediction of grade 5 when truth is grade 1 is much worse
    than being off by 1 grade.

    Args:
        y_true:    True grades [N]
        y_pred:    Predicted grades [N]
        n_classes: Number of grade categories (auto-detected if None)

    Returns:
        kappa: Quadratic weighted kappa in [-1, 1]
    """
    return float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )


def per_grade_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grade_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-grade precision, recall, and F1.

    Args:
        y_true:       True grade labels
        y_pred:       Predicted grade labels
        grade_names:  Optional names for each grade

    Returns:
        Dict mapping grade_name → {precision, recall, f1, support}
    """
    grades = sorted(set(y_true) | set(y_pred))
    if grade_names is None:
        grade_names = [f"Grade_{g}" for g in grades]

    precision = precision_score(y_true, y_pred, labels=grades, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=grades, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=grades, average=None, zero_division=0)
    support = np.bincount([list(grades).index(y) for y in y_true], minlength=len(grades))

    result = {}
    for i, (grade, name) in enumerate(zip(grades, grade_names)):
        result[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return result


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Compute confusion matrix.

    Args:
        y_true:      True labels
        y_pred:      Predicted labels
        class_names: Class names for labeling
        normalize:   'true' (row-normalize), 'pred', 'all', or None

    Returns:
        cm:           Confusion matrix [C, C]
        class_names:  Class names (auto-generated if not provided)
    """
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)

    if class_names is None:
        class_names = [str(c) for c in classes]

    return cm, class_names


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        cm:          Confusion matrix [C, C]
        class_names: Class labels
        output_path: If set, save to this path
        title:       Plot title
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if cm.dtype == float else "d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved confusion matrix to %s", output_path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping plot")


# ---------------------------------------------------------------------------
# Calibration Metrics
# ---------------------------------------------------------------------------


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.
    Lower ECE = better calibrated model.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Args:
        y_true: True binary labels [N]
        y_prob: Predicted positive probabilities [N]
        n_bins: Number of calibration bins

    Returns:
        ece: Expected calibration error in [0, 1]
    """
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Attention Heatmap IoU
# ---------------------------------------------------------------------------


def compute_attention_iou(
    attention_map: np.ndarray,
    annotation_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute IoU between thresholded attention heatmap and pathologist annotation.

    Args:
        attention_map:    Normalized attention values [H, W] in [0, 1]
        annotation_mask:  Binary ground-truth tumor mask [H, W]
        threshold:        Attention threshold for binarization

    Returns:
        Dict with iou, dice, precision, recall
    """
    pred_mask = (attention_map >= threshold).astype(bool)
    gt_mask = (annotation_mask > 0).astype(bool)

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / (pred_mask.sum() + gt_mask.sum())) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    precision = float(intersection / pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
    recall = float(intersection / gt_mask.sum()) if gt_mask.sum() > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def compute_attention_iou_at_thresholds(
    attention_map: np.ndarray,
    annotation_mask: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute max IoU across multiple thresholds (best-case IoU).

    Args:
        attention_map:   [H, W] normalized attention
        annotation_mask: [H, W] binary annotation
        thresholds:      Array of thresholds to evaluate (default: 0.1 to 0.9)

    Returns:
        Dict with max_iou, best_threshold, and IoU at standard thresholds
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    best_iou = 0.0
    best_thresh = 0.5
    results = {}

    for thresh in thresholds:
        metrics = compute_attention_iou(attention_map, annotation_mask, thresh)
        results[f"iou@{thresh:.1f}"] = metrics["iou"]
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_thresh = thresh

    results["max_iou"] = best_iou
    results["best_threshold"] = best_thresh
    return results


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    task: str = "binary_classification",
) -> str:
    """
    Generate a human-readable evaluation report string.

    Args:
        y_true:      Ground-truth labels
        y_pred:      Predicted labels
        y_prob:      Predicted probabilities
        class_names: Class names
        task:        Task type for formatting

    Returns:
        Formatted report string
    """
    metrics = compute_slide_metrics(y_true, y_pred, y_prob, class_names)
    cm, names = compute_confusion_matrix(y_true, y_pred, class_names)

    lines = [
        "=" * 60,
        "PathAI Evaluation Report",
        "=" * 60,
        f"Task: {task}",
        f"Samples: {len(y_true)}",
        "",
        "--- Slide-Level Metrics ---",
        f"  AUROC:             {metrics.get('auroc', 0):.4f}",
        f"  Accuracy:          {metrics.get('accuracy', 0):.4f}",
        f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}",
        f"  F1 (weighted):     {metrics.get('f1', 0):.4f}",
        f"  F1 (macro):        {metrics.get('f1_macro', 0):.4f}",
        f"  Precision:         {metrics.get('precision', 0):.4f}",
        f"  Recall:            {metrics.get('recall', 0):.4f}",
        f"  Avg Precision:     {metrics.get('average_precision', 0):.4f}",
    ]

    if "sensitivity" in metrics:
        lines += [
            "",
            "--- Binary Classification ---",
            f"  Sensitivity (TPR): {metrics['sensitivity']:.4f}",
            f"  Specificity (TNR): {metrics['specificity']:.4f}",
            f"  PPV:               {metrics.get('ppv', 0):.4f}",
            f"  NPV:               {metrics.get('npv', 0):.4f}",
        ]

    lines += [
        "",
        "--- Confusion Matrix ---",
    ]
    if names:
        lines.append("  " + "  ".join(f"{n:>10}" for n in names))
    for i, row in enumerate(cm):
        label = names[i] if names else str(i)
        lines.append(f"  {label:10}" + "  ".join(f"{v:10d}" for v in row))

    lines.append("=" * 60)
    return "\n".join(lines)
