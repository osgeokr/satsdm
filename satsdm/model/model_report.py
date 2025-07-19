from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from satsdm.model.core import MaxentModel

def get_jackknife_report(model, X, y):
    """
    Return a text-formatted jackknife variable importance report.
    """
    jack_df = model.jackknife_importance(X, y)
    report = "=== Jackknife Variable Importance ===\n"
    report += jack_df.to_string(index=False)
    return report

def generate_evaluation_summary(y_true, y_pred_prob, threshold=0.5):
    """
    Generate evaluation summary at a specific threshold.
    """
    # Apply threshold to convert probabilities to binary predictions
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_prob)

    summary = (
        "Evaluation Summary (Maxent Model)\n\n"
        f"- AUC       : {auc:.3f} (good if > 0.75)\n"
        "\n"
        "Note:\n"
        "- AUC is the primary metric for presence-background models like Maxent.\n"
        "- Precision, Recall, and F1-score are less informative due to class imbalance.\n"
    )

    """
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)

    # Generate formatted summary
    summary = (
        f"Evaluation Summary (Threshold = {threshold:.1f})\n"
        f"AUC       : {auc:.3f}\n"
        f"Precision : {precision:.3f}\n"
        f"Recall    : {recall:.3f}\n"
        f"F1-score  : {f1:.3f}\n\n"
        f"Explanation:\n"
        f"- AUC       : How well the model separates presence from absence (1.0 = perfect).\n"
        f"- Precision : Of all locations predicted as 'presence', how many were actually presence.\n"
        f"- Recall    : Of all actual 'presence' locations, how many the model captured.\n"
        f"- F1-score  : Balance between Precision and Recall (higher = better)."
    )
    """
    
    return summary

def save_roc_curve(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
