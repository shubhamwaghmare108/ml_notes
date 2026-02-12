import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.metrics import roc_curve, roc_auc_score

# =========================
# 1. Realistic Dummy Data
# =========================
np.random.seed(42)
n = 200

y_true = np.random.randint(0, 2, size=n)

y_probs = np.where(
    y_true == 1,
    np.random.uniform(0.4, 0.95, size=n),   # positives
    np.random.uniform(0.05, 0.7, size=n)    # negatives
)

# =========================
# 2. ROC + AUC
# =========================
fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
auc = roc_auc_score(y_true, y_probs)

# Youden's J for optimal threshold
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_fpr = fpr[best_idx]
best_tpr = tpr[best_idx]
best_threshold = roc_thresholds[best_idx]

# =========================
# 3. Plot Setup
# =========================
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.5)

roc_line, = ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
random_line, = ax.plot([0, 1], [0, 1], linestyle='--', label="Random")

current_point, = ax.plot([], [], marker='o', label="Current Threshold")
best_point, = ax.plot(best_fpr, best_tpr, marker='o', label="Optimal (Youden J)")

ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("Professional Interactive ROC Analyzer")
ax.legend(loc="lower right")

# =========================
# 4. Metrics Text Box
# =========================
metrics_text = ax.figure.text(
    0.05, 0.10, "",
    fontsize=9,
    verticalalignment='bottom',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.95)
)




# =========================
# 5. Slider
# =========================
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
threshold_slider = Slider(
    ax_slider, "Threshold", 0.0, 1.0, valinit=0.5
)

# =========================
# 6. Update Function
# =========================
def update(val):
    threshold = threshold_slider.val
    y_pred = (y_probs >= threshold).astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    TPR = TP / (TP + FN) if (TP + FN) else 0
    FPR = FP / (FP + TN) if (FP + TN) else 0

    Precision = TP / (TP + FP) if (TP + FP) else 0
    Recall = TPR
    F1 = (2 * Precision * Recall / (Precision + Recall)
          if (Precision + Recall) else 0)

    current_point.set_data([FPR], [TPR])

    metrics_text.set_text(
        f"Threshold = {threshold:.3f}\n\n"
        f"Confusion Matrix:\n"
        f"TP = {TP}   FP = {FP}\n"
        f"FN = {FN}   TN = {TN}\n\n"
        f"Metrics:\n"
        f"TPR (Recall) = {TPR:.3f}\n"
        f"FPR = {FPR:.3f}\n"
        f"Precision = {Precision:.3f}\n"
        f"F1-score = {F1:.3f}\n\n"
        f"AUC = {auc:.3f}\n\n"
        f"Optimal Threshold (Youden J) = {best_threshold:.3f}"
    )

    fig.canvas.draw_idle()

# =========================
# 7. Connect
# =========================
threshold_slider.on_changed(update)
update(0.5)


plt.show()
