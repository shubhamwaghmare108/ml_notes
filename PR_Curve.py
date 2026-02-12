import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

# --------------------------------------------------
# Data & model
# --------------------------------------------------
X, y = make_classification(
    n_samples=3000,
    n_features=5,
    n_informative=3,
    weights=[0.95, 0.05],
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]

# --------------------------------------------------
# Precompute PR curve + metrics
# --------------------------------------------------
precision, recall, thresholds = precision_recall_curve(y, y_prob)

# thresholds has length n-1
thresholds = np.r_[thresholds, 1.0]

# Confusion counts
tp = recall * np.sum(y)
fp = tp * (1 / precision - 1)
fn = np.sum(y) - tp

f1 = 2 * precision * recall / (precision + recall + 1e-9)

# --------------------------------------------------
# Initial index
# --------------------------------------------------
idx = len(thresholds) // 2

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.28)

ax.plot(recall, precision, label="PR Curve")
point, = ax.plot([recall[idx]], [precision[idx]], "o", markersize=8)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Fast Interactive PR Curve")
ax.legend()

text_box = ax.text(
    0.02, 0.02,
    f"Threshold: {thresholds[idx]:.2f}\n"
    f"Precision: {precision[idx]:.3f}\n"
    f"Recall: {recall[idx]:.3f}\n"
    f"F1: {f1[idx]:.3f}",
    transform=ax.transAxes,
    bbox=dict(facecolor="white", alpha=0.85)
)

# --------------------------------------------------
# Threshold slider (INDEX based → FAST)
# --------------------------------------------------
ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
slider = Slider(ax_slider, "Index", 0, len(thresholds)-1, valinit=idx, valstep=1)

def update(val):
    i = int(slider.val)
    point.set_data([recall[i]], [precision[i]])

    text_box.set_text(
        f"Threshold: {thresholds[i]:.2f}\n"
        f"Precision: {precision[i]:.3f}\n"
        f"Recall: {recall[i]:.3f}\n"
        f"F1: {f1[i]:.3f}"
    )

    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()



