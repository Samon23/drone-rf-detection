"""Step 3: train a classifier and test it on segments it has never seen."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from common import FIG_DIR
from features_lib import band_centres_mhz

data = np.load("features.npz")
X, y, groups = data["X"], data["y"], data["groups"]

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
all_true, all_pred, scores = [], [], []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], pred)
    scores.append(acc); all_true.extend(y[test_idx]); all_pred.extend(pred)
    print(f"Fold {fold}: tested on {sorted(str(g) for g in set(groups[test_idx]))}, accuracy {acc:.3f}")

print(f"\nMean accuracy on unseen segments: {np.mean(scores):.3f}")

cm = confusion_matrix(all_true, all_pred)
ConfusionMatrixDisplay(cm, display_labels=["No drone", "Drone"]).plot(cmap="Blues")
plt.title("Confusion matrix (all folds)")
plt.tight_layout(); plt.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=120)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
imp = model.feature_importances_
freqs = band_centres_mhz()
width = freqs[1] - freqs[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, part, name in zip(axes, (imp[:64], imp[64:]), ("Lower receiver", "Upper receiver")):
    ax.bar(freqs, part, width=width * 0.9)
    ax.set_title(name)
    ax.set_xlabel("Frequency within the receiver's capture (MHz)")
axes[0].set_ylabel("Importance")
fig.suptitle("Which frequency bands the classifier relies on")
fig.tight_layout(); fig.savefig(FIG_DIR / "06_feature_importance.png", dpi=120)
print("Saved confusion matrix and feature importance figures.")
