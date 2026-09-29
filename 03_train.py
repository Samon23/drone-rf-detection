"""Step 3: train a classifier and test it on segments it has never seen."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from common import FIG_DIR

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
plt.figure(figsize=(10, 3))
plt.bar(range(X.shape[1]), model.feature_importances_)
plt.axvline(63.5, color="red", linestyle="--")
plt.xlabel("Feature (0-63 = lower band, 64-127 = upper band)")
plt.ylabel("Importance"); plt.title("Which frequency bands the model relies on")
plt.tight_layout(); plt.savefig(FIG_DIR / "06_feature_importance.png", dpi=120)
print("Saved confusion matrix and feature importance figures.")
