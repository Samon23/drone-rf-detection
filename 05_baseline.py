"""Step 5: how hard is this task really? Compare the Random Forest with a one-feature baseline.

The baseline uses a single number per window, the average log power across the
upper receiver's 64 bands, and a logistic regression on top. If this alone gets
close to 100%, the dataset separates easily and the README should say so.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

data = np.load("features.npz")
X, y, groups = data["X"], data["y"], data["groups"]

# One feature per window: mean log power of the upper receiver (features 64-127)
upper_power = X[:, 64:].mean(axis=1, keepdims=True)

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    model = LogisticRegression().fit(upper_power[train_idx], y[train_idx])
    acc = accuracy_score(y[test_idx], model.predict(upper_power[test_idx]))
    scores.append(acc)
    print(f"Fold {fold}: baseline accuracy {acc:.3f}")

print(f"\nOne-feature baseline, mean accuracy on unseen recordings: {np.mean(scores):.3f}")
