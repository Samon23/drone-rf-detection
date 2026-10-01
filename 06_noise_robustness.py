"""Step 6: where does the detector break? Accuracy on unseen recordings as noise is added.

For each fold the Random Forest is trained on clean features. The test recordings
are then reloaded, white Gaussian noise is added to every window, features are
recomputed, and accuracy is measured. The noise level is set relative to each
window's own power, so 0 dB means the added noise is as strong as the window itself.
The recordings already contain some noise, so this is not the true SNR of the drone signal.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from common import CHUNK, FIG_DIR, list_segments, load_signal
from features_lib import band_features

LEVELS_DB = [30, 20, 10, 5, 0, -5, -10]   # window power relative to added noise, in dB
rng = np.random.default_rng(0)


def add_noise(x, level_db):
    """Add white Gaussian noise so that window power / noise power = level_db."""
    noise_power = np.mean(x ** 2) / 10 ** (level_db / 10)
    return x + rng.normal(0.0, np.sqrt(noise_power), size=x.shape)


data = np.load("features.npz")
X, y, groups = data["X"], data["y"], data["groups"]
files = {seg_id: (low, high) for low, high, _, seg_id in list_segments()}

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
acc = {level: [] for level in LEVELS_DB}
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X[train_idx], y[train_idx])
    for seg_id in sorted(set(groups[test_idx])):
        x_low, x_high = (load_signal(f) for f in files[seg_id])
        label = y[groups == seg_id][0]
        n_chunks = min(len(x_low), len(x_high)) // CHUNK
        for level in LEVELS_DB:
            feats = []
            for i in range(n_chunks):
                a, b = i * CHUNK, (i + 1) * CHUNK
                feats.append(np.concatenate([band_features(add_noise(x_low[a:b], level)),
                                             band_features(add_noise(x_high[a:b], level))]))
            pred = model.predict(np.array(feats))
            acc[level].append(accuracy_score(np.full(len(pred), label), pred))
    print(f"Fold {fold} done")

means = [np.mean(acc[level]) for level in LEVELS_DB]
for level, m in zip(LEVELS_DB, means):
    print(f"{level:>4} dB: accuracy {m:.3f}")

plt.figure(figsize=(6, 4))
plt.plot(LEVELS_DB, means, marker="o")
plt.gca().invert_xaxis()
plt.axhline(0.5, color="grey", linestyle="--", label="Chance level")
plt.xlabel("Window power relative to added noise (dB)")
plt.ylabel("Accuracy on unseen recordings")
plt.title("Detection accuracy as noise increases")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG_DIR / "07_noise_robustness.png", dpi=120)
print("Saved figures/07_noise_robustness.png")
