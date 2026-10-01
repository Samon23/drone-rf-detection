"""Step 2: turn raw signals into a table of numbers a model can learn from."""
import numpy as np
from common import CHUNK, list_segments, load_signal
from features_lib import band_features

rows, labels, groups = [], [], []
for low_file, high_file, label, seg_id in list_segments():
    x_low, x_high = load_signal(low_file), load_signal(high_file)
    n_chunks = min(len(x_low), len(x_high)) // CHUNK
    for i in range(n_chunks):
        a, b = i * CHUNK, (i + 1) * CHUNK
        feats = np.concatenate([band_features(x_low[a:b]), band_features(x_high[a:b])])
        rows.append(feats); labels.append(label); groups.append(seg_id)
    print(f"{seg_id}: label={label}, windows={n_chunks}")

np.savez("features.npz", X=np.array(rows), y=np.array(labels), groups=np.array(groups))
print(f"Saved features.npz with {len(rows)} windows and {len(rows[0])} features each.")
