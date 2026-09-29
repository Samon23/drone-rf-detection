"""Step 1: look at the raw signals before doing anything clever."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from common import FIG_DIR, FS, list_segments, load_signal

FIG_DIR.mkdir(exist_ok=True)
segments = list_segments()
background = next(s for s in segments if s[2] == 0)   # first no-drone segment
drone = next(s for s in segments if s[2] == 1)        # first drone segment

x_bg = load_signal(background[0])
x_dr = load_signal(drone[0])
print(f"Background file: {background[0].name}, {len(x_bg):,} samples")
print(f"Drone file:      {drone[0].name}, {len(x_dr):,} samples")

# 1. Time domain: the raw samples
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
t_us = np.arange(2000) / FS * 1e6
axes[0].plot(t_us, x_bg[:2000]); axes[0].set_title("No drone (background)")
axes[1].plot(t_us, x_dr[:2000]); axes[1].set_title("Drone present")
axes[1].set_xlabel("Time (microseconds)")
fig.tight_layout(); fig.savefig(FIG_DIR / "01_time_domain.png", dpi=120)

# 2. Spectrogram: how the frequency content changes over time
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, x, title in [(axes[0], x_bg, "No drone"), (axes[1], x_dr, "Drone present")]:
    f, t, Sxx = signal.spectrogram(x[:200_000], fs=FS, nperseg=512)
    ax.pcolormesh(t * 1e3, f / 1e6, 10 * np.log10(Sxx + 1e-12), shading="auto")
    ax.set_title(title); ax.set_xlabel("Time (ms)")
axes[0].set_ylabel("Frequency (MHz)")
fig.tight_layout(); fig.savefig(FIG_DIR / "02_spectrogram.png", dpi=120)

# 3. Power spectral density: average power at each frequency
fig, ax = plt.subplots(figsize=(10, 4))
for x, name in [(x_bg, "No drone"), (x_dr, "Drone present")]:
    f, pxx = signal.welch(x, fs=FS, nperseg=1024)
    ax.semilogy(f / 1e6, pxx, label=name)
ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Power"); ax.legend()
ax.set_title("Welch power spectral density")
fig.tight_layout(); fig.savefig(FIG_DIR / "03_psd.png", dpi=120)

# 4. From real samples to I/Q with the Hilbert transform
z = signal.hilbert(x_dr[:500].astype(np.float64))   # complex analytic signal
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(z.real, label="I (in-phase)"); ax.plot(z.imag, label="Q (quadrature)")
ax.set_title("Analytic signal of the drone capture"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "04_iq_hilbert.png", dpi=120)

print("Saved 4 figures to the figures folder.")
