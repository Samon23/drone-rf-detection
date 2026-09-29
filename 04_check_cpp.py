"""Step 4: prove the C++ filter gives the same answer as NumPy."""
import subprocess
import sys
import numpy as np
from common import list_segments, load_signal

N = 8
x = load_signal(list_segments()[0][0])[:5000].astype(np.float64)
np.savetxt("cpp/input.txt", x)

exe = "cpp/moving_average.exe" if sys.platform == "win32" else "./cpp/moving_average"
subprocess.run([exe, "cpp/input.txt", "cpp/output.txt", str(N)], check=True)

y_cpp = np.loadtxt("cpp/output.txt")
y_numpy = np.convolve(x, np.ones(N) / N, mode="valid")
print(f"Max difference between C++ and NumPy: {np.max(np.abs(y_cpp - y_numpy)):.2e}")
