import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# h(r, q) = r q^2 - (3r - 4) q + 4   (alpha = 1)
# h < 0  -> a bump of that size grows (Turing instability)
# h > 0  -> the bump decays

r = np.linspace(0.01, 8, 600)
q = np.linspace(0, 3, 600)
R, Q = np.meshgrid(r, q)
H = R * Q**2 - (3 * R - 4) * Q + 4

fig, ax = plt.subplots(figsize=(8, 5.5))

# Diverging colormap centered at 0 so growth (negative) and decay (positive) separate clearly
norm = TwoSlopeNorm(vmin=H.min(), vcenter=0, vmax=H.max())
mesh = ax.pcolormesh(R, Q, H, cmap="RdBu_r", norm=norm, shading="auto")
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(r"$h(r,q) = rq^2 - (3r-4)q + 4$")

# Boundary of the unstable region: h = 0
ax.contour(R, Q, H, levels=[0], colors="black", linewidths=1.5)

# Critical point: r = 4, q = 1, where h just touches 0
ax.axvline(4, color="gray", linestyle=":", linewidth=1)
ax.plot(4, 1, "ko", markersize=7, label=r"critical point $(r, q) = (4, 1)$")

ax.set_xlabel(r"diffusion ratio $r$")
ax.set_ylabel(r"$q=sk^2$ (s multiplier)")
ax.set_title("Plot of (r, s)")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("turing_threshold.png", dpi=200)
plt.show()