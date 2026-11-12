"""FFT comparison of the r=8 blue- vs white-noise runs (runs/r8).

Figure rows:
  1. 2D power spectra of each pattern (same step) and of the beta noise that drove it
  2. power vs direction on the main ring | power vs spacing | diagonal bias over time

Diagonal bias = (P_diag - P_axis) / (P_diag + P_axis), using spectral power within
15 deg of the diagonals (45/135) vs the axes (0/90). 0 = no preference.
"""
import glob
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from blue_noise import gen_blue_noise_slice

RUNS = "runs/r8"
STEP = "02000000"
N = 1024
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
runs = [("blue_s0", BLUE, "-"), ("blue_s1", BLUE, "--"), ("white_s0", ORANGE, "-"), ("white_s1", ORANGE, "--")]

f = np.fft.fftshift(np.fft.fftfreq(N))
FX, FY = np.meshgrid(f, f)
K = np.hypot(FX, FY)
TH = np.degrees(np.arctan2(FY, FX)) % 180


def spectrum(img):
    img = img.astype(np.float64)
    img -= img.mean()
    return np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2


def ring_mask(P):
    k0 = (P * K).sum() / P.sum()
    return (K > 0.75 * k0) & (K < 1.25 * k0)


def diag_bias(P):
    m = ring_mask(P)
    d = np.minimum(np.abs(TH - 45), np.abs(TH - 135)) < 15
    a = (np.minimum(TH, 180 - TH) < 15) | (np.abs(TH - 90) < 15)
    pd, pa = P[m & d].sum(), P[m & a].sum()
    return (pd - pa) / (pd + pa)


def angular(P, bins=90):
    m = ring_mask(P)
    h, e = np.histogram(TH[m], bins=bins, range=(0, 180), weights=P[m])
    c, _ = np.histogram(TH[m], bins=bins, range=(0, 180))
    prof = h / np.maximum(c, 1)
    prof = np.convolve(np.r_[prof[-2:], prof, prof[:2]], np.ones(5) / 5, "valid")  # +-4 deg smoothing, wraps
    return (e[:-1] + e[1:]) / 2, prof / prof.mean()


def radial(P, bins=120):
    h, e = np.histogram(K.ravel(), bins=bins, range=(0, 0.12), weights=P.ravel())
    c, _ = np.histogram(K.ravel(), bins=bins, range=(0, 0.12))
    prof = h / np.maximum(c, 1)
    return (e[:-1] + e[1:]) / 2, prof / prof.max()


pats = {n: spectrum(cv2.imread(f"{RUNS}/{n}_{STEP}.png", 0)) for n, _, _ in runs}

torch.manual_seed(0)
blue_beta = gen_blue_noise_slice(N, N, 4, 0.8)[:, :, 0].cpu().numpy()
white_beta = np.random.default_rng(0).random((N, N))
beta_spec = {"blue-noise β": spectrum(blue_beta), "white-noise β": spectrum(white_beta)}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig = plt.figure(figsize=(17, 9.2), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.38, wspace=0.32)

zoom = 0.1
ext = [-0.5, 0.5, -0.5, 0.5]
for j, (name, P) in enumerate(list(pats.items()) + list(beta_spec.items())):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(np.log10(P + 1e-9), cmap="Blues", extent=ext, origin="lower",
              vmin=np.percentile(np.log10(P + 1e-9), 50), vmax=np.log10(P.max()))
    lim = zoom if j < 4 else 0.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    sub = f"pattern at {int(STEP)/1e6:.1f}M steps" if j < 4 else "noise that drives β"
    ax.set_title(f"{name}\n{sub}", fontsize=10, color=INK)
    ax.text(0.03, 0.03, f"diag bias {diag_bias(P):+.2f}", transform=ax.transAxes, fontsize=9,
            color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))

def style(ax, title, xl, yl):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 0:2])
for n, c, ls in runs:
    x, y = angular(pats[n])
    ax.plot(x, y, color=c, ls=ls, lw=2, label=n)
for v in (45, 135):
    ax.axvline(v, color=INK2, lw=0.8, ls=":")
ax.set_xticks([0, 45, 90, 135, 180])
style(ax, "Power vs stripe direction (main ring)", "direction of wavevector (deg)", "power / mean")
ax.legend(frameon=False, ncol=2, fontsize=9)

ax = fig.add_subplot(gs[1, 2:4])
for n, c, ls in runs:
    x, y = radial(pats[n])
    ax.plot(x, y, color=c, ls=ls, lw=2, label=n)
k0 = (pats["blue_s0"] * K).sum() / pats["blue_s0"].sum()
ax.annotate(f"spacing ≈ {1/k0:.1f} px", (k0, 1.0), xytext=(k0 + 0.02, 0.8), color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
style(ax, "Power vs spatial frequency", "frequency (cycles / px)", "power / peak")
ax.legend(frameon=False, ncol=2, fontsize=9)

ax = fig.add_subplot(gs[1, 4:6])
for n, c, ls in runs:
    files = sorted(glob.glob(f"{RUNS}/{n}_0*.png"))
    steps = [int(re.search(r"_(\d{8})\.png", fn).group(1)) / 1e6 for fn in files]
    vals = [diag_bias(spectrum(cv2.imread(fn, 0))) for fn in files]
    ax.plot(steps, vals, color=c, ls=ls, lw=2, marker="o", ms=4, label=n)
ax.axhline(0, color=INK2, lw=0.8)
style(ax, "Diagonal bias over time", "step (millions)", "diag bias  (+ = diagonal)")
ax.legend(frameon=False, ncol=2, fontsize=9)

fig.suptitle("r = 8 (D_a/D_b): blue- vs white-noise β, FFT comparison", x=0.01, ha="left",
             fontsize=14, color=INK)
out = f"{RUNS}/fft_compare.png"
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved", out)
for n in pats:
    print(f"{n:9s} diag bias at {STEP}: {diag_bias(pats[n]):+.3f}")
for n, P in beta_spec.items():
    print(f"{n:14s} diag bias: {diag_bias(P):+.3f}")
