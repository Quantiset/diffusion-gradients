"""Pick where a cone's apex sits on a flat periodic pattern so it wraps with the smallest seam.

A cone is developable: unrolled, it is a flat sector of angle 2*pi*sin(alpha).
With sin(alpha) = 1/6 that sector is exactly 60 degrees, which matches the 6-fold
symmetry of a hexagonal lattice: if the apex is on a spot, the two seam edges of the
sector carry the same pattern. This script searches spots (and the sector's rotation)
for the best seam match and prints the shader uniforms for scenes/cone.gdshader.

Usage: python cone_unroll.py pattern.png [--tile-world 8] [--top-radius 0.4692]
"""
import argparse
import math

import cv2
import numpy as np
from scipy import ndimage

p = argparse.ArgumentParser()
p.add_argument("img")
p.add_argument("--tile-world", type=float, default=8.0, help="world units covered by one texture tile")
p.add_argument("--bottom-radius", type=float, default=1.0)
p.add_argument("--top-radius", type=float, default=0.4692)
p.add_argument("--height", type=float, default=3.14)
p.add_argument("--candidates", type=int, default=150)
p.add_argument("--preview", default=None)
p.add_argument("--no-strain-fix", action="store_true")
args = p.parse_args()

img = cv2.imread(args.img, 0).astype(np.float32) / 255.0
N = img.shape[0]
px = N / args.tile_world  # pixels per world unit

dr = args.bottom_radius - args.top_radius
slant = math.hypot(dr, args.height)
sin_a = dr / slant
sector = 2 * math.pi * sin_a
rho0, rho1 = args.top_radius / sin_a, args.bottom_radius / sin_a
print(f"sin(alpha)={sin_a:.4f}  sector={math.degrees(sector):.2f} deg  apex distance {rho0:.3f}..{rho1:.3f} world")

# Lattice strain: the periodic box can force a slightly squashed hex lattice, which
# breaks the 60-degree match at the seam. Measure the 3 reciprocal vectors from the FFT
# and find M with K_meas @ M = K_ideal; texture offset = M @ (offset in an ideal lattice).
P = np.abs(np.fft.fft2(img - img.mean())) ** 2
P[0, 0] = 0
f = np.fft.fftfreq(N) * N
pk = np.argwhere((P == ndimage.maximum_filter(P, size=9, mode="wrap")) & (P > 0.05 * P.max()))
pk = pk[np.argsort(-P[tuple(pk.T)])][:6]
K = np.array([[f[c], f[r]] for r, c in pk])          # (k_col, k_row) in cycles per tile
flip = (K[:, 1] < 0) | ((K[:, 1] == 0) & (K[:, 0] < 0))   # fold +k/-k pairs into the upper half
K[flip] *= -1
K = np.unique(K, axis=0)
K = K[np.argsort(np.arctan2(K[:, 1], K[:, 0]))][:3]
angs = np.arctan2(K[:, 1], K[:, 0])
dev = angs - np.arange(3) * np.pi / 3 - angs[0]
base = angs[0] + np.angle(np.mean(np.exp(6j * dev))) / 6
kmag = np.hypot(*K.T).mean()
K_ideal = np.stack([kmag * np.array([math.cos(base + i * math.pi / 3), math.sin(base + i * math.pi / 3)]) for i in range(3)])
M = np.linalg.lstsq(K, K_ideal, rcond=None)[0]   # maps ideal offsets -> texture offsets
if args.no_strain_fix:
    M = np.eye(2)
print("lattice angles (deg):", np.round(np.degrees(angs), 2), " row spacing (px):", np.round(N / np.hypot(*K.T), 2))
print("strain correction M:", np.round(M, 4).tolist())

# spot centres
sm = ndimage.gaussian_filter(img, 2)
mx = ndimage.maximum_filter(sm, size=11, mode="wrap")
spots = np.argwhere((sm == mx) & (sm > sm.mean() + 0.5 * sm.std())).astype(np.float64)
c = np.array([N / 2, N / 2])
spots = spots[np.argsort(((spots - c) ** 2).sum(1))][: args.candidates]

r = np.linspace(rho0, rho1, 400) * px
w = np.linspace(-15, 15, 11)  # sample a strip across the seam ray (px)


def strip(apex, ang):
    d = (M @ np.array([math.cos(ang), math.sin(ang)]))[::-1]   # (row, col) direction
    n = (M @ np.array([-math.sin(ang), math.cos(ang)]))[::-1]
    pts = apex[:, None, None] + d[:, None, None] * r[None, :, None] + n[:, None, None] * w[None, None, :]
    return ndimage.map_coordinates(img, pts.reshape(2, -1), order=1, mode="grid-wrap")


best = (-2, None, None)
for apex in spots:
    for psi in np.radians(np.arange(0, 60, 1.0)):
        s1 = strip(apex, psi + sector / 2)
        s2 = strip(apex, psi - sector / 2)
        corr = np.corrcoef(s1, s2)[0, 1]
        if corr > best[0]:
            best = (corr, apex, psi)

corr, apex, psi = best

# local refinement: small extra strain, rotation and apex shift that best close the seam
from scipy.optimize import minimize
M0 = M.copy()


def neg_corr(x):
    global M
    M = M0 @ np.array([[1 + x[0], x[2]], [x[2], 1 + x[1]]])
    a = apex + x[4:6]
    return -np.corrcoef(strip(a, x[3] + sector / 2), strip(a, x[3] - sector / 2))[0, 1]


x0 = np.array([0, 0, 0, psi, 0, 0])
res = minimize(neg_corr, x0, method="Nelder-Mead",
               options=dict(maxiter=3000, xatol=1e-5, fatol=1e-5))
print(f"coarse seam correlation {corr:.3f} -> refined {-res.fun:.3f}")
neg_corr(res.x)
corr, psi, apex = -res.fun, res.x[3], apex + res.x[4:6]
print(f"best seam correlation {corr:.3f} (1 = invisible seam)")
print("shader uniforms:")
print(f"  apex_uv     = vec2({apex[1] / N:.5f}, {apex[0] / N:.5f})")
print(f"  sector_rot  = {psi:.5f}")
print(f"  tile_world  = {args.tile_world}")
print(f"  top_radius  = {args.top_radius}")
print(f"  lattice_fix = mat2(vec2({M[0,0]:.5f}, {M[1,0]:.5f}), vec2({M[0,1]:.5f}, {M[1,1]:.5f}))")

if args.preview:
    # flattened cone (sector) for a quick 2D look
    S = 1200
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    X = (xx - S / 2) / (S / 2) * rho1
    Y = (S - yy) / S * rho1 * 1.0
    rho = np.hypot(X, Y)
    phi = np.arctan2(X, Y)
    mask = (rho >= rho0) & (rho <= rho1) & (np.abs(phi) <= sector / 2)
    ang = psi + phi
    cc = apex[1] + rho * px * (M[0, 0] * np.cos(ang) + M[0, 1] * np.sin(ang))
    rr = apex[0] + rho * px * (M[1, 0] * np.cos(ang) + M[1, 1] * np.sin(ang))
    out = ndimage.map_coordinates(img, [rr.ravel(), cc.ravel()], order=1, mode="grid-wrap").reshape(S, S)
    out[~mask] = 0.35
    cv2.imwrite(args.preview, (out * 255).astype(np.uint8))
    print("preview", args.preview)
