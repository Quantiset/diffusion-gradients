"""Fast, regular Turing patterns near threshold (r = D_a/D_b just above 4).

Same model as batch.py:  a' = d(16 - ab) + r*s*lap(a),  b' = d(ab - b - beta) + s*lap(b)
with the same 9-point Laplacian, but:

1. Semi-implicit (IMEX) time stepping: diffusion is solved exactly in Fourier space,
   reaction is explicit. Steady states and the Turing threshold are identical to the
   explicit scheme, but dt can be ~20x larger.
2. Moving-front quench (a gradient plot in time): a band of higher r sweeps across the
   domain in y. Behind it the pattern crystallises row by row into one hexagonal grain
   instead of nucleating everywhere at random. Afterwards r is set to the target
   everywhere and the pattern relaxes until it stops changing.

Usage:
  python rd_front.py --mode front --rho 4.01
  python rd_front.py --mode uniform --rho 4.01      # old behaviour, for comparison
"""
import argparse
import time

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import Delaunay

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=1024)
p.add_argument("--mode", choices=["front", "uniform"], default="front")
p.add_argument("--rho", type=float, default=4.01, help="target D_a/D_b")
p.add_argument("--rho-hi", type=float, default=4.1, help="D_a/D_b behind the moving front")
p.add_argument("--rho-ahead", type=float, default=3.97, help="D_a/D_b ahead of the front (below 4 = stable, so nothing nucleates there)")
p.add_argument("--speed", type=float, default=0.001, help="front speed, pixels per time unit")
p.add_argument("--s", type=float, default=0.0625, help="D_b")
p.add_argument("--d", type=float, default=0.005, help="reaction rate (diff_rate)")
p.add_argument("--dt", type=float, default=10.0)
p.add_argument("--tol", type=float, default=3e-7, help="stop when max|da/dt| < tol")
p.add_argument("--max-steps", type=int, default=400_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
p.add_argument("--snap", type=int, default=0, help="save a snapshot every N steps")
p.add_argument("--log", type=int, default=5000)
args = p.parse_args()

torch.manual_seed(args.seed)
dev = torch.device("cuda")
dtype = torch.float32
N, d, s, dt = args.N, args.d, args.s, args.dt

# Fourier symbol of the 9-point kernel [[.25,.5,.25],[.5,-3,.5],[.25,.5,.25]]
k = 2 * np.pi * torch.fft.fftfreq(N, device=dev).to(dtype)
KY, KX = torch.meshgrid(k, k, indexing="ij")
lap_hat = -3 + torch.cos(KX) + torch.cos(KY) + torch.cos(KX) * torch.cos(KY)

yy = torch.arange(N, device=dev, dtype=dtype).unsqueeze(1).expand(N, N)

a = 4 + 1e-4 * torch.randn(N, N, device=dev, dtype=dtype)
b = 4 + 1e-4 * torch.randn(N, N, device=dev, dtype=dtype)
beta = 12.0

if args.mode == "front":
    # small seed at the bottom edge so the first rows start promptly
    a[:24] += 0.05 * torch.randn(24, N, device=dev, dtype=dtype)


def rho_field(t):
    if args.mode == "uniform":
        return None
    y_f = args.speed * t
    if y_f > N + 60:
        return None
    w = 8.0  # front width (px)
    band = torch.sigmoid((y_f - yy) / w)          # 1 behind the front, 0 ahead
    tail = torch.sigmoid((y_f - 3 * w - yy) / w)  # well behind: hold at rho_hi until the sweep ends
    return args.rho_ahead + (args.rho_hi - args.rho_ahead) * torch.maximum(band, tail)


def step(a, b, rho):
    fa = d * (16 - a * b)
    fb = d * (a * b - b - beta)
    if rho is None:
        Da = args.rho * s
        a_new = torch.fft.ifft2(torch.fft.fft2(a + dt * fa) / (1 - dt * Da * lap_hat)).real
    else:
        # variable D_a: split into constant implicit part + explicit remainder (stable since rho <= rho_hi)
        Dmax = args.rho_hi * s
        la = torch.fft.ifft2(torch.fft.fft2(a) * lap_hat).real
        rhs = a + dt * (fa + (rho * s - Dmax) * la)
        a_new = torch.fft.ifft2(torch.fft.fft2(rhs) / (1 - dt * Dmax * lap_hat)).real
    b_new = torch.fft.ifft2(torch.fft.fft2(b + dt * fb) / (1 - dt * s * lap_hat)).real
    return a_new.clamp_(min=0), b_new.clamp_(min=0)


def regularity(img):
    """Fraction of spots with exactly 6 Delaunay neighbours (1.0 = perfect hex)."""
    sm = ndimage.gaussian_filter(img, 2)
    mx = ndimage.maximum_filter(sm, size=11, mode="wrap")
    pts = np.argwhere((sm == mx) & (sm > sm.mean() + 0.5 * sm.std()))
    if len(pts) < 20:
        return 0.0, len(pts)
    tri = Delaunay(pts)
    indptr, nbr = tri.vertex_neighbor_vertices
    deg = np.diff(indptr)
    lo, hi = pts.min(0) + 30, pts.max(0) - 30
    inner = np.all((pts > lo) & (pts < hi), axis=1)
    return float((deg[inner] == 6).mean()), len(pts)


t0 = time.time()
t = 0.0
for i in range(args.max_steps):
    rho = rho_field(t)
    a_old = a
    a, b = step(a, b, rho)
    t += dt
    if args.snap and i % args.snap == 0:
        sn = a.cpu().numpy(); sn = (sn - sn.min()) / (sn.max() - sn.min() + 1e-6)
        cv2.imwrite((args.out or "snap.png").replace(".png", f"_{i:07d}.png"), (sn * 255).astype(np.uint8))
    if i % args.log == 0:
        rate = (a - a_old).abs().max().item() / dt
        print(f"step {i:7d}  t={t:9.0f}  front={'done' if rho is None else f'{args.speed*t:6.0f}px'}  "
              f"max|da/dt|={rate:.2e}  a=[{a.min().item():.2f},{a.max().item():.2f}]  {time.time()-t0:5.0f}s",
              flush=True)
        if rho is None and rate < args.tol and a.max() - a.min() > 1:
            break

img = a.cpu().numpy()
reg, nspots = regularity(img)
elapsed = time.time() - t0
print(f"done: {i+1} steps, {elapsed:.0f}s, {nspots} spots, hex-regularity {reg:.3f}")

out = args.out or f"rd_{args.mode}_N{N}_rho{args.rho}.png"
norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
cv2.imwrite(out, (norm * 255).astype(np.uint8))
np.save(out.replace(".png", ".npy"), img)
print("saved", out)
