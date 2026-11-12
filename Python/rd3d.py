"""

The Laplacian follows NewtonBioMorphX (Jahanbakhsh & Milinkovitch 2027, Sec. 3 and
src/nvrtc_kernels/compute_force_default.cu): 

python rd3d.py --noise blue  --shape cone --out runs/r8_3d/blue.png
python rd3d.py --noise white --shape box  --out runs/r8_3d/white_box.png
"""
import argparse
import math
import random
import time

import cv2
import numpy as np
import torch

from blue_noise import gen_blue_noise_slice

p = argparse.ArgumentParser()
p.add_argument("--noise", choices=["blue", "white"], required=True)
p.add_argument("--shape", choices=["cone", "box"], default="cone")
p.add_argument("--r", type=float, default=0.5)      # D_a  (r/s = 8)
p.add_argument("--s", type=float, default=0.0625)   # D_b
p.add_argument("--max-steps", type=int, default=4_000_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()

h = 1.0
THICK = 64.0 
R1, R0, H = 100.0, 46.92, 314.0 
BOX = 64
dt_max, d, beta_rand = 0.5, 0.005, 0.873
blue_epoch, blue_slices, blue_alpha = 2, 4, 0.8
check, tol, patience = 20_000, 2e-4, 3

random.seed(args.seed)
torch.manual_seed(args.seed)
dev = torch.device("cuda")
torch.set_default_dtype(torch.float32)

nk = round(THICK / h)
if args.shape == "cone":
    ni, nj, wrap = round(2 * math.pi * R1 / h), round(math.hypot(H, R1 - R0) / h), True
    I, J, K = np.meshgrid(np.arange(ni), np.arange(nj + 1), np.arange(nk + 1), indexing="ij")
    th, t = 2 * math.pi * I / ni, J / nj 
    beta_n = math.atan2(R1 - R0, H)
    off = (K / nk - 0.5) * THICK
    R = R1 + (R0 - R1) * t + off * math.cos(beta_n)
    pos = np.stack([R * np.cos(th), -H / 2 + H * t + off * math.sin(beta_n), R * np.sin(th)], -1)
else:
    ni, nj, wrap = BOX, BOX, False
    I, J, K = np.meshgrid(np.arange(ni + 1), np.arange(nj + 1), np.arange(nk + 1), indexing="ij")
    pos = np.stack([I, J, K * THICK / nk], -1) * h
shape = pos.shape[:3]
pos = torch.tensor(pos.reshape(-1, 3), device=dev, dtype=torch.float32)


def idx(i, j, k):
    if wrap:
        i = i % shape[0]
    return (i * shape[1] + j) * shape[2] + k


ci, cj, ck = (x.ravel() for x in np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij"))
corner = lambda b: idx(ci + b[0], cj + b[1], ck + b[2])
tets = []
for a, b in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
    e1 = [0, 0, 0]; e1[a] = 1
    e2 = list(e1); e2[b] = 1
    tets.append(np.stack([corner((0, 0, 0)), corner(e1), corner(e2), corner((1, 1, 1))], 1))
tets = torch.tensor(np.concatenate(tets), device=dev)

x = pos[tets]
Jx = (x[:, 1:] - x[:, :1]).transpose(1, 2) 
Jinv = torch.linalg.inv(Jx)
G = torch.cat([-Jinv.sum(1, keepdim=True), Jinv], 1) 
Vt = torch.linalg.det(Jx).abs() / 6 
Vn = torch.zeros(len(pos), device=dev).index_add_(0, tets.ravel(), Vt.repeat_interleave(4) / 4)

n = len(pos)
Ke = (-Vt[:, None, None] * G @ G.transpose(1, 2) / Vn[tets][:, :, None]).reshape(-1)   # (T*16)
del G, Jinv, x
key, inv = torch.unique((tets[:, :, None] * n + tets[:, None, :]).reshape(-1), return_inverse=True)
vals = torch.zeros(len(key), device=dev).index_add_(0, inv, Ke)   # merge duplicate (row, col)
del Ke, inv
L = torch.sparse_coo_tensor(torch.stack([key // n, key % n]), vals, (n, n), is_coalesced=True).to_sparse_csr()
del key, vals
torch.cuda.empty_cache()


def lap(c):
    return L @ c


v = torch.rand(len(pos), device=dev)
for _ in range(60):
    v = -lap(v); lam = v.norm().item(); v /= lam
dt = min(dt_max, 1.8 / (max(args.r, args.s) * lam))
print(f"{len(pos)} nodes, {len(tets)} tets, lambda_max {lam:.2f}, dt {dt:.3f}")

if args.noise == "blue":
    blue = gen_blue_noise_slice(shape[0], shape[1], blue_slices, blue_alpha).to(dev, torch.float32)
step_n, sx, sy = 0, 0, 0


def next_beta():
    global step_n, sx, sy
    if args.noise == "white":
        return 12.0 + beta_rand * (torch.rand(len(pos), device=dev) * 0.1 - 0.05)
    step_n += 1
    if random.randint(0, 1) == 0:
        sx += random.randint(-2, 2)
    else:
        sy += random.randint(-2, 2)
    sl = torch.roll(blue[:, :, (step_n // blue_epoch) % blue_slices], (sx, sy), (0, 1))
    return 12.0 + beta_rand * (0.1 * sl[:, :, None].expand(shape).reshape(-1) - 0.05)


def save(path, c):  
    img = c.view(shape)[:, :, -1].T.flip(0).cpu().numpy()
    cv2.imwrite(path, ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(np.uint8))


a = torch.full((len(pos),), 4.0, device=dev)
b = torch.full((len(pos),), 4.0, device=dev)
t0, prev, calm = time.time(), None, 0
for it in range(1, args.max_steps + 1):
    beta = next_beta()
    a, b = ((a + dt * (d * (16 - a * b) + args.r * lap(a))).clamp(min=0),
            (b + dt * (d * (a * b - b - beta) + args.s * lap(b))).clamp(min=0))
    if it % check == 0:
        mask = a > a.mean()
        flips = float("nan") if prev is None else (mask != prev).float().mean().item()
        prev = mask
        calm = calm + 1 if (a.max() - a.min() > 1 and flips < tol) else 0
        print(f"step {it:8d}  t {it*dt:9.0f}  flipped {flips:.2e}  calm {calm}  {time.time()-t0:6.0f}s", flush=True)
        if it % (check * 10) == 0:
            save(args.out.replace(".png", f"_{it:08d}.png"), a)
        if calm >= patience:
            break

save(args.out, a)
np.savez(args.out.replace(".png", ".npz"), pos=pos.cpu().numpy(), tets=tets.cpu().numpy(),
         a=a.cpu().numpy(), shape=np.array(shape))

o = np.arange(len(pos)).reshape(shape)[:, :, -1]
o1 = np.roll(o, -1, 0)                                      # next i (wraps around the cone)
quads = np.stack([o[:ni, :-1], o1[:ni, :-1], o1[:ni, 1:], o[:ni, 1:]], -1).reshape(-1, 4)
used, inv = np.unique(quads, return_inverse=True)
faces = inv.reshape(-1, 4)
av = a.cpu().numpy()[used]
col = ((av - av.min()) / (av.max() - av.min() + 1e-6) * 255).astype(np.uint8)
P = pos.cpu().numpy()[used]
with open(args.out.replace(".png", ".ply"), "w") as fh:
    fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
             f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
             f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    np.savetxt(fh, np.c_[P, col, col, col], fmt="%.3f %.3f %.3f %d %d %d")
    np.savetxt(fh, np.c_[np.full(len(faces), 4), faces], fmt="%d")
print(f"{'converged' if calm >= patience else 'NOT converged'} at step {it} (t={it*dt:.0f}), {time.time()-t0:.0f}s -> {args.out}")
