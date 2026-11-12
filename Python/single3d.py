import random

import torch
import cv2
import numpy as np
from blue_noise import gen_blue_noise_slice

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

# cone shell (Godot cone x100): bottom radius, top radius, height, thickness. 1 unit = 1 pixel of the 2D runs
R1, R0, H, THICK = 100.0, 46.92, 314.0, 3.0
NI = round(2 * np.pi * R1)            # nodes around
NJ = round(np.hypot(H, R1 - R0)) + 1  # nodes along the slant
NK = round(THICK) + 1                 # nodes through the thickness

dt = 0.5
beta_randomization = 0.873
blue_epoch = 2
diff_rate = 0.005
use_blue_noise = True

# ---- mesh: structured grid on the cone, each cell cut into 6 tets (NewtonBioMorphX-style FEM) ----
I, J, K = np.meshgrid(np.arange(NI), np.arange(NJ), np.arange(NK), indexing="ij")
theta, t = 2 * np.pi * I / NI, J / (NJ - 1)
tilt = np.arctan2(R1 - R0, H)
off = (K / (NK - 1) - 0.5) * THICK
rad = R1 + (R0 - R1) * t + off * np.cos(tilt)
pos = torch.tensor(np.stack([rad * np.cos(theta), -H / 2 + H * t + off * np.sin(tilt), rad * np.sin(theta)], -1).reshape(-1, 3), device=device, dtype=torch.float32)

ci, cj, ck = (x.ravel() for x in np.meshgrid(np.arange(NI), np.arange(NJ - 1), np.arange(NK - 1), indexing="ij"))
node = lambda di, dj, dk: (((ci + di) % NI) * NJ + cj + dj) * NK + ck + dk   # wraps around the cone
tets = []
for p in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
    e1 = np.eye(3, dtype=int)[p[0]]
    e2 = e1 + np.eye(3, dtype=int)[p[1]]
    tets.append(np.stack([node(0, 0, 0), node(*e1), node(*e2), node(1, 1, 1)], 1))
tets = torch.tensor(np.concatenate(tets), device=device)

# per tet: shape-function gradients and volume (paper Eq. 20-23); per node: volume (Eq. 22)
Jx = (pos[tets][:, 1:] - pos[tets][:, :1]).transpose(1, 2)
Ginv = torch.linalg.inv(Jx)
G = torch.cat([-Ginv.sum(1, keepdim=True), Ginv], 1)
Vt = torch.linalg.det(Jx).abs() / 6
Vn = torch.zeros(len(pos), device=device).index_add_(0, tets.ravel(), Vt.repeat_interleave(4) / 4)

# Laplacian matrix: flux through each tet face to its nodes (Eq. 34), divided by node volume
n = len(pos)
vals = (-Vt[:, None, None] * G @ G.transpose(1, 2) / Vn[tets][:, :, None]).reshape(-1)
key, inv = torch.unique((tets[:, :, None] * n + tets[:, None, :]).reshape(-1), return_inverse=True)
vals = torch.zeros(len(key), device=device).index_add_(0, inv, vals)
lap_matrix = torch.sparse_coo_tensor(torch.stack([key // n, key % n]), vals, (n, n), is_coalesced=True).to_sparse_csr()
del Jx, Ginv, G, vals, key, inv

# narrow cells at the top need a smaller dt than the 2D grid (explicit Euler: dt * D * lambda_max < 2)
v = torch.rand(n, device=device)
for _ in range(60):
    v = -(lap_matrix @ v); lam = v.norm().item(); v /= lam
print(f"{n} nodes, {len(tets)} tets, lambda_max {lam:.1f}")

# ---- same as single*.py from here on; a, b are (NI, NJ, NK) instead of (N, N) ----
blue_step = 0
blue_noise = gen_blue_noise_slice(NI, NJ, 4, 0.8).to(device, torch.float32)
beta = 12.0 + (torch.rand((NI, NJ, NK), device=device) * 0.1 - 0.05)
shiftx, shifty = 0, 0


def step_beta(step=1):
    global beta, blue_step, shiftx, shifty

    if not use_blue_noise:
        beta = 12.0 + beta_randomization * (torch.rand((NI, NJ, NK), device=device) * 0.1 - 0.05)
        return

    blue_step += 1
    if random.randint(0, 1) == 0:
        shiftx += random.randint(-2, 2) * step
    else:
        shifty += random.randint(-2, 2) * step

    noise_this = torch.roll(blue_noise[:, :, (blue_step // blue_epoch) % blue_noise.shape[2]], shifts=(shiftx % NI, shifty % NJ), dims=(0, 1))
    beta = 12.0 + beta_randomization * (0.1 * noise_this[:, :, None] - 0.05)
step_beta()


def get_laplacian(Z):
    return (lap_matrix @ Z.reshape(-1)).view(NI, NJ, NK)


def f(a, b, diff_rate):
    return diff_rate * (16.0 - a * b)


def g(a, b, diff_rate, beta):
    return diff_rate * (a * b - b - beta)


def step(a, b, r, s):
    a += dt * (f(a, b, diff_rate) + r * get_laplacian(a))
    b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

    a.clamp_(min=0)
    b.clamp_(min=0)


def sim(r, s):
    global dt
    dt = min(dt, 1.8 / (max(r, s) * lam))
    print("dt", dt)

    a = torch.full((NI, NJ, NK), 4.0, device=device)
    b = torch.full((NI, NJ, NK), 4.0, device=device)

    for i in range(10000000):
        step(a, b, r, s)

        if i % 250 == 0:
            img_np = a[:, :, -1].T.flip(0).cpu().numpy()   # outer surface, unrolled by grid index
            print(i, img_np.min(), img_np.max())
            img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
            cv2.imshow("RD", img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

        step_beta()

    cv2.imwrite("img_3d.png", img * 255)
    np.savez("rd3d_state.npz", pos=pos.cpu().numpy(), tets=tets.cpu().numpy(), a=a.cpu().numpy().reshape(-1))


sim(r=0.5, s=0.0625)
