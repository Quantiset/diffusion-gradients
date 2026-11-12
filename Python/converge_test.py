"""Run the batch2.py model (explicit Euler, 9-point Laplacian, noisy beta) until the
pattern stops changing, for blue- or white-noise beta.

Converged = the thresholded pattern (a > mean) flips fewer than --tol of its pixels
between checkpoints, for --patience checkpoints in a row. (beta is re-noised every
step, so the raw field never stops jiggling; the pattern layout does.)

python converge_test.py --noise blue  --r 0.5 --s 0.0625
python converge_test.py --noise white --r 0.5 --s 0.0625
"""
import argparse
import random
import time

import cv2
import numpy as np
import torch

from blue_noise import gen_blue_noise_slice

p = argparse.ArgumentParser()
p.add_argument("--noise", choices=["blue", "white"], required=True)
p.add_argument("--N", type=int, default=1024)
p.add_argument("--r", type=float, default=0.5, help="D_a (r/s = 8 with defaults)")
p.add_argument("--s", type=float, default=0.0625, help="D_b")
p.add_argument("--dt", type=float, default=0.5)
p.add_argument("--beta-rand", type=float, default=0.873)
p.add_argument("--blue-epoch", type=int, default=2)
p.add_argument("--blue-slices", type=int, default=4)
p.add_argument("--blue-alpha", type=float, default=0.8)
p.add_argument("--check", type=int, default=20_000, help="steps between convergence checks")
p.add_argument("--tol", type=float, default=2e-4)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--max-steps", type=int, default=4_000_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fp64", action="store_true", help="match batch2.py precision (slow on consumer GPUs)")
p.add_argument("--out", required=True)
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
dev = torch.device("cuda")
dtype = torch.float64 if args.fp64 else torch.float32
torch.set_default_dtype(dtype)
N, dt, d = args.N, args.dt, 0.005

a = torch.full((N, N), 4.0, device=dev)
b = torch.full((N, N), 4.0, device=dev)
ker = torch.tensor([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]], device=dev).view(1, 1, 3, 3)

if args.noise == "blue":
    blue = gen_blue_noise_slice(N, N, args.blue_slices, args.blue_alpha).to(dev, dtype)
blue_step, sx, sy = 0, 0, 0


def next_beta():
    global blue_step, sx, sy
    if args.noise == "white":
        return 12.0 + args.beta_rand * (torch.rand((N, N), device=dev) * 0.1 - 0.05)
    blue_step += 1
    if random.randint(0, 1) == 0:
        sx += random.randint(-2, 2)
    else:
        sy += random.randint(-2, 2)
    sl = blue[:, :, (blue_step // args.blue_epoch) % blue.shape[2]]
    return 12.0 + args.beta_rand * (0.1 * torch.roll(sl, (sx % N, sy % N), (0, 1)) - 0.05)


def lap(Z):
    return torch.nn.functional.conv2d(torch.nn.functional.pad(Z[None, None], (1, 1, 1, 1), mode="circular"), ker)[0, 0]


@torch.compile
def step(a, b, beta):
    la, lb = lap(a), lap(b)
    a2 = (a + dt * (d * (16 - a * b) + args.r * la)).clamp(min=0)
    b2 = (b + dt * (d * (a * b - b - beta) + args.s * lb)).clamp(min=0)
    return a2, b2


def save(path, x):
    x = x.cpu().numpy()
    cv2.imwrite(path, ((x - x.min()) / (x.max() - x.min() + 1e-6) * 255).astype(np.uint8))


t0 = time.time()
prev = None
calm = 0
beta = next_beta()
for i in range(1, args.max_steps + 1):
    a, b = step(a, b, beta)
    beta = next_beta()
    if i % args.check == 0:
        mask = a > a.mean()
        contrast = (a.max() - a.min()).item()
        flips = float("nan") if prev is None else (mask != prev).float().mean().item()
        prev = mask
        calm = calm + 1 if (contrast > 1 and flips < args.tol) else 0
        print(f"step {i:8d}  contrast {contrast:6.3f}  flipped {flips:.2e}  calm {calm}  {time.time()-t0:6.0f}s", flush=True)
        if i % (args.check * 10) == 0:
            save(args.out.replace(".png", f"_{i:08d}.png"), a)
        if calm >= args.patience:
            break

save(args.out, a)
np.save(args.out.replace(".png", ".npy"), a.cpu().numpy())
print(f"{'converged' if calm >= args.patience else 'NOT converged (hit max steps)'} at step {i}, {time.time()-t0:.0f}s -> {args.out}")
