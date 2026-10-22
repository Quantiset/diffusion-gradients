import random

import sys
from scipy.fftpack import fft2
import torch
import cv2
import numpy as np
from blue_noise import gen_blue_noise, gen_blue_noise_slice, plot_3d
from matplotlib import pyplot as plt

# from anisotropy import find_stripiness

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

def until_reaction(r): return int(7336.347 / (r - 3.6470) + 3739.7676)

# dt = 1.5, beta = 3, beta_random = 0.5
N = int(sys.argv[1])
dt = 0.5
beta_randomization = float(sys.argv[2])
blue_epoch = int(sys.argv[3])

diff_rate = 0.005
blue_steps_per_reaction_step = 2 / until_reaction(4)
blue_steps_until_next_iter = None

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)

random_points = [(torch.randint(0, N, ()), torch.randint(0, N, ())) for _ in range(10)]
points_values = []
middles = []

blue_step = 0
blue_noise = gen_blue_noise_slice(N, N, N//int(sys.argv[4]), 0.8).to(device)
beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)

use_blue_noise = bool(int(sys.argv[7]))

size_scale = float(sys.argv[8])

shiftx, shifty = 0, 0

ker = torch.tensor([
    [0.25, 0.5, 0.25],
    [0.5, -3, 0.5],
    [0.25, 0.5, 0.25]
], device=device).view(1,1,3,3)

def step_beta(step=1):
    global beta, blue_step, shiftx, shifty

    if not use_blue_noise:
        beta = 12.0 + beta_randomization * (torch.rand((N, N), device=device) * 0.1 - 0.05)
        return

    blue_step += 1
    
    if random.randint(0,1) == 0:
        shiftx += random.randint(-2, 2) * step
    else:
        shifty += random.randint(-2, 2) * step
    shifts = (int(shiftx) % N, int(shifty) % N)

    noise_this = torch.roll(blue_noise[:, :, (blue_step // blue_epoch) % blue_noise.shape[2]], shifts=shifts, dims=(0, 1)) 
    beta = 12.0 + beta_randomization * (0.1 * noise_this -  0.05)
step_beta()

def get_laplacian(Z):
    Zp = torch.nn.functional.pad(
        Z.unsqueeze(0).unsqueeze(0),
        (1,1,1,1),
        mode="circular"
    )
    return torch.nn.functional.conv2d(Zp, ker).squeeze()
    cross = (torch.roll(Z,1,0) + torch.roll(Z,-1,0) + torch.roll(Z,1,1) + torch.roll(Z,-1,1) - 4*Z)
    diag = ( 
        torch.roll(torch.roll(Z,1,0),1,1) + torch.roll(torch.roll(Z,1,0),-1,1) +
        torch.roll(torch.roll(Z,-1,0),1,1) + torch.roll(torch.roll(Z,-1,0),-1,1) - 4*Z
    )
    return (cross + 0.5*diag)/2

def f(a, b, diff_rate):
    return diff_rate * (16.0 - a * b)

def g(a, b, diff_rate, beta):
    return diff_rate * (a * b - b - beta)

@torch.compile
def step(a, b, r, s):
    a += dt * (f(a, b, diff_rate) + size_scale * r * get_laplacian(a))
    b += dt * (g(a, b, diff_rate, beta) + size_scale * s * get_laplacian(b))

    a.clamp_(min=0)
    b.clamp_(min=0)

_activated = False
def sim(r, s):
    global _activated
    
    a = torch.full((N, N), 4.0, device=device)
    b = torch.full((N, N), 4.0, device=device)

    for i in range(8_500_000):
        if i % 400 == 1:
            img_np = a.cpu().numpy()
            img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
            cv2.imshow("RD", img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
        step(a, b, r, s)
        step_beta()

    img_np = a.cpu().numpy()
    img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
    # save the image

    cv2.imwrite(f"img_{float(sys.argv[1])}_{float(sys.argv[2])}_{float(sys.argv[3])}_{float(sys.argv[4])}_{float(sys.argv[5])}_{float(sys.argv[6])}_{int(sys.argv[7])}_{float(sys.argv[8])}.png", img * 255)

sim(r=float(sys.argv[5]), s=float(sys.argv[6]))


"""
a) we get voronoi patterns because of race around conditions and voronoi
b) so slow because of meanderings

two reasons:

1) rushing
2) rely on random noise -- but create race around conditions


possible idea is that we slow down the process: but that can't get rid of that

we want this to be repeated in the time domain.

but practical problem is how to scale -- we don't have solid insight but we can noise shift 8 reps per plane

EXPERTIMENTS:
1)
try different patterns with noise with differing resolutions 

2)
2048 default with differing slopes (with shifting)

shift doesn't matter

3)
(ONLY DOTS with multiple resolutions)
use fewer and fewer z-direction planes
"""
