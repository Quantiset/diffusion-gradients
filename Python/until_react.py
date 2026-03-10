import torch
import cv2
import numpy as np
from blue_noise import gen_blue_noise
from collections import deque

from anisotropy import find_stripiness
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

N = 256
dt = 0.004

diff_rate = 1.0
blue_steps_until_next_iter = 120

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)

plots = []
diffs = []

blue_step = 0
blue_noise = gen_blue_noise(N, N, N).to(device)
beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)

use_blue_noise = True

def step_beta():
    global beta, blue_step

    if not use_blue_noise:
        return
    blue_step += 1
    blue_step = blue_step % blue_noise.shape[2]
    beta = 12.0 + 4 * (0.1 * blue_noise[:, :, blue_step] -  0.05)
step_beta()

def get_laplacian(Z):
    Ztop = Z.roll(1, 0)
    Zleft = Z.roll(1, 1)
    Zbottom = Z.roll(-1, 0)
    Zright = Z.roll(-1, 1)
    return Ztop + Zleft + Zbottom + Zright - 4 * Z

def f(a, b, diff_rate):
    return diff_rate * (16.0 - a * b)

def g(a, b, diff_rate, beta):
    return diff_rate * (a * b - b - beta)

def sim(r, s):

    print(np.linspace(4, 7.8, 22))

    for r in np.linspace(4, 7.8, 22):
        a = torch.full((N, N), 4.0, device=device)
        b = torch.full((N, N), 4.0, device=device)

        vals = deque()
        for i in range(10_000_000):
            a += dt * (f(a, b, diff_rate) + r * s * get_laplacian(a))
            b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

            a.clamp_(min=0)
            b.clamp_(min=0)

            if i % 100 == 0:
                img_np = a.cpu().numpy()
                
                min_val, max_val = img_np.min(), img_np.max()
                print(f"r: {r:.3f}, step: {i}, min: {min_val:.4f}, max: {max_val:.4f}")
                deque.append(vals, (min_val, max_val))
                if len(vals) > 20:
                    vals.popleft()
                    diff = (sum(vals[i][1] - vals[i+1][1] for i in range(len(vals)-1))/len(vals)*10000)
                    if diff > 100:
                        plots.append((r, i))
                        break
                    diffs.append(diff)

                img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
                cv2.imshow("RD", img)
                
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if key == ord('f'):
                    A_fft = torch.fft.fft2(a)
                    A_fft_shifted = torch.fft.fftshift(A_fft)

                    magnitude = torch.log1p(torch.abs(A_fft_shifted))
                    mag_np = magnitude.detach().cpu().numpy()

                    mag_np = (mag_np - mag_np.min()) / (mag_np.max() - mag_np.min() + 1e-6)

                    cv2.imshow("Fourier", mag_np)
                    cv2.waitKey(1)
            
            if i % blue_steps_until_next_iter == 0 and i > 10:
                step_beta()

        # if i == 999:
        #     with open("data.txt", "a") as file:
        #         line = f"{r} {s} {img_np.min()} {img_np.max()} {find_stripiness(a, device)}\n"
        #         print(line)
        #         file.write(line)

sim(r=4, s=4)
print(plots)