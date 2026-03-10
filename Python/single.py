import torch
import cv2
import numpy as np
from blue_noise import gen_blue_noise
from matplotlib import pyplot as plt

from anisotropy import find_stripiness

# torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

def until_reaction(r): return int(7336.347 / (r - 3.6470) + 3739.7676)

N = 256
dt = 1.0

diff_rate = 0.005
blue_steps_per_reaction_step = 15 / until_reaction(4)
blue_steps_until_next_iter = None

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)

random_points = [(torch.randint(0, N, ()), torch.randint(0, N, ())) for _ in range(10)]
points_values = []

blue_step = 0
blue_noise = gen_blue_noise(N, N, N).to(device)
beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)

use_blue_noise = True 

def step_beta():
    global beta, blue_step

    if not use_blue_noise:
        beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05) * 8
        return

    blue_step += 1
    blue_step = blue_step % blue_noise.shape[2]
    beta = 12.0 + (0.1 * blue_noise[:, :, blue_step] -  0.05) * 8
step_beta()

def get_laplacian(Z):
    cross = (
        torch.roll(Z,1,0) +
        torch.roll(Z,-1,0) +
        torch.roll(Z,1,1) +
        torch.roll(Z,-1,1) -
        4*Z
    )

    diag = (
        torch.roll(torch.roll(Z,1,0),1,1) +
        torch.roll(torch.roll(Z,1,0),-1,1) +
        torch.roll(torch.roll(Z,-1,0),1,1) +
        torch.roll(torch.roll(Z,-1,0),-1,1) -
        4*Z
    )

    return (cross + 0.5*diag)/2

def f(a, b, diff_rate):
    return diff_rate * (16.0 - a * b)

def g(a, b, diff_rate, beta):
    return diff_rate * (a * b - b - beta)

def sim(r, s):
    
    a = torch.full((N, N), 4.0, device=device)
    b = torch.full((N, N), 4.0, device=device)

    for i in range(10000000):
        a += dt * (f(a, b, diff_rate) + r * get_laplacian(a))
        b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

        a.clamp_(min=0)
        b.clamp_(min=0)

        if i % 50 == 0:
            # update temporal spectra
            for x, y in random_points:
                points_values.append((x.item(), y.item(), a[x, y].item()))

        if i % 100 == 0:
            img_np = a.cpu().numpy()
            print(img_np.min(), img_np.max())
            img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
            cv2.imshow("RD", img)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('f'):
                A_fft = torch.fft.fft2(a)
                A_fft_shifted = torch.fft.fftshift(A_fft)

                magnitude = torch.log1p(torch.abs(A_fft_shifted)**2)
                mag_np = magnitude.detach().cpu().numpy()

                mag_np = (mag_np - mag_np.min()) / (mag_np.max() - mag_np.min() + 1e-6)

                cv2.imshow("Fourier", mag_np)
                cv2.waitKey(1)
            
            if key == ord('k'):
                lap = get_laplacian(a)

                lap_np = lap.detach().cpu().numpy().flatten()

                plt.figure()
                plt.hist(lap_np, bins=100)
                plt.title("Histogram of Laplacian Strengths")
                plt.xlabel("Laplacian Value")
                plt.ylabel("Frequency")
                plt.show()
        
        step_beta()

        # if i == 999:
        #     with open("data.txt", "a") as file:
        #         line = f"{r} {s} {img_np.min()} {img_np.max()} {find_stripiness(a, device)}\n"
        #         print(line)
        #         file.write(line)

sim(r=0.25, s=0.0625)

points_values_np = np.array(points_values)
with open("points_values.txt", "w") as f:
    for val in points_values_np:
        f.write(f"{val}\n")