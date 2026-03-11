import random

from scipy.fftpack import fft2
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
dt = 0.5

diff_rate = 0.005
blue_steps_per_reaction_step = 2 / until_reaction(4)
blue_steps_until_next_iter = None

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)

random_points = [(torch.randint(0, N, ()), torch.randint(0, N, ())) for _ in range(10)]
points_values = []
middles = []

blue_step = 0
blue_noise = gen_blue_noise(N, N, N//2).to(device)
beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)
beta_randomization = 8.0

use_blue_noise = True

shiftx, shifty = 0, 0


def step_beta(step=N):
    global beta, blue_step, shiftx, shifty

    if not use_blue_noise:
        beta = 12.0 + beta_randomization * (torch.rand((N, N), device=device) * 0.1 - 0.05)
        return

    blue_step += 1
    shiftx += random.uniform(0, 1) * step
    shifty += random.uniform(0, 1) * step
    shifts = (int(shiftx) % N, int(shifty) % N)

    noise_this = torch.roll(blue_noise[:, :, 0], shifts=shifts, dims=(0, 1)) 
    beta = 12.0 + beta_randomization * (0.1 * noise_this -  0.05)
step_beta()

def get_laplacian(Z):
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

_activated = False
def sim(r, s):
    global _activated
    
    a = torch.full((N, N), 4.0, device=device)
    b = torch.full((N, N), 4.0, device=device)

    for i in range(10000000):
        a += dt * (f(a, b, diff_rate) + r * get_laplacian(a))
        b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

        a.clamp_(min=0)
        b.clamp_(min=0)

        if i % 10 == 0:
            for x, y in random_points:
                points_values.append((x.item(), y.item(), beta[x, y].item()))
            points_values.append(("---", "---", "---"))

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
            
        if not _activated:
            middles.append(beta[N-100:N+100, N-100:N+100].clone().cpu())
        if i == N:

            _activated = True
            beta_cpu = torch.stack(middles[N//2:3*N//2])

            field_vis = beta_cpu[beta_cpu.shape[0] // 2].numpy()
            field_vis = (field_vis - field_vis.min()) / (field_vis.max() - field_vis.min() + 1e-6)

            plt.figure()
            plt.imshow(field_vis, aspect='auto', cmap='gray')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Middle time slice of beta volume")
            plt.colorbar()
            plt.show()

            T = beta_cpu.shape[0]

            field = beta_cpu - beta_cpu.mean()

            fft3 = torch.fft.fftn(field)
            power3 = torch.abs(fft3) ** 2

            ft = torch.fft.fftfreq(T)
            fx = torch.fft.fftfreq(field.shape[1])
            fy = torch.fft.fftfreq(field.shape[2])

            kt, kx, ky = torch.meshgrid(ft, fx, fy, indexing='ij')

            freqs = torch.sqrt(kt**2 + kx**2 + ky**2)

            freqs_flat = freqs.flatten()
            p_flat = power3.flatten()

            mask = freqs_flat > 0
            k_np = freqs_flat[mask].detach().cpu().numpy()
            p_np = p_flat[mask].detach().cpu().numpy()

            num_bins = 100
            bins = np.linspace(k_np.min(), k_np.max(), num_bins)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            radial_power = np.zeros(num_bins - 1)
            for j in range(len(bins) - 1):
                mask = (k_np >= bins[j]) & (k_np < bins[j+1])
                if np.any(mask):
                    radial_power[j] = p_np[mask].mean()
                else:
                    radial_power[j] = np.nan
            k_fit = bin_centers[(bin_centers > np.percentile(bin_centers, 15)) & (bin_centers < np.percentile(bin_centers, 90))]
            radial_power_fit = radial_power[(bin_centers > np.percentile(bin_centers, 15)) & (bin_centers < np.percentile(bin_centers, 90))]

            log_k = np.log(k_fit)
            log_p = np.log(radial_power_fit)

            slope, intercept = np.polyfit(log_k, log_p, 1)

            plt.figure()
            plt.plot(log_k, log_p, label="")
            plt.plot(log_k, slope * log_k + intercept,
                    label=f"Slope = {slope:.2f}")
            plt.xlabel("log freq")
            plt.ylabel("log strength")
            plt.title("Beta volume 3D radial power spectrum")
            plt.legend()
            plt.show() 

            print("Estimated alpha:", slope)
        
        step_beta()
        # if i % (blue_steps_per_reaction_step * until_reaction(r)) == 0 and i > 10:
        #     step_beta()

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