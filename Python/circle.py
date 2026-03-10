import torch
import cv2
import numpy as np
from blue_noise import gen_blue_noise
from anisotropy import find_stripiness

device = torch.device("cuda")

def until_reaction(r): 
    return int(7336.347 / (r - 3.6470) + 3739.7676)

N = 256
dt = 0.002
diff_rate = 1.0
blue_steps_per_reaction_step = 20 / until_reaction(4)

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)

blue_noise_3d = gen_blue_noise(N, N, N).to(device)
beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)

use_blue_noise = True
blue_slice_index = 0 

radius = 10
theta = 0
dtheta = 0.05

def step_beta():
    global beta, theta

    if not use_blue_noise:
        beta[:] = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)
        return

    offset_x = int(radius * np.cos(theta))
    offset_y = int(radius * np.sin(theta))
    theta += dtheta

    slice_2d = blue_noise_3d[:, :, blue_slice_index]
    beta[:] = 12.0 + (0.1 * torch.roll(slice_2d, shifts=(offset_x, offset_y), dims=(0,1)) - 0.05)

step_beta()

def get_laplacian(Z):
    Ztop = Z.roll(1, 0); Zleft = Z.roll(1, 1); Zbottom = Z.roll(-1, 0); Zright = Z.roll(-1, 1)
    Ztop_left = Z.roll(1, 0).roll(1, 1); Ztop_right = Z.roll(1, 0).roll(-1, 1)
    Zbottom_left = Z.roll(-1, 0).roll(1, 1); Zbottom_right = Z.roll(-1, 0).roll(-1, 1)
    return Ztop + Zleft + Zbottom + Zright + 4 * (Ztop_left + Ztop_right + Zbottom_left + Zbottom_right) - 20 * Z

def f(a, b, diff_rate):
    return diff_rate * (16.0 - a * b)

def g(a, b, diff_rate, beta):
    return diff_rate * (a * b - b - beta)

def sim(r, s):
    a = torch.full((N, N), 4.0, device=device)
    b = torch.full((N, N), 4.0, device=device)

    for i in range(10000000):
        a += dt * (f(a, b, diff_rate) + r * s * get_laplacian(a))
        b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

        a.clamp_(min=0)
        b.clamp_(min=0)

        if i % 100 == 0:
            img_np = a.cpu().numpy()
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

        if i % (blue_steps_per_reaction_step * until_reaction(r)) == 0 and i > 10:
            step_beta()

sim(r=4, s=4)
