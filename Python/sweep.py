import torch
import cv2
import numpy as np
from blue_noise import gen_blue_noise
from anisotropy import find_stripiness

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

N = 256
dt = 0.005
diff_rate = 1.0

r_begin, r_end = 4.0, 14.0
s_begin, s_end = 4.0, 4.0
s = 4.0

def until_reaction(r):
    return int(7336.347 / (r - 3.6470) + 3739.7676)

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)

r_sweep = torch.linspace(r_begin, r_end, N, device=device)[:, None]

s_sweep = torch.linspace(s_begin, s_end, N, device=device)[None, :]

diff_sweep = torch.ones((1, N), device=device)

blue_noise = gen_blue_noise(N, N, N).to(device)
blue_step = 0
use_blue_noise = True

beta = torch.zeros((N, N), device=device)

def step_beta():
    global beta, blue_step

    if not use_blue_noise:
        beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)
        return

    blue_step = (blue_step + 1) % blue_noise.shape[2]
    beta = 12.0 + (0.1 * blue_noise[:, :, blue_step] - 0.05)

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

reaction_interval = until_reaction(r_begin)
blue_steps_per_reaction_step = 20
beta_update_interval = 50

for i in range(1_000_000):

    a += dt * (f(a, b, diff_sweep) + r_sweep * s * get_laplacian(a))
    b += dt * (g(a, b, diff_sweep, beta) + s * get_laplacian(b))

    a.clamp_(min=0)
    b.clamp_(min=0)

    if i % beta_update_interval == 0 and i > 10:
        step_beta()

    if i % 100 == 0:
        img_np = a.detach().cpu().numpy()
        print(img_np.min(), img_np.max())

        img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        cv2.imshow("RD Sweep", img)

        if cv2.waitKey(1) == ord('q'):
            break