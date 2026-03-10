import math
import torch
import cv2
import numpy as np

from anisotropy import find_stripiness

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

N = 512
dt = 0.1

diff_rate = 1.0

da, db = 0.2, 0.1
beta = 0.0 + (torch.rand((N, N), device=device) * 0.1)


def get_laplacian(Z):
    Ztop = Z.roll(1, 0)
    Zleft = Z.roll(1, 1)
    Zbottom = Z.roll(-1, 0)
    Zright = Z.roll(-1, 1)
    return Ztop + Zleft + Zbottom + Zright - 4 * Z

def f(a, b, f, k, diff_rate):
    return diff_rate * (-a * b * b + f * (1 - a))

def g(a, b, f, k, diff_rate, beta):
    return diff_rate * (a * b * b - (k + f) * b)

a = torch.full((N, N), 1.0, device=device)
b = torch.full((N, N), 0.0, device=device)
# set set small square in the middle to 1.0
b += beta

f_begin, f_end = 0.038, 0.038
k_begin, k_end = 0.062, 0.062
f_sweep = torch.linspace(f_begin, f_end, N, device=device)[:, None]
k_sweep = torch.linspace(k_begin, k_end, N, device=device)[None, :]

for i in range(1000000):
    a += dt * (f(a, b, f_sweep, k_sweep, diff_rate) + da * get_laplacian(a))
    b += dt * (g(a, b, f_sweep, k_sweep, diff_rate, beta) + db * get_laplacian(b)) + beta * math.sqrt(dt)
    a.clamp_(min=0, max=1)
    b.clamp_(min=0, max=1)

    if i % 100 == 0:
        img_np = b.cpu().numpy()
        img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
        print(img_np.min(), img_np.max())
        cv2.imshow("RD", img)
        
        if cv2.waitKey(1) == ord('q'): 
            break


