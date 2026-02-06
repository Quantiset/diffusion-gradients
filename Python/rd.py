import torch
import cv2
import numpy as np

from anisotropy import find_stripiness

# torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

N = 1024
dt = 0.005

diff_rate = 1.0

a = torch.full((N, N), 4.0, device=device)
b = torch.full((N, N), 4.0, device=device)
beta = 12.0 + (torch.rand((N, N), device=device) * 0.1 - 0.05)


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
    
    a = torch.full((N, N), 4.0, device=device)
    b = torch.full((N, N), 4.0, device=device)

    for i in range(10000000):
        a += dt * (f(a, b, diff_rate) + r * s * get_laplacian(a))
        b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

        a.clamp_(min=0)
        b.clamp_(min=0)

        if i % 100 == 0:
            img_np = b.cpu().numpy()
            print(img_np.min(), img_np.max())
            img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
            cv2.imshow("RD", img)
            
            if cv2.waitKey(1) == ord('q'): 
                break

        # if i == 999:
        #     with open("data.txt", "a") as file:
        #         line = f"{r} {s} {img_np.min()} {img_np.max()} {find_stripiness(a, device)}\n"
        #         print(line)
        #         file.write(line)

sim(r=1, s=20)

