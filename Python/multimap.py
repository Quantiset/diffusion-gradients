import torch
import cv2
import numpy as np

from anisotropy import find_stripiness

device = torch.device("cuda")

N = 256
dt = 0.01

diff_rate = 1.0

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


def sim(r, s, max_steps=10000):
    
    a = torch.full((N, N), 4.0, device=device)
    b = torch.full((N, N), 4.0, device=device)
    
    last_max = None

    for i in range(max_steps):
        a += dt * (f(a, b, diff_rate) + r * s * get_laplacian(a))
        b += dt * (g(a, b, diff_rate, beta) + s * get_laplacian(b))

        a.clamp_(min=0)
        b.clamp_(min=0)

        
        if i == max_steps - 3: 
            last_max = current_max

        if i % 100 == 0:
            img_np = a.cpu().numpy()
            current_max = img_np.max()
            
            img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 0.001)
            cv2.imshow("RD", img)
            
            if cv2.waitKey(1) == ord('q'): 
                return None

    img_np = a.cpu().numpy()
    current_max = img_np.max()
    stripiness = find_stripiness(a, device)

    return {
        'r': r,
        's': s,
        'min': img_np.min(),
        'max': current_max,
        'volatility': abs(current_max - last_max) if last_max is not None else 0.0,
        'stripiness': stripiness
    }

r_values = np.linspace(1, 20, 15)
s_values = np.linspace(1, 20, 15)

with open("data.txt", "w") as file:
    file.write("r s min max volatility stripiness\n")

for r in r_values:
    for s in s_values:
        print(f"\n{'='*50}")
        print(f" r={r:.3f}, s={s:.3f}")
        print(f"{'='*50}")
        
        result = sim(r, s, max_steps=5000)
        
        if result is None: 
            cv2.destroyAllWindows()
            exit()
        
        with open("data.txt", "a") as file:
            line = f"{result['r']:.6f} {result['s']:.6f} {result['min']:.6f} {result['max']:.6f} {result['volatility']:.6f} {result['stripiness']:.6f}\n"
            file.write(line)
            print(f"Logged: {line.strip()}")

cv2.destroyAllWindows()