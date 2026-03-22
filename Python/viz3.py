import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda")

Ns = [2.25, 4.5, 7.5, 10]
Gflops = [2.9, 3.5, 3.51, 5]

plt.plot(Ns, Gflops, marker='o', label='GFLOPS')
plt.legend()
plt.title("GFLOPS vs N")
plt.xlabel("N (thousands)")
plt.ylabel("GFLOPS")
plt.show()