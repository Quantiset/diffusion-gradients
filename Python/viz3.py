import math
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from blue_noise import gen_blue_noise, gen_blue_noise_slice, plot_3d

# device = torch.device("cuda")
device = torch.device("cpu")

noise = gen_blue_noise(256, 256, 256, 30).cpu().numpy()

ptr = (256,256)
horizontal = []
rez = 0
for i in range(255):
    horizontal.append(rez)
    rez += random.randint(-1,1)
    dist = 1
    ret = random.choice([(dist,0), (-dist,0), (0,dist), (0,-dist)])
    ptr = (ptr[0] + ret[0], ptr[1] + ret[1])
    ptr = (ptr[0] % noise.shape[0], ptr[1] % noise.shape[1])
    print(ptr)

N = len(horizontal)
freqs = np.fft.rfftfreq(N)
power = np.abs(np.fft.rfft(horizontal))**2

log_freqs = np.log(freqs[1:])
log_power = np.log(power[1:])
print(len(log_freqs), len(log_power))
slope = np.polyfit(log_freqs, log_power, 1)[0]
print(f"Slope of power spectrum: {slope:.2f}")

plt.plot(freqs[1:], power[1:])
plt.yscale("log")
plt.xscale("log")
plt.title("Power spectrum of blue noise")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.show()