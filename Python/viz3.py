import math
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from blue_noise import gen_blue_noise, gen_blue_noise_slice

# device = torch.device("cuda")
device = torch.device("cpu")

noise = gen_blue_noise(512, 512, 1).cpu().numpy()

ptr = (256,256)
horizontal = []
for i in range(255):
    horizontal.append(math.sqrt((ptr[0] - 256)**2 + (ptr[1] - 256)**2))
    dist = 1
    ret = random.choice([(dist,0), (-dist,0), (0,dist), (0,-dist)])
    ptr = (ptr[0] + ret[0], ptr[1] + ret[1])
    ptr = (ptr[0] % noise.shape[0], ptr[1] % noise.shape[1])
    print(ptr)
# horizontal = noise[0, :, 0]

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