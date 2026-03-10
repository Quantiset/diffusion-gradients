import torch
import numpy as np
import cv2

device = torch.device("cuda")


# Test 1D case first
noise = torch.randn((256, 256, 256), device=device)
fft = torch.fft.fft2(noise[:, :, 0])
freqsx = torch.fft.fftfreq(256, device=device)
freqsy = torch.fft.fftfreq(256, device=device)
freqs = torch.sqrt(freqsx ** 2 + freqsy ** 2)

# Apply k^alpha filter
alpha = 0.5
freqs[0] = 1
fft_filtered = fft * freqs ** alpha

power = torch.abs(fft_filtered) ** 2
log_k = torch.log(freqs[1:]).cpu().numpy()
log_p = torch.log(power[1:]).cpu().numpy()
slope, intercept = np.polyfit(log_k, log_p, 1)
print("Expected:", 2*alpha, "Got:", slope)