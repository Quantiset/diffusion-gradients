import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda")

def gen_blue_noise(x, y, z, alpha=1.0):
    noise = torch.rand((x, y, z), device=device)
    fft = torch.fft.fftn(noise)
    fft = torch.fft.fftshift(fft)
    xc, yc, zc = np.meshgrid(np.arange(-x/2, x/2), np.arange(-y/2, y/2), np.arange(-z/2, z/2), indexing='ij')
    k = torch.sqrt(torch.from_numpy(xc**2 + yc**2 + zc**2).to(device))
    fft = torch.fft.ifftshift(fft * k ** alpha)
    op = torch.fft.ifftn(fft).real
    op = (op - op.min()) / (op.max() - op.min())
    return op

def gen_blue_noise_slice(x, y, z, alpha=1.0):
    noise = torch.rand((x, y, z), device=device)
    fft = torch.fft.fftn(noise)
    fft = torch.fft.fftshift(fft)
    xc, yc, zc = np.meshgrid(np.arange(-x/2, x/2), np.arange(-y/2, y/2), np.arange(-z/2, z/2), indexing='ij')
    kx = torch.from_numpy(np.abs(xc)).to(device)
    ky = torch.from_numpy(np.abs(yc)).to(device)
    kz = torch.from_numpy(np.abs(zc)).to(device)
    k = (kx ** alpha) * (ky ** alpha) * (kz ** alpha)
    k = k / k.mean()
    fft = torch.fft.ifftshift(fft * k)
    op = torch.fft.ifftn(fft).real
    op = (op - op.min()) / (op.max() - op.min())
    return op

def show_noise():
    noise = gen_blue_noise(256, 256, 1, 12.5)
    fft_mag = torch.abs(noise)

    fft_np = fft_mag.cpu().numpy()

    fft_img = np.zeros_like(fft_np, dtype=np.uint8)
    for i in range(fft_np.shape[2]):
        channel = fft_np[:, :, i]
        channel = np.log1p(channel)
        channel = 255 * (channel / np.max(channel))
        fft_img[:, :, i] = channel.astype(np.uint8)

    cv2.imshow("FFT Magnitude", fft_img)
    cv2.waitKey(0)




if __name__ == "__main__":
    show_noise()
