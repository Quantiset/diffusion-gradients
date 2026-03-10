import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda")

def gen_blue_noise(x, y, z, alpha=0.5):
    noise = torch.randn((x, y, z), device=device)
    fft = torch.fft.fftn(noise)
    fft = torch.fft.fftshift(fft)
    xcoord, ycoord, zcoord = np.meshgrid(np.arange(-x/2, x/2), np.arange(-y/2, y/2), np.arange(-z/2, z/2), indexing='ij')
    k = torch.sqrt(torch.from_numpy(xcoord**2 + ycoord**2 + zcoord**2).to(device))
    fft = torch.fft.ifftshift(fft * k ** alpha)
    op = torch.fft.ifftn(fft).real
    op = (op - op.min()) / (op.max() - op.min())
    return op

def show_noise():
    noise = gen_blue_noise(512, 512, 1, -2.5)
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

def take_line_and_show_fft():
    noise = gen_blue_noise(512, 512, 1, 0.5)[:, :, 0]
    noise_np = noise.cpu().numpy()
    fft2 = np.fft.fftshift(np.fft.fft2(noise_np))
    power = np.abs(fft2)**2
    y, x = np.indices(power.shape)
    center = np.array(power.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    power[center[0], center[1]] = 0
    power = power / np.max(power)
    cv2.imshow("Blue Noise", power / np.max(power))
    plt.loglog(radialprofile[1:])
    plt.title("Radially Averaged Power Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.show()

if __name__ == "__main__":
    take_line_and_show_fft()