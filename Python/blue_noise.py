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


def plot_3d(img):
    beta_cpu = img.detach().cpu()

    if beta_cpu.ndim == 2:
        beta_cpu = beta_cpu.unsqueeze(0)

    if beta_cpu.ndim != 3:
        return

    T = beta_cpu.shape[0]

    field = beta_cpu - beta_cpu.mean()

    fft3 = torch.fft.fftn(field)
    power3 = torch.abs(fft3) ** 2

    ft = torch.fft.fftfreq(T)
    fx = torch.fft.fftfreq(field.shape[1])
    fy = torch.fft.fftfreq(field.shape[2])

    kt, kx, ky = torch.meshgrid(ft, fx, fy, indexing='ij')

    freqs = torch.sqrt(kt**2 + kx**2 + ky**2)

    freqs_flat = freqs.flatten()
    p_flat = power3.flatten()

    mask = freqs_flat > 0
    k_np = freqs_flat[mask].detach().cpu().numpy()
    p_np = p_flat[mask].detach().cpu().numpy()

    num_bins = 100
    bins = np.linspace(k_np.min(), k_np.max(), num_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    radial_power = np.zeros(num_bins - 1)
    for j in range(len(bins) - 1):
        mask = (k_np >= bins[j]) & (k_np < bins[j+1])
        if np.any(mask):
            radial_power[j] = p_np[mask].mean()
        else:
            radial_power[j] = np.nan
    k_fit = bin_centers[(bin_centers > np.percentile(bin_centers, 15)) & (bin_centers < np.percentile(bin_centers, 90))]
    radial_power_fit = radial_power[(bin_centers > np.percentile(bin_centers, 15)) & (bin_centers < np.percentile(bin_centers, 90))]

    log_k = np.log(k_fit)
    log_p = np.log(radial_power_fit)

    slope, intercept = np.polyfit(log_k, log_p, 1)

    plt.figure()
    plt.plot(log_k, log_p, label="")
    plt.plot(log_k, slope * log_k + intercept,
            label=f"Slope = {slope:.2f}")
    plt.xlabel("log freq")
    plt.ylabel("log strength")
    plt.title("Beta volume 3D radial power spectrum")
    plt.legend()
    plt.show() 

    print("Estimated alpha:", slope)


if __name__ == "__main__":
    show_noise()
