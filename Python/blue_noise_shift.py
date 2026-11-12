import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

BLUE_EXPONENT = 0.5 

N = 512  # one blue-noise line of length N drives the 1D, 2D and 3D constructions


def gen_blue_noise_1d(n, alpha=BLUE_EXPONENT):
    """Same construction as blue_noise.gen_blue_noise, in 1D: white noise whose
    spectrum is multiplied by |k|**alpha (k = distance to DC), then back.
    alpha=0.5 gives amplitude ~ k**0.5, i.e. power ~ k**1 (that .py defaults to
    alpha=1.0, which is power ~ k**2)."""
    noise = rng.random(n)
    spec = np.fft.fftshift(np.fft.fft(noise))
    kc = np.arange(-(n // 2), n - n // 2)          # centered frequency grid
    k = np.sqrt(kc.astype(float) ** 2)            # = |kc|
    spec = np.fft.ifftshift(spec * k ** alpha)
    out = np.fft.ifftn(spec).real
    out = (out - out.min()) / (out.max() - out.min())
    return out


def build_shifted_2d(v):
    """img[i, j] = v[(j - i) mod n]  --  row i is v rolled by i."""
    n = v.size
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return v[(j - i) % n]


def build_shifted_3d(v):
    """Slice i of the volume is the 2D image rolled by i pixels along x:
    the same shift-by-one trick that built y, now applied along z."""
    n = v.size
    img = build_shifted_2d(v).astype(np.float32)
    vol = np.empty((n, n, n), np.float32)
    for i in range(n):
        vol[i] = np.roll(img, i, axis=1)
    return vol


def power_spectrum_1d(v):
    """Power |F(k)|^2 vs |k|, DC dropped."""
    n = v.size
    p = np.abs(np.fft.fft(v)) ** 2
    k = np.abs(np.fft.fftfreq(n, d=1.0 / n))
    m = k > 0
    return k[m], p[m]


def power_spectrum_field(field):
    field = np.asarray(field, np.float32)
    field = field - np.float32(field.mean(dtype=np.float64))
    f = sfft.rfftn(field, workers=-1)
    p = (f.real ** 2 + f.imag ** 2)
    freqs = [np.fft.fftfreq(s, d=1.0 / s).astype(np.float32) for s in field.shape[:-1]]
    freqs.append(np.fft.rfftfreq(field.shape[-1], d=1.0 / field.shape[-1]).astype(np.float32))
    return p, freqs


def radial_reduce(p, freqs):
    d2 = np.zeros(p.shape, np.float32)
    for axis, fr in enumerate(freqs):
        shp = [1] * p.ndim
        shp[axis] = -1
        d2 = d2 + fr.reshape(shp) ** 2
    dist = np.sqrt(d2, out=d2)
    m = dist > 0
    return dist[m], p[m]


def radial_bin(k, p, num_bins=80):
    edges = np.logspace(np.log10(max(k.min(), 1e-9)), np.log10(k.max()), num_bins)
    psum, _ = np.histogram(k, bins=edges, weights=p)
    ksum, _ = np.histogram(k, bins=edges, weights=k)
    cnt, _ = np.histogram(k, bins=edges)
    m = cnt > 0
    return ksum[m] / cnt[m], psum[m] / cnt[m]


def trim_edges(k, p, frac=0.10):
    lo = int(np.floor(len(k) * frac))
    return k[lo:len(k) - lo], p[lo:len(p) - lo]


def loglog_fit(k, p):
    m = (k > 0) & (p > 0)
    slope, intercept = np.polyfit(np.log(k[m]), np.log(p[m]), 1)
    k_line = np.array([k[m].min(), k[m].max()])
    p_line = np.exp(intercept) * k_line ** slope
    return slope, k_line, p_line


def plot_spectrum(ax, k, p, title):
    kc, pc = radial_bin(k, p)
    keep = pc > pc.max() * 1e-8  # drop shells that caught only float-zero pixels
    kc, pc = kc[keep], pc[keep]
    kc, pc = trim_edges(kc, pc, 0.10)
    slope, k_line, p_line = loglog_fit(kc, pc)
    ax.loglog(kc, pc, lw=1.3, label="binned power")
    ax.loglog(k_line, p_line, "r-", lw=2.0, label=f"fit slope = {slope:.2f}")
    ax.set_title(title)
    ax.set_xlabel("distance to DC  |k|")
    ax.set_ylabel("power |F(k)|^2")
    ax.legend(fontsize=8)


def random_pillar(field):
    nd = field.ndim
    a = int(rng.integers(nd))
    idx = [int(rng.integers(field.shape[d])) for d in range(nd)]
    sl = list(idx)
    sl[a] = slice(None)
    return a, np.asarray(field[tuple(sl)], np.float64)


def main():
    v = gen_blue_noise_1d(N)

    img = build_shifted_2d(v)
    vol = build_shifted_3d(v)

    row_idx = int(rng.integers(N // 4, 3 * N // 4))
    cross = img[row_idx, :]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # --- 1D ---
    axes[0, 0].plot(v, lw=0.8)
    axes[0, 0].set_title(f"(a) 1D blue-noise sequence v (N={N})")
    axes[0, 0].set_xlabel("pixel")

    plot_spectrum(axes[0, 1], *power_spectrum_1d(v), "(b) power spectrum of v")
    plot_spectrum(axes[0, 2], *power_spectrum_1d(cross),
                  f"(c) power spectrum of a 1D cross section")

    # --- 2D ---
    axes[1, 0].imshow(img, cmap="gray", interpolation="nearest")
    axes[1, 0].set_title(f"(d) {N}x{N} image: row i = v rolled by i")

    p2, fr2 = power_spectrum_field(img)
    plot_spectrum(axes[1, 1], *radial_reduce(p2, fr2),
                  "(e) 2D radial power spectrum ")
    a2, pil2 = random_pillar(img)
    plot_spectrum(axes[1, 2], *power_spectrum_1d(pil2),
                  f"(f) power spectrum of a 1D cross section")

    # --- 3D ---
    mid = N // 2
    axes[2, 0].imshow(vol[mid], cmap="gray", interpolation="nearest")
    axes[2, 0].set_title(f"(g) {N}^3 volume: slice i = {mid}")

    a3, pil3 = random_pillar(vol)
    assert np.allclose(np.abs(np.fft.fft(pil3)), np.abs(np.fft.fft(v)), atol=1e-3), \
        "a volume pillar must have the same |FFT| as its 1D line"
    p3, fr3 = power_spectrum_field(vol)
    del vol
    plot_spectrum(axes[2, 1], *radial_reduce(p3, fr3),
                  "(h) 3D radial power spectrum")
    plot_spectrum(axes[2, 2], *power_spectrum_1d(pil3),
                  f"(i) power spectrum of a 1D cross section")

    fig.tight_layout()
    out_path = "blue_noise_shift_spectrum.png"
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
