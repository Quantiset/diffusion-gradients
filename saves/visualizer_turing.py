import numpy as np
import math
import matplotlib.pyplot as plt

f_min, f_max = 0.0001, 0.1
k_min, k_max = 0.0001, 0.1
resolution = 400

f_vals = np.linspace(f_min, f_max, resolution)
k_vals = np.linspace(k_min, k_max, resolution)
F, K = np.meshgrid(f_vals, k_vals, indexing='xy')

def get_strength(f, k):
    disc = f*f - 4*f*(f+k)*(f+k)
    if disc < 0:
        disc = 0.0
    b = (f + math.sqrt(disc)) / (2*(f+k))
    b2 = b*b

    if b2 > k:
        # return 1
        if b2 > f:
            return 2
        else:
            return 1
    else:
        if b2 > f:
            return 3
        else:
            return 0

    return 1 if b2 > k and b2 > f else 0

vec_get_strength = np.vectorize(get_strength)
strength = vec_get_strength(F, K)
def plot_and_save(f_vals, k_vals, strength, outpath='turing_strength_heatmap.png'):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        strength,
        origin='lower',
        extent=[f_vals[0], f_vals[-1], k_vals[0], k_vals[-1]],
        aspect='auto',
        cmap='viridis'
    )
    plt.xlabel('f')
    plt.ylabel('k')
    plt.title('Viability')
    cbar = plt.colorbar(im)
    cbar.set_label('viability')
    plt.tight_layout()
    # plt.savefig(outpath, dpi=300)
    print(f'Saved heatmap to {outpath}')
    plt.show()

if __name__ == '__main__':
    plot_and_save(f_vals, k_vals, strength)

