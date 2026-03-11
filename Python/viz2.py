import numpy as np
from matplotlib import pyplot as plt


raw = []
with open("points_values.txt", "r") as f:
    for line in f:
        line = line.strip().strip("[]")
        if not line:
            continue
        vals = line.split()
        if "'---'" in vals:
            continue
        raw.append([float(v.removeprefix("'").removesuffix("'")) for v in vals])

raw = np.array(raw)

NUM_POINTS = 10

series: dict[tuple, list] = {}
for row in raw:
    x, y, v = int(row[0]), int(row[1]), row[2]
    key = (x, y)
    if key not in series:
        series[key] = []
    series[key].append(v)

print(f"Loaded {len(raw)} records for {len(series)} unique point(s).")
for k, v in series.items():
    print(f"  Point {k}: {len(v)} timesteps")
raw = raw[100:]

POINT_INDEX = 0

keys = list(series.keys())
chosen_key = keys[POINT_INDEX]
signal = np.array(series[chosen_key], dtype=np.float64)
N = len(signal)

print(f"\nRunning FFT on point {chosen_key}  ({N} samples, sampled every 50 sim steps)")

SIM_STEPS_PER_SAMPLE = 50

all_power = []
signals = []

for key, vals in series.items():
    signal = np.array(vals, dtype=np.float64)

    signal_zero = signal - signal.mean()

    fft_vals = np.fft.rfft(signal_zero)
    power = np.abs(fft_vals)**2

    all_power.append(power)
    signals.append(signal)

all_power = np.array(all_power)

N = len(signals[0])
freqs = np.fft.rfftfreq(N)
freqs_sim = freqs / SIM_STEPS_PER_SAMPLE

mean_power = all_power.mean(axis=0)


fig, axes = plt.subplots(3,1,figsize=(10,9))
fig.suptitle("Average spectrum across tracked points")

for s in signals:
    timesteps = np.arange(len(s)) * SIM_STEPS_PER_SAMPLE
    axes[0].plot(timesteps, s, alpha=0.5)

axes[0].set_xlabel("Sim iteration")
axes[0].set_ylabel("a value")
axes[0].set_title("Signals from tracked points")

axes[1].plot(freqs_sim[1:], mean_power[1:])
axes[1].set_xlabel("Frequency")
axes[1].set_ylabel("Power")
axes[1].set_title("Average power spectrum")

valid = (freqs_sim > 0) & (mean_power > 0)
log_f = np.log(freqs_sim[valid])
log_p = np.log(mean_power[valid])

axes[2].plot(log_f, log_p)

slope, intercept = np.polyfit(log_f, log_p, 1)
axes[2].plot(log_f, slope*log_f + intercept, "--", label=f"Slope = {slope:.2f}")
axes[2].legend()

axes[2].set_xlabel("log frequency")
axes[2].set_ylabel("log power")
axes[2].set_title("Log-log spectrum (blueness)")

print(f"Average spectral slope (blueness): {slope:.4f}")

plt.tight_layout()
plt.show()