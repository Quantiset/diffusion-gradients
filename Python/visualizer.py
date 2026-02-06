import matplotlib.pyplot as plt
import math

filename = "data.txt"

x_vals = []
y_vals = []
z_vals = []

with open(filename, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4: 
            x = float(parts[0])
            y = float(parts[1])
            z = parts[4]
            if z == 'nan':
                continue
            else:
                z = float(z)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

plt.figure(figsize=(6,2))
sc = plt.scatter(x_vals, y_vals, c=z_vals, cmap="viridis")
plt.colorbar(sc, label="Max")
plt.xlabel("r")
plt.ylabel("s")
plt.title("Circular-ness by (r, s) Position")
# plt.savefig("ani_rs.png")
plt.show()