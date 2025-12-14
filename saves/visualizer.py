import matplotlib.pyplot as plt

filename = "2025-11-11T21-25-31.txt"

x_vals = []
y_vals = []
z_vals = []

with open(filename, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4: 
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[6])
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

plt.figure(figsize=(6,5))
sc = plt.scatter(x_vals, y_vals, c=z_vals, cmap="viridis")
plt.colorbar(sc, label="Determinant")
plt.xlabel("f")
plt.ylabel("k")
plt.title("Determinant by (f, k) Position")
plt.savefig("determinant.png")
plt.show()