
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
# 1. Data  (rows follow the order shown in the LaTeX table)
# ---------------------------------------------------------------
rows      = ["ALS‑MLP", "ALS‑Transformer", "ALS‑CNN",
             "1KG‑MLP", "1KG‑Transformer", "1KG‑CNN",
             "Average"]
cols      = ["CNN", "MLP", "MLP+CNN", "Transformer"]

data = np.array([
    [71.51, 96.69, 94.26, 73.77],   # ALS‑MLP
    [66.06, 93.44, 93.89, 69.30],   # ALS‑Transformer
    [69.88, 91.72, 93.17, 68.72],   # ALS‑CNN
    [15.58, 65.80, 93.02, 13.28],   # 1KG‑MLP
    [16.23, 62.99, 84.98,  8.38],   # 1KG‑Transformer
    [19.52, 56.57, 77.54, 21.21],   # 1KG‑CNN
    [43.17, 78.06, 89.48, 42.56],   # Average
])

# ---------------------------------------------------------------
# 2. Build the heat‑map (single Axes)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.set_anchor('W') 
im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100)

# ‑‑ Axes styling
ax.set_xticks(np.arange(len(cols)), labels=cols, fontsize=9, rotation=45)
ax.set_yticks(np.arange(len(rows)), labels=rows, fontsize=9)
ax.set_xlabel("Synthetic data generator", fontweight="bold", labelpad=8)
ax.set_ylabel("Classifier", fontweight="bold", labelpad=8)
#ax.set_tite("Recovery Rates (%) on Hold‑out Genotypes\nTrained Classifiers vs. Synthetic Data Type",
   #          fontsize=12, fontweight="bold", pad=12)

# ---------------------------------------------------------------
# 3. Annotate each cell (bold the row‑best value)
# ---------------------------------------------------------------
for i in range(data.shape[0]):
    row_max_idx = data[i].argmax()
    for j in range(data.shape[1]):
        kw = dict(ha="center", va="center")
        if j == row_max_idx:
            kw["fontweight"] = "bold"
        ax.text(j, i, f"{data[i, j]:.2f}", **kw, fontsize=8)

# ---------------------------------------------------------------
# 4. Extra polish
# ---------------------------------------------------------------
# Optional cell borders for print clarity
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="w", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Color‑bar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Recovery rate (%)", rotation=270, labelpad=15)

fig.tight_layout()
plt.show()

plt.savefig("figures/recoveryrate.png", dpi=300, bbox_inches='tight')