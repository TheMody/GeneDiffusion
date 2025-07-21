import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# 1. Data (taken directly from your LaTeX table)
# ------------------------------------------------------------------
percentages = [5, 10, 20, 50]                     # x‑axis “amount real data”
als_no  = [70.96, 76.50, 80.90, 84.60]            # ALS:   no synthetic data
als_yes = [84.83, 85.01, 85.34, 85.70]            # ALS:   with synthetic data
kg_no   = [29.01, 43.99, 71.52, 85.19]            # 1KG:   no synthetic data
kg_yes  = [83.98, 86.93, 87.11, 87.50]            # 1KG:   with synthetic data

# ------------------------------------------------------------------
# 2. Plot parameters
# ------------------------------------------------------------------
x      = np.arange(len(percentages))   # bar positions
width  = 0.35                          # bar width
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4), dpi=120
)

datasets = [
    ("ALS Data",  als_no,  als_yes),
    ("1KG Data",  kg_no,   kg_yes),
]

# ------------------------------------------------------------------
# 3. Build the two panels
# ------------------------------------------------------------------
for ax, (title, no_syn, with_syn) in zip(axes, datasets):
    ax.bar(x - width/2, no_syn,  width,
           label="No synthetic",   edgecolor="black")
    ax.bar(x + width/2, with_syn, width,
           label="With synthetic", edgecolor="black", hatch="//")

    ax.set_title(title,  fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} %" for p in percentages])
    ax.set_xlabel("Amount of real data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(frameon=False, fontsize="small")

axes[0].set_ylim(60, 90)
axes[1].set_ylim(20, 90)
# ------------------------------------------------------------------
# 4. Overall figure tweaks
# ------------------------------------------------------------------
#fig.suptitle("Effect of Synthetic Data on Model Accuracy", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("figures/synimpro.png", dpi=300, bbox_inches='tight')
plt.show()
