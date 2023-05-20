import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rmsd = np.loadtxt('./output/rmsd-syn.dat')
rmsd_paa = np.loadtxt('./output/rmsd-paa.dat')
rmsd_rs1 = np.loadtxt('./output/rmsd-rs1.dat')

# Choose a colormap
# cmap = 'Spectral'
cmap = 'gnuplot'
# cmap = 'viridis'

# Generate a list of discrete colors
n_colors = 3
colors = sns.color_palette(cmap, n_colors=n_colors)

fig, ax = plt.subplots(figsize=(5, 7))

# Create the KDE plot
sns.kdeplot(rmsd_rs1, shade=True, color=colors[0], label='RS1')
sns.kdeplot(rmsd_paa, shade=True, color=colors[1], label='PaaA2')
sns.kdeplot(rmsd, shade=True, color=colors[2], label=r'$\alpha$-synuclein')

ax.axvline(0.511, color=colors[0], linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(0.885, color=colors[1], linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(2.714, color=colors[2], linestyle='--', linewidth=1, alpha=0.7)

ax.set_xlabel("Backbone Reconstruction RMSD", fontsize=24, fontname='Palatino Linotype')
ax.set_xticks(np.arange(0, 5, 0.4), fontsize=16)

ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

legend_font = {
    'family': 'Palatino Linotype',
    'size': 16,
}
# plt.legend(prop=legend_font)
plt.tight_layout()
plt.savefig('C:/MyFiles/SJTU/Junior/New-Topic/Tests and Article/RawFigs/syn.png', dpi=300)
plt.show()
