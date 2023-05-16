import numpy as np
import matplotlib.pyplot as plt

rmsd = np.loadtxt('rmsd-syn.dat')
print(np.max(rmsd))
print(np.argmax(rmsd))
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(rmsd,
        # weights=[1 / len(rmsd) for _ in range(len(rmsd))],
        bins=100,
        rwidth=0.9,
        color='#00c6c6')
# ax.set_xlim(0, 2.5)
ax.set_xlabel("Reconstruction RMSD", fontsize=16, fontname='Palatino Linotype')
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.show()
