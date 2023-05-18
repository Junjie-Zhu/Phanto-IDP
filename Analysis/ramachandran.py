import os

import biotite.structure as struc
import biotite.structure.io as strucio
import matplotlib.pyplot as plt
import numpy as np

inpath = ''

# Parse file
phi, psi, omega = np.array([]), np.array([]), np.array([])
for file in os.listdir(inpath):
    atom_array = strucio.load_structure(file)
    # Calculate backbone dihedral angles
    # from one of the two identical chains in the asymmetric unit
    ph, ps, omg = struc.dihedral_backbone(atom_array)

    # Remove invalid values (NaN) at first and last position
    ph = ph[1:-1]
    ps = ps[1:-1]

    phi = np.append(phi, ph)
    psi = np.append(psi, ps)
    omega = np.append(omega, omg)

    # A threshold that only 2W phi-psis are accepted
    if len(phi) >= 20000:
        break

# Conversion from radians into degree
phi *= 180 / np.pi
psi *= 180 / np.pi

# Save results
np.savetxt('phi.dat', phi)
np.savetxt('psi.dat', psi)
np.savetxt('omega.dat', omega)

# Plot density
figure = plt.figure()
ax = figure.add_subplot(111)
h, xed, yed, image = ax.hist2d(phi, psi, bins=(200, 200),
                               cmap="RdYlGn_r", cmin=1)
cbar = figure.colorbar(image, orientation="vertical")
cbar.set_label("Count")
ax.set_aspect("equal")
ax.set_xlim(-180, 175)
ax.set_ylim(-180, 175)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\psi$")
figure.tight_layout()
plt.show()
