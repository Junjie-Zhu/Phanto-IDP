import biotite.structure as struc
import biotite.structure.io as strucio

subject = '../StrucRef/predicted.150.47.pdb'
reference = '../StrucRef/predicted.150.47.pdb'

subject_array = strucio.load_structure(subject)
reference_array = strucio.load_structure(reference)

print(struc.rmsd(reference_array, subject_array))
