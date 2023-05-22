import biotite.structure as struc
import biotite.structure.io as strucio

subject = '../StrucRef/predicted.5.35.pdb'
reference = '../StrucRef/target.5.35.pdb'

subject_array = strucio.load_structure(subject)
reference_array = strucio.load_structure(reference)
superimposed, _ = struc.superimpose(reference_array, subject_array)

print(struc.rmsd(reference_array, superimposed))
