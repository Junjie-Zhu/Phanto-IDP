import biotite
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import biotite.structure.annotate_sse as sse
import biotite.structure.io as strucio
import biotite.structure as struc
import numpy as np
import torch
from typing import Iterable

# A lot of functions are here:
# 1. rmsd and motif rmsd
# 2. extract atom_array, sequence, plddt



restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
alphabet = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
}
alphabet_map = {v:k for k,v in alphabet.items()}

def extract_seq(protein,chain_id=None):
    if isinstance(protein,str):
        atom_array = strucio.load_structure(protein,model=1)
    elif isinstance(protein, struc.AtomArrayStack):
        atom_array = protein[0]
    elif isinstance(protein, struc.AtomArray):
        atom_array = protein

    # add multiple chain sequence subtract function
    all_chains = get_chains(atom_array)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain_id is None:
        chain_ids = all_chains
    elif isinstance(chain_id, list):
        chain_ids = chain_id
    else:
        chain_ids = [chain_id] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in atom_array]
    atom_array = atom_array[chain_filter]

    # mask canonical aa
    aa_mask = struc.filter_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]
    residue_identities = get_residues(atom_array)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return seq    

def extract_dihedral(protein):
    if isinstance(protein, str):
        atom_array = strucio.load_structure(protein, model=1)
    phi, psi, omega = struc.dihedral_backbone(
        atom_array[atom_array.chain_id == "A"]
    )
    # Conversion from radians into degree
    phi *= 180/np.pi
    psi *= 180/np.pi
    omega *= 180/np.pi
    # Remove invalid values (NaN) at first and last position
    phi = phi[1:-1]
    psi = psi[1:-1]
    omega = omega[1:-1]
    return phi, psi, omega

def extract_plddt(protein,chain_id=None):
    """
    extract the plddt from a target protein file or atom array
    chain_id is the specific chain number
    """
    if isinstance(protein,str):
        # model = 1 to load a AtomArray object
        # extra_fields to load the b_factor column
        atom_array = strucio.load_structure(protein,model=1,extra_fields=["b_factor"])
    elif isinstance(protein, struc.AtomArrayStack):
        atom_array = protein[0]
    elif isinstance(protein, struc.AtomArray):
        atom_array = protein

    # add multiple chain sequence subtract function
    all_chains = get_chains(atom_array)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain_id is None:
        chain_ids = all_chains
    elif isinstance(chain_id, list):
        chain_ids = chain_id
    else:
        chain_ids = [chain_id] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in atom_array]
    atom_array = atom_array[chain_filter]

    # mask canonical aa 
    aa_mask = struc.filter_canonical_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]

    # ca atom only
    atom_array = atom_array[atom_array.atom_name == "CA"]

    plddt = np.array([i.b_factor for i in atom_array])

    return plddt, np.mean(plddt)

def secondary_structure(protein):
    if isinstance(protein,"str"):
        protein = strucio.load_structure(protein,model=1)
    return sse(protein)

def rmsd(reference,target):
    reference = strucio.load_structure(reference,model=1)
    target = strucio.load_structure(target,model=1)
    mask_reference = (((reference.atom_name == "N") | (reference.atom_name == "CA") | (reference.atom_name == "C") | (reference.atom_name == "O")) & (biotite.structure.filter_amino_acids(reference)))
    # mask_reference = reference.atom_name == "CA"
    reference = reference[mask_reference]
    mask_target = (((target.atom_name == "N") | (target.atom_name == "CA") | (target.atom_name == "C") | (target.atom_name == "O")) & (biotite.structure.filter_amino_acids(target)) )
    # mask_target = target.atom_name == "CA"
    target = target[mask_target]
    superimposed, _ = struc.superimpose(reference, target)
    rms = struc.rmsd(reference, superimposed)
    return rms

def recur_print(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return f"{x.shape}_{x.dtype}"
    elif isinstance(x, dict):
        return {k: recur_print(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return [recur_print(v) for v in x]
    else:
        raise RuntimeError(x)
