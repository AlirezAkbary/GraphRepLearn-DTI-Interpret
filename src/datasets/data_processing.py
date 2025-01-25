import numpy as np
from rdkit import Chem
from typing import Any, List, Tuple, Dict

def smile_to_graph(smile: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Converts a SMILES string to graph representation.

    Args:
        smile (str): SMILES string.

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: Number of atoms, atom features, and edge indices.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")

    c_size = mol.GetNumAtoms()

    # Get atom features
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    features = np.array(features, dtype=np.float32) # shape: num atoms (varies among molecules) * num_features (fix among molecules)

    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([start, end])
        edge_indices.append([end, start])  # Undirected graph

    edge_index = np.array(edge_indices, dtype=np.int64).T # 2 (start and end node) * number of edges in the molecule
    return c_size, features, edge_index


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    Generates atom features including:
    - Atom symbol (one-hot encoding)
    - Degree (one-hot encoding)
    - Total number of Hs (one-hot encoding)
    - Implicit valence (one-hot encoding)
    - Aromaticity (boolean)

    Args:
        atom (Chem.Atom): RDKit atom object.

    Returns:
        np.ndarray: Atom feature vector.
    """
    atom_symbols = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
        'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
        'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
        'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
        'Pb', 'Unknown'
    ]

    symbol = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols)
    degree = one_of_k_encoding_unk(atom.GetDegree(), list(range(11)))
    total_num_hs = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
    implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
    aromaticity = [int(atom.GetIsAromatic())]

    features = symbol + degree + total_num_hs + implicit_valence + aromaticity
    return np.array(features, dtype=np.float32)


def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[int]:
    """
    Creates a one-hot encoding of x in the allowable set.

    Args:
        x (Any): The item to encode.
        allowable_set (List[Any]): List of allowable items.

    Returns:
        List[int]: One-hot encoding of x.
    """
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return [int(x == s) for s in allowable_set]


def one_of_k_encoding_unk(x: Any, allowable_set: List[Any]) -> List[int]:
    """
    Creates a one-hot encoding of x in the allowable set, mapping unknown values to the last element.

    Args:
        x (Any): The item to encode.
        allowable_set (List[Any]): List of allowable items.

    Returns:
        List[int]: One-hot encoding of x.
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]



def encode_protein_sequence(prot: str, max_seq_len: int = 1000) -> np.ndarray:
    """
    Encodes a protein sequence as a fixed-length vector.

    Args:
        prot (str): Protein sequence.
        max_seq_len (int): Maximum sequence length.

    Returns:
        np.ndarray: Encoded protein sequence.
    """
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    x = np.zeros(max_seq_len, dtype=np.int64)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)  # Unknown characters mapped to 0
    return x


