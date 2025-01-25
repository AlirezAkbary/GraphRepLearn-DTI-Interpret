import argparse
import os
import zipfile
import requests
import io
import numpy as np
import json
import pickle
from collections import OrderedDict
from typing import List, Dict, Tuple

from rdkit import Chem
from sklearn.utils import resample

from src.datasets.drug_target_dataset import DrugTargetAffinityDataset
from src.datasets.data_processing import smile_to_graph, encode_protein_sequence


def download_and_extract_data():
    """
    Downloads the DeepDTA data from the GitHub repository and extracts the 'data' directory.
    """
    # GitHub repository ZIP URL
    zip_url = 'https://github.com/hkmztrk/DeepDTA/archive/refs/heads/master.zip'

    print("Downloading data from GitHub repository...")
    try:
        response = requests.get(zip_url)
        response.raise_for_status()  # Check if the request was successful
    except requests.RequestException as e:
        print(f"Error downloading data: {e}")
        return False

    print("Extracting data...")
    try:
        # Use BytesIO to read the ZIP file from the response content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract only the 'data' directory
            for member in zip_ref.namelist():
                if member.startswith('DeepDTA-master/data/'):
                    zip_ref.extract(member, '.')
        # Move the extracted 'data' directory to the current directory
        if not os.path.exists('data'):
            os.rename('DeepDTA-master/data', 'data')
        # Clean up the extracted repository directory
        if os.path.exists('DeepDTA-master'):
            import shutil
            shutil.rmtree('DeepDTA-master')
        print("Data downloaded and extracted successfully.")
        return True
    except zipfile.BadZipFile as e:
        print(f"Error extracting data: {e}")
        return False


def load_deepdta_dataset(dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str], np.ndarray, List[int], List[int], List[int]]:
    """
    Loads the DeepDTA dataset for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'davis').

    Returns:
        Tuple containing ligands, proteins, affinity matrix, train indices, validation indices, and test indices.
    """
    fpath = os.path.join('data', dataset_name)
    # Load folds
    with open(os.path.join(fpath, 'folds', 'train_fold_setting1.txt')) as f:
        whole_train_fold = json.load(f)
    
    train_folds = whole_train_fold[:-1] # len: num of folds
    validation_fold = whole_train_fold[-1] # len: num of samples
    test_fold = json.load(open(os.path.join(fpath, 'folds', 'test_fold_setting1.txt'))) # len: num of samples

    # Flatten train folds
    train_fold = [ee for fold in train_folds for ee in fold] # len: num of samples

    # Load ligands and proteins
    ligands = json.load(open(os.path.join(fpath, 'ligands_can.txt')), object_pairs_hook=OrderedDict) # davis len: 68
    proteins = json.load(open(os.path.join(fpath, 'proteins.txt')), object_pairs_hook=OrderedDict) # davis len: 442

    # Load affinity matrix
    affinity = pickle.load(open(os.path.join(fpath, 'Y'), 'rb'), encoding='latin1')
    if dataset_name == 'davis':
        affinity = -np.log10(affinity / 1e9)
    affinity = np.asarray(affinity) # davis shape: 68 * 442
    return ligands, proteins, affinity, train_fold, validation_fold, test_fold


def generate_pytorch_data(datasets: List[str]):
    """
    Generates PyTorch datasets from the DeepDTA datasets.

    Args:
        datasets (List[str]): List of dataset names to process.
    """
    # building a set of unique canonical isomeric SMILES strings from the ligands
    compound_iso_smiles = set()
    for dataset_name in datasets:
        ligands, proteins, _, _, _, _ = load_deepdta_dataset(dataset_name)
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            # print(d, ligands[d], lg, Chem.MolToSmiles(Chem.MolFromSmiles(lg), isomericSmiles=True))
            compound_iso_smiles.add(lg)
    
    # Build SMILES:graph dictionary. value is a tuple (n_atoms, features of atoms, edges)
    smile_graph = {}
    for smile in compound_iso_smiles:
        smile_graph[smile] = smile_to_graph(smile)
        # print(smile, smile_graph[smile][0], smile_graph[smile][1].shape, smile_graph[smile][2].shape)

    
    # Now process each dataset
    for dataset_name in datasets:
        ligands, proteins, affinity, train_fold, validation_fold, test_fold = load_deepdta_dataset(dataset_name)

        # Process ligands and proteins
        drugs = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            drugs.append(lg)
        prots = list(proteins.values())

        # Get indices where affinity is not NaN
        rows, cols = np.where(~np.isnan(affinity))
        

        # Define splits
        splits = {
            'train': train_fold,
            'validation': validation_fold,
            'test': test_fold
        }

        for split_name, indices in splits.items():
            split_indices = indices
            split_rows = rows[split_indices]
            split_cols = cols[split_indices]
            split_affinities = affinity[split_rows, split_cols]

            split_drugs = [drugs[i] for i in split_rows]
            split_prots = [prots[j] for j in split_cols]

            # Encode proteins
            split_prots_encoded = [encode_protein_sequence(p) for p in split_prots]

            # Prepare data & saves it to disk
            print(f'Preparing {dataset_name}_{split_name} dataset...')
            data_set = DrugTargetAffinityDataset(
                root='data',
                dataset=f'{dataset_name}_{split_name}',
                xd=split_drugs,
                xt=split_prots_encoded,
                y=split_affinities,
                smile_graph=smile_graph
            )

 


def main(datasets=['davis']):
    # Check if 'data' directory exists
    if not os.path.exists('data'):
        print("'data' directory not found.")
        success = download_and_extract_data()
        if not success:
            print("Failed to download and extract data. Exiting.")
            return
    else:
        print("'data' directory found. Proceeding with data preparation.")

    
    generate_pytorch_data(datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare datasets for drug-target affinity prediction.')
    parser.add_argument('--datasets', nargs='+', default=['davis'], help='List of datasets to process.')
    
    args = parser.parse_args()

    main(datasets=args.datasets)
