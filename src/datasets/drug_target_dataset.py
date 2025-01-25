import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Tuple
import os

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


class DrugTargetAffinityDataset(InMemoryDataset):
    """
    Custom Dataset for Drug-Target Affinity Prediction using PyTorch Geometric.
    """
    # Rest of the class definition remains the same...
    def __init__(
        self,
        root: str = '/tmp',
        dataset: str = 'davis',
        xd: Optional[List[str]] = None,
        xt: Optional[List[np.ndarray]] = None,
        y: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        smile_graph: Optional[Dict[str, Tuple[int, np.ndarray, np.ndarray]]] = None,
    ):
        """
        Initializes the dataset.

        Args:
            root (str): Root directory where the dataset should be saved.
            dataset (str): Name of the dataset.
            xd (List[str], optional): List of SMILES strings.
            xt (List[np.ndarray], optional): List of encoded targets.
            y (List[float], optional): List of labels (affinities).
            transform (Callable, optional): A function/transform that takes in a Data object and returns a transformed version.
            pre_transform (Callable, optional): A function/transform that takes in a Data object and returns a transformed version before being saved to disk.
            smile_graph (Dict[str, Tuple[int, np.ndarray, np.ndarray]], optional): Mapping from SMILES to graph representations.
        """
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset = dataset
        if osp.isfile(self.processed_paths[0]):
            print(f'Pre-processed data found: {self.processed_paths[0]}, loading...')
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(f'Pre-processed data {self.processed_paths[0]} not found, processing...')
            if xd is None or xt is None or y is None or smile_graph is None:
                raise ValueError("xd, xt, y, and smile_graph must be provided for processing.")
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the raw file names.
        """
        return []
    

    def download(self):
        """
        Downloads the dataset.
        """
        # Implement this method if the dataset needs to be downloaded.
        pass

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the processed file names.
        """
        return [f'{self.dataset}.pt']
    
    @property
    def processed_paths(self) -> List[str]:
        """
        Returns the paths to the processed data files.
        """
        return [osp.join(self.processed_dir, f'{self.dataset}.pt')]

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(
        self,
        xd: List[str],
        xt: List[np.ndarray],
        y: List[float],
        smile_graph: Dict[str, Tuple[int, np.ndarray, np.ndarray]],
    ):
        """
        Processes the dataset and saves it to disk.

        Args:
            xd (List[str]): List of SMILES strings.
            xt (List[np.ndarray]): List of encoded targets.
            y (List[float]): List of labels (affinities).
            smile_graph (Dict[str, Tuple[int, np.ndarray, np.ndarray]]): Mapping from SMILES to graph representations.
        """
        assert len(xd) == len(xt) == len(y), "xd, xt, and y must have the same length."

        data_list = []
        for i in range(len(xd)):
            smiles = xd[i]
            target = xt[i]
            label = y[i]

            if smiles not in smile_graph:
                raise ValueError(f"SMILES '{smiles}' not found in smile_graph.")

            c_size, features, edge_index = smile_graph[smiles]

            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor([label], dtype=torch.float),
                target=torch.tensor(target, dtype=torch.long),
                c_size=torch.tensor([c_size], dtype=torch.long)
            )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


