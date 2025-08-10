import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import List, Dict
import pandas as pd

class ProteinLigandDataset(Dataset):
    def __init__(self, processed_data: List[Dict]):
        self.data = processed_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    protein_embeddings = torch.stack([item['protein_embedding'].squeeze() for item in batch])
    mol_graphs = Batch.from_data_list([item['mol_graph'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
    
    return {
        'protein_embedding': protein_embeddings,
        'mol_graph': mol_graphs,
        'label': labels
    }
