import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data, DataLoader
from transformers import EsmTokenizer, EsmModel
import gc
from typing import List, Tuple, Dict
from tqdm import tqdm

class MolecularGraph:
    def __init__(self):
        self.atom_features_dim = 9
        
    def smiles_to_graph(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        
        node_features = []
        for atom in atoms:
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetMass()
            ]
            node_features.append(features)
        
        edge_indices = []
        edge_attrs = []
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
            
            bond_type = bond.GetBondType()
            bond_features = [
                float(bond_type == Chem.rdchem.BondType.SINGLE),
                float(bond_type == Chem.rdchem.BondType.DOUBLE),
                float(bond_type == Chem.rdchem.BondType.TRIPLE),
                float(bond.GetIsAromatic()),
                float(bond.IsInRing())
            ]
            edge_attrs.extend([bond_features, bond_features])
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class ProteinEmbedder:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def embed_protein(self, sequence: str, max_length: int = 1024) -> torch.Tensor:
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
            
        inputs = self.tokenizer(sequence, return_tensors="pt", 
                              padding=True, truncation=True, 
                              max_length=max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu()
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class DataProcessor:
    def __init__(self):
        self.mol_graph = MolecularGraph()
        self.protein_embedder = ProteinEmbedder()
        
    def process_batch(self, batch_data: pd.DataFrame) -> List[Dict]:
        processed_data = []
        
        for idx, row in tqdm(batch_data.iterrows(), total=len(batch_data)):
            try:
                mol_graph = self.mol_graph.smiles_to_graph(row['smiles'])
                if mol_graph is None:
                    continue
                    
                protein_emb = self.protein_embedder.embed_protein(row['sequence'])
                
                processed_data.append({
                    'mol_graph': mol_graph,
                    'protein_embedding': protein_emb,
                    'label': row['label']
                })
                
                if len(processed_data) % 100 == 0:
                    self.protein_embedder.clear_cache()
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
                
        return processed_data
    
    def cleanup(self):
        del self.protein_embedder.model
        self.protein_embedder.clear_cache()
