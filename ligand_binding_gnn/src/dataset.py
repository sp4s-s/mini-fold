import os, glob, numpy as np
from torch.utils.data import Dataset
try:
    from rdkit import Chem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

class CachedPairDataset(Dataset):
    def __init__(self, cache_dir, mode='embed', max_samples=None):
        self.files = sorted(glob.glob(os.path.join(cache_dir, 'pair_*.npz')))
        if max_samples:
            self.files = self.files[:max_samples]
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        npz = np.load(f, allow_pickle=True)
        protein = npz['protein'].astype(np.float32)
        item = {'protein': protein}
        if 'ligand' in npz:
            item['ligand'] = npz['ligand'].astype(np.float32)
        else:
            if 'atom_feats' in npz and 'adj' in npz:
                item['atom_feats'] = npz['atom_feats'].astype(np.float32)
                item['adj'] = npz['adj'].astype(np.float32)
            else:
                raise RuntimeError(f'No ligand representation found in {f}')
        item['label'] = float(npz['label'].item())
        return item

import torch
def collate_fn(batch):
    if 'ligand' in batch[0]:
        proteins = torch.tensor([b['protein'] for b in batch])
        ligands = torch.tensor([b['ligand'] for b in batch])
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
        return {'protein': proteins, 'ligand': ligands, 'label': labels}
    else:
        max_nodes = max(b['atom_feats'].shape[0] for b in batch)
        feat_dim = batch[0]['atom_feats'].shape[1]
        atom_feats = torch.zeros((len(batch), max_nodes, feat_dim), dtype=torch.float32)
        adjs = torch.zeros((len(batch), max_nodes, max_nodes), dtype=torch.float32)
        masks = torch.zeros((len(batch), max_nodes), dtype=torch.bool)
        for i,b in enumerate(batch):
            n = b['atom_feats'].shape[0]
            atom_feats[i,:n] = torch.tensor(b['atom_feats'])
            adjs[i,:n,:n] = torch.tensor(b['adj'])
            masks[i,:n] = 1
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
        proteins = torch.tensor([b['protein'] for b in batch])
        return {'protein': proteins, 'atom_feats': atom_feats, 'adj': adjs, 'mask': masks, 'label': labels}
