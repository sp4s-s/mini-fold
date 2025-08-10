import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Batch
from typing import Optional

class MolecularGNN(nn.Module):
    def __init__(self, node_dim: int = 9, edge_dim: int = 5, 
                 hidden_dim: int = 256, num_layers: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, 
                                    edge_dim=hidden_dim, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x = bn(x_new)
            x = F.relu(x)
            x = self.dropout(x)
        
        return global_mean_pool(x, batch)

class ProteinLigandPredictor(nn.Module):
    def __init__(self, protein_dim: int = 320, mol_hidden_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.1,
                 num_gnn_layers: int = 3):
        super().__init__()
        
        self.mol_gnn = MolecularGNN(
            hidden_dim=mol_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.mol_proj = nn.Linear(mol_hidden_dim, hidden_dim)
        
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, protein_emb, mol_data):
        batch_size = protein_emb.size(0)
        
        mol_emb = self.mol_gnn(
            mol_data.x, 
            mol_data.edge_index, 
            mol_data.edge_attr, 
            mol_data.batch
        )
        
        protein_emb = self.protein_proj(protein_emb)
        mol_emb = self.mol_proj(mol_emb)
        
        combined = torch.stack([protein_emb, mol_emb], dim=1)
        fused, _ = self.fusion(combined, combined, combined)
        fused = fused.mean(dim=1)
        
        return self.classifier(fused)
