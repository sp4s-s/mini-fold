import torch, torch.nn as nn

class DenseGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = nn.Linear(in_dim, out_dim)
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x, adj):
        h_self = self.lin_node(x)
        m = torch.matmul(adj, x)
        h_msg = self.lin_msg(m)
        out = self.act(h_self + h_msg)
        return out

class LigandGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            idim = in_dim if i==0 else hidden
            self.layers.append(DenseGNNLayer(idim, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_feats, adj, mask=None):
        x = atom_feats
        for layer in self.layers:
            x = layer(x, adj)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(x.dtype) if mask is not None else x.new_tensor(x.shape[1])
        pooled = x.sum(dim=1) / denom.squeeze(-1)
        return self.dropout(pooled)

class EmbedEncoder(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class FusionPredictor(nn.Module):
    def __init__(self, protein_dim, ligand_dim, hidden, fusion='concat', dropout=0.1):
        super().__init__()
        self.fusion = fusion
        if fusion == 'concat':
            inp = protein_dim + ligand_dim
            self.head = nn.Sequential(
                nn.Linear(inp, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2),
                nn.GELU(),
                nn.Linear(hidden//2, 1)
            )
        elif fusion == 'coattn':
            self.proj_p = nn.Linear(protein_dim, hidden)
            self.proj_l = nn.Linear(ligand_dim, hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden*2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1)
            )
        else:
            raise ValueError('fusion must be concat or coattn')

    def forward(self, p, l):
        if self.fusion == 'concat':
            x = torch.cat([p, l], dim=1)
            return self.head(x).squeeze(1)
        else:
            P = self.proj_p(p)
            L = self.proj_l(l)
            x = torch.cat([P, L], dim=1)
            return self.head(x).squeeze(1)

class BindingPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.data.mode
        self.fusion = cfg.model.fusion
        self.protein_proj = nn.Sequential(nn.Linear(cfg.model.protein_dim, cfg.model.hidden), nn.GELU())
        if self.mode == 'graph':
            self.ligand_encoder = LigandGraphEncoder(in_dim=cfg.model.graph_in_dim if hasattr(cfg.model,'graph_in_dim') else 34,
                                                     hidden=cfg.model.graph_hidden,
                                                     num_layers=cfg.model.graph_layers,
                                                     dropout=cfg.model.dropout)
            ligand_out_dim = cfg.model.graph_hidden
        else:
            self.ligand_encoder = EmbedEncoder(cfg.model.ligand_dim, cfg.model.hidden, dropout=cfg.model.dropout)
            ligand_out_dim = cfg.model.hidden
        prot_out_dim = cfg.model.hidden
        self.predictor = FusionPredictor(protein_dim=prot_out_dim, ligand_dim=ligand_out_dim, hidden=cfg.model.hidden, fusion=self.fusion, dropout=cfg.model.dropout)

    def forward(self, batch):
        p = batch['protein']
        p = self.protein_proj(p)
        if 'ligand' in batch:
            l = self.ligand_encoder(batch['ligand'])
        else:
            l = self.ligand_encoder(batch['atom_feats'], batch['adj'], mask=batch['mask'])
        out = self.predictor(p, l)
        return out
