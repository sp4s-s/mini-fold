Pipeline
```markdown
[Protein Sequence] ──► [ESM-2] ─┐
                               │
                               ├─► [Concat or CoAttention] ─► [Predictor (MLP or GNN)] ─► Binding Affinity
[SMILES] ───────────► [SMILES Transformer or GNN] ─┘
```