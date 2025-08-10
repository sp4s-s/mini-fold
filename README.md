Pipeline
```markdown
[Protein Sequence] ──► [ESM-2] ─┐
                               │
                               ├─► [Concat or CoAttention] ─► [Predictor (MLP or GNN)] ─► Binding Affinity
[SMILES] ───────────► [SMILES Transformer or GNN] ─┘
```

View Runs at
- https://wandb.ai/sp4ss-self/binding-affinity-mlp?nw=nwusersp4ss
- https://wandb.ai/sp4ss-self/huggingface?nw=nwusersp4ss

  Dataset created
  - https://huggingface.co/datasets/Pingsz/joint-feature-dataset

