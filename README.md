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
 
Model Trained
- https://huggingface.co/Pingsz/mini-fold-classifier-esm2-sub-esm2-650M/tree/main

 [12280/17268 3:10:36 < 1:17:26, 1.07 it/s, Epoch 2.13/3]
Epoch	Training Loss	Validation Loss	F1 Weighted	F1 Micro	F1 Macro
1	0.194900	0.119250	0.961667	0.962832	0.933105
2	0.093600	0.090340	0.971425	0.971921	0.950401
