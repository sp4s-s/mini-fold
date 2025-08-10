import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    hidden_dim: int = 256
    gnn_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    max_protein_length: int = 1024

@dataclass
class DataConfig:
    dataset_name: str = "Pingsz/ligand_protein_binding"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class WandbConfig:
    project: str = "ligand-protein-binding"
    entity: Optional[str] = None
    run_name: Optional[str] = None

@dataclass
class HFConfig:
    model_repo: str = "Pingsz/lig_protein_binding"
    dataset_repo: str = "Pingsz/ligand_protein_binding_dataset"
    
config = ModelConfig()
data_config = DataConfig()
wandb_config = WandbConfig()
hf_config = HFConfig()
