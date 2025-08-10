import torch
import esm
from transformers import AutoTokenizer, AutoModel
import os
    
def load_esm_650m_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("Loading ESM-2 3B model...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50S()
    esm_model = esm_model.to(device)
    esm_model.eval()
    return esm_model, alphabet, device

# SMILES Transformer model loader
def load_smiles_transformer(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("Loading SMILES Transformer...")
    tokenizer = AutoTokenizer.from_pretrained("lvsn/smiles_transformer")
    model = AutoModel.from_pretrained("lvsn/smiles_transformer").to(device)
    model.eval()
    return tokenizer, model, device


def load_esm_3b_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("Loading ESM-2 3B model...")
    # This is esm2_t36_3B_UR50D, the official 3B param esm-2 model from Meta
    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    return esm_model, alphabet, device