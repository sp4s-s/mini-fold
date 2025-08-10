import os
import numpy as np
import torch
import polars as pl
from load_models import load_esm_model, load_smiles_transformer
from esm.data import BatchConverter

DATA_DIR = "./data"
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def embed_protein(sequence, esm_model, alphabet, device):
    batch_converter = BatchConverter(alphabet)
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:-1].mean(0).cpu().numpy()
    return embedding

def embed_ligand(smiles, tokenizer, model, device):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def cache_embeddings(df, esm_model, alphabet, esm_device, tokenizer, smiles_model, smiles_device):
    print(f"Processing {len(df)} pairs for embedding...")
    for idx, row in enumerate(df.iter_rows(named=True)):
        protein_seq = row['Protein Sequence']
        ligand_smiles = row['Ligand SMILES']
        affinity = row['Affinity']
        try:
            p_emb = embed_protein(protein_seq, esm_model, alphabet, esm_device)
            l_emb = embed_ligand(ligand_smiles, tokenizer, smiles_model, smiles_device)
            filename = os.path.join(CACHE_DIR, f"pair_{idx}.npz")
            np.savez(filename, protein=p_emb, ligand=l_emb, label=affinity)
            if idx % 100 == 0:
                print(f"Saved embeddings for pair {idx}")
        except Exception as e:
            print(f"Failed to process index {idx}: {e}")

def load_cached_embedding(filepath):
    data = np.load(filepath)
    return data['protein'], data['ligand'], data['label']

if __name__ == "__main__":
    df = pl.read_csv(os.path.join(DATA_DIR, "bindingdb_filtered.csv"))
    esm_model, alphabet, esm_device = load_esm_model()
    tokenizer, smiles_model, smiles_device = load_smiles_transformer()
    cache_embeddings(df, esm_model, alphabet, esm_device, tokenizer, smiles_model, smiles_device)
