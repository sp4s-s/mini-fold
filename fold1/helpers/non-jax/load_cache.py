import os
import numpy as np

CACHE_DIR = "./cache"

def load_all_embeddings():
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
    proteins, ligands, labels = [], [], []
    for f in files:
        data = np.load(os.path.join(CACHE_DIR, f))
        proteins.append(data['protein'])
        ligands.append(data['ligand'])
        labels.append(data['label'])
    return np.array(proteins), np.array(ligands), np.array(labels)

if __name__ == "__main__":
    proteins, ligands, labels = load_all_embeddings()
    print(f"Loaded {len(labels)} cached samples")
