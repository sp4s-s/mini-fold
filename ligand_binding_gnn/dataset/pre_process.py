# from esm.data import BatchConverter
# import esm

# esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50S()
# batch_converter = alphabet.get_batch_converter()



import torch
import esm

# This will give you the ESM-2 650M model & alphabet
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(
    "esm2_t33_650M_UR50S"
)
batch_converter = alphabet.get_batch_converter()

esm_model = esm_model.to("cuda" if torch.cuda.is_available() else "cpu")
esm_model.eval()












import os
import urllib.request
import gzip
import polars as pl
import torch
import esm
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo


HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
HF_USERNAME = "Pingsz"
REPO_NAME = "mini_ligand_binding_dataset"
PRIVATE = False

DATA_DIR = "./data"
CACHE_DIR = "./cache"
BINDINGDB_URL = "https://www.bindingdb.org/chemsearch/marvin/BindingDB_All.tsv.gz"
BINDINGDB_PATH = os.path.join(DATA_DIR, "BindingDB_All.tsv.gz")
PROCESSED_CSV = os.path.join(DATA_DIR, "bindingdb_filtered.csv")


def download_bindingdb():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(BINDINGDB_PATH):
        print("Downloading BindingDB dataset...")
        urllib.request.urlretrieve(BINDINGDB_URL, BINDINGDB_PATH)
        print("Download complete.")
    else:
        print("BindingDB dataset already exists.")


def extract_relevant_data():
    if os.path.exists(PROCESSED_CSV):
        print(f"{PROCESSED_CSV} already exists. Skipping.")
        return

    print("Loading and filtering BindingDB with polars...")
    with gzip.open(BINDINGDB_PATH, 'rt') as f:
        df = pl.read_csv(f, sep='\t')

    filtered = df.select([
        "Protein Sequence",
        "Ligand SMILES",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)"
    ]).filter(
        (pl.col("Protein Sequence").is_not_null()) &
        (pl.col("Ligand SMILES").is_not_null())
    )

    def pick_affinity(row):
        for col in ["Ki (nM)", "Kd (nM)", "IC50 (nM)"]:
            val = row[col]
            if val is not None:
                try:
                    return float(val)
                except:
                    return None
        return None

    affinity = filtered.apply(pick_affinity, return_dtype=pl.Float64)
    filtered = filtered.with_columns(pl.Series("Affinity", affinity))
    filtered = filtered.filter(pl.col("Affinity").is_not_null())

    filtered.write_csv(PROCESSED_CSV)
    print(f"Filtered data saved to {PROCESSED_CSV}")

def load_esm_650m_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("Loading ESM-2 650M model...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50S()
    esm_model = esm_model.to(device)
    esm_model.eval()
    return esm_model, alphabet, device

def load_smiles_transformer(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("Loading SMILES Transformer...")
    tokenizer = AutoTokenizer.from_pretrained("lvsn/smiles_transformer")
    model = AutoModel.from_pretrained("lvsn/smiles_transformer").to(device)
    model.eval()
    return tokenizer, model, device


def embed_protein(sequence, esm_model, alphabet, device):
    batch_converter = BatchConverter(alphabet)
    data = [("protein1", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:-1].mean(0).cpu().numpy()
    return jnp.array(embedding)

def embed_ligand(smiles, tokenizer, model, device):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return jnp.array(embeddings)


# Cache embeddings
def cache_embeddings(df, esm_model, alphabet, esm_device, tokenizer, smiles_model, smiles_device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for idx, row in enumerate(df.iter_rows(named=True)):
        out_file = os.path.join(CACHE_DIR, f"pair_{idx}.npz")
        if os.path.exists(out_file):
            continue
        try:
            p_emb = embed_protein(row['Protein Sequence'], esm_model, alphabet, esm_device)
            l_emb = embed_ligand(row['Ligand SMILES'], tokenizer, smiles_model, smiles_device)
            np.savez(out_file, protein=np.array(p_emb), ligand=np.array(l_emb), label=row['Affinity'])
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)}")
        except Exception as e:
            print(f"Failed at {idx}: {e}")


# Upload to HF
def upload_to_hf():
    # if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
    #     raise ValueError("Please set your HF_TOKEN at the top or via environment variable.")

    HfFolder.save_token(HF_TOKEN)
    api = HfApi()
    full_repo_name = f"{HF_USERNAME}/{REPO_NAME}"

    try:
        create_repo(full_repo_name, repo_type="dataset", private=PRIVATE, exist_ok=True, token=HF_TOKEN)
        print(f"Repo ready: https://huggingface.co/datasets/{full_repo_name}")
    except Exception as e:
        print(f"Repo creation warning: {e}")

    # Upload CSV
    if os.path.exists(PROCESSED_CSV):
        upload_file(path_or_fileobj=PROCESSED_CSV,
                    path_in_repo=os.path.basename(PROCESSED_CSV),
                    repo_id=full_repo_name,
                    repo_type="dataset",
                    token=HF_TOKEN)

    for file in os.listdir(CACHE_DIR):
        if file.endswith(".npz"):
            upload_file(path_or_fileobj=os.path.join(CACHE_DIR, file),
                        path_in_repo=f"cache/{file}",
                        repo_id=full_repo_name,
                        repo_type="dataset",
                        token=HF_TOKEN)

    print("✅ Upload complete.")


if __name__ == "__main__":
    download_bindingdb()
    extract_relevant_data()

    df = pl.read_csv(PROCESSED_CSV)
    esm_model, alphabet, esm_device = load_esm_650m_model()
    tokenizer, smiles_model, smiles_device = load_smiles_transformer()
    cache_embeddings(df, esm_model, alphabet, esm_device, tokenizer, smiles_model, smiles_device)

    upload_to_hf()
