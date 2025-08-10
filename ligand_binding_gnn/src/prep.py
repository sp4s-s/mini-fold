#!/usr/bin/env python3
"""
preprocess_and_upload.py

Single-file preprocessor:
- Downloads BindingDB All TSV (gz)
- Filters for Protein Sequence, Ligand SMILES, and affinity columns
- Loads ESM-2 650M (preferred local checkpoint if available)
- Loads SMILES transformer
- Embeds protein sequences and ligand SMILES -> saves ./cache/pair_{idx}.npz
- Optionally uploads processed CSV and cache files to Hugging Face dataset repo

Usage:
  python src/preprocess_and_upload.py --max-samples 100 --upload --hf-username yourname

Requirements:
  pip install polars torch transformers huggingface_hub esm tqdm
  (If you don't have esm or it's incompatible, put a local ESM checkpoint and pass --local-esm models/esm2_t33_650M_UR50S.pt)
"""
import os
import sys
import argparse
import urllib.request
import gzip
import math
from pathlib import Path
import shutil
import zipfile
import time
import traceback

import polars as pl
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo

# Optional: esm import may vary by installed package/version
try:
    import esm
except Exception:
    esm = None

from tqdm import tqdm

# ---- CONFIG DEFAULTS ----
DATA_DIR = Path("./data")
CACHE_DIR = Path("./cache")
MODELS_DIR = Path("./models")
BINDINGDB_URL = "https://www.bindingdb.org/chemsearch/marvin/BindingDB_All.tsv.gz"
BINDINGDB_PATH = DATA_DIR / "BindingDB_All.tsv.gz"
PROCESSED_CSV = DATA_DIR / "bindingdb_filtered.csv"
LOCAL_ESM_DEFAULT = MODELS_DIR / "esm2_t33_650M_UR50S.pt"

# ---- FUNCTIONS ----

def download_bindingdb(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if BINDINGDB_PATH.exists() and not force:
        print(f"[skip] BindingDB archive already exists at {BINDINGDB_PATH}")
        return
    print("Downloading BindingDB (this is large, ~400 MB)...")
    urllib.request.urlretrieve(BINDINGDB_URL, str(BINDINGDB_PATH))
    print("Download complete.")

def extract_relevant_data(force=False):
    if PROCESSED_CSV.exists() and not force:
        print(f"[skip] Processed CSV already exists at {PROCESSED_CSV}")
        return
    print("Filtering BindingDB and writing processed CSV... (this can take some time)")
    with gzip.open(BINDINGDB_PATH, "rt") as f:
        df = pl.read_csv(f, sep="\t", try_parse_dates=False)

    # keep columns and non-null sequences & smiles
    want_cols = ["Protein Sequence", "Ligand SMILES", "Ki (nM)", "Kd (nM)", "IC50 (nM)"]
    available = [c for c in want_cols if c in df.columns]
    if "Protein Sequence" not in available or "Ligand SMILES" not in available:
        raise RuntimeError("BindingDB TSV doesn't contain required columns; file format may have changed.")

    filtered = df.select(available).filter(
        (pl.col("Protein Sequence").is_not_null()) & (pl.col("Ligand SMILES").is_not_null())
    )

    # helper to pick first available affinity (Ki/Kd/IC50)
    def pick_affinity(row):
        for c in ["Ki (nM)", "Kd (nM)", "IC50 (nM)"]:
            if c in row and row[c] not in (None, ""):
                try:
                    return float(row[c])
                except Exception:
                    return None
        return None

    affinities = filtered.apply(pick_affinity, return_dtype=pl.Float64)
    filtered = filtered.with_columns(pl.Series("Affinity", affinities))
    filtered = filtered.filter(pl.col("Affinity").is_not_null())

    filtered.write_csv(PROCESSED_CSV)
    print(f"Wrote processed CSV to {PROCESSED_CSV} (rows: {len(filtered)})")

# --- ESM loader with fallbacks ---
def load_esm_650m_model(local_path: Path = None, device=None, verbose=True):
    """
    Try these in order:
    1) If local_path exists: use esm.pretrained.load_model_and_alphabet_local(local_path)
    2) If esm.pretrained has loader functions: try load_model_and_alphabet_hub or esm2_t33_650M_UR50S()
    3) If nothing works, raise with helpful message.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if esm is None:
        raise RuntimeError("`esm` package not importable. Install the FAIR ESM package (`pip install fair-esm`) or provide a local checkpoint.")

    # prefer a local checkpoint if provided
    if local_path and Path(local_path).exists():
        lp = str(local_path)
        if verbose: print(f"Loading local ESM checkpoint from {lp}")
        try:
            # many esm versions provide this
            if hasattr(esm, "pretrained") and hasattr(esm.pretrained, "load_model_and_alphabet_local"):
                model, alphabet = esm.pretrained.load_model_and_alphabet_local(lp)
            else:
                # last-ditch: try to load via torch and instantiate model via esm API (rare)
                raise AttributeError("installed 'esm' doesn't support load_model_and_alphabet_local")
            model = model.to(device)
            model.eval()
            return model, alphabet, device
        except Exception as e:
            print("Failed to load local ESM checkpoint:", e)
            print("Falling back to hub/local API attempts...")

    # try hub variants
    if hasattr(esm, "pretrained"):
        pre = esm.pretrained
        # try explicit hub loader
        try:
            if hasattr(pre, "load_model_and_alphabet_hub"):
                if verbose: print("Loading ESM from hub via load_model_and_alphabet_hub('esm2_t33_650M_UR50S') ...")
                model, alphabet = pre.load_model_and_alphabet_hub("esm2_t33_650M_UR50S")
                model = model.to(device)
                model.eval()
                return model, alphabet, device
        except Exception as e:
            print("Hub loader load_model_and_alphabet_hub failed:", e)
        # try convenience function if present
        try:
            if hasattr(pre, "esm2_t33_650M_UR50S"):
                if verbose: print("Loading ESM via esm.pretrained.esm2_t33_650M_UR50S() ...")
                model, alphabet = pre.esm2_t33_650M_UR50S()
                model = model.to(device)
                model.eval()
                return model, alphabet, device
        except Exception as e:
            print("esm.pretrained.esm2_t33_650M_UR50S() failed:", e)

    # fallback: maybe esm has load_model_and_alphabet (older/newer)
    try:
        if hasattr(esm, "load_model_and_alphabet"):
            if verbose: print("Loading ESM via esm.load_model_and_alphabet() ...")
            model, alphabet = esm.load_model_and_alphabet("esm2_t33_650M_UR50S")
            model = model.to(device)
            model.eval()
            return model, alphabet, device
    except Exception as e:
        print("esm.load_model_and_alphabet failed:", e)

    # nothing worked
    raise RuntimeError(
        "Could not load ESM model. Options:\n"
        " - install fair-esm (pip install fair-esm) and allow downloads\n"
        " - or download the checkpoint manually to models/esm2_t33_650M_UR50S.pt and pass --local-esm\n"
        "See script comments for details."
    )

# --- SMILES transformer loader ---
def load_smiles_transformer(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading SMILES transformer tokenizer+model (this downloads from Hugging Face if needed)...")
    tok = AutoTokenizer.from_pretrained("lvsn/smiles_transformer")
    model = AutoModel.from_pretrained("lvsn/smiles_transformer").to(device)
    model.eval()
    return tok, model, device

# --- Embedding helpers ---
def embed_protein(sequence: str, esm_model, alphabet, device):
    """
    Use alphabet.get_batch_converter() for compatibility.
    Returns numpy 1D vector.
    """
    batch_converter = None
    if hasattr(alphabet, "get_batch_converter"):
        batch_converter = alphabet.get_batch_converter()
    else:
        # some versions put BatchConverter elsewhere; try esm.data.BatchConverter
        try:
            from esm.data import BatchConverter
            # If alphabet contains a mapping, create a BatchConverter(alphabet) - fallback
            batch_converter = BatchConverter(alphabet)
        except Exception:
            raise RuntimeError("Could not obtain batch converter from ESM alphabet. Incompatible 'esm' package version.")

    data = [("prot", sequence)]
    try:
        labels, strs, tokens = batch_converter(data)
    except Exception:
        # some variants return only 2-tuple
        out = batch_converter(data)
        if len(out) == 3:
            labels, strs, tokens = out
        elif len(out) == 2:
            labels, tokens = out
            strs = None
        else:
            raise

    tokens = tokens.to(device)
    with torch.no_grad():
        # choose a high-level repr layer: many esm2 models have 33 as final layer; if missing, model may accept repr_layers argument
        try:
            results = esm_model(tokens, repr_layers=[33], return_contacts=False)
            reps = results["representations"][33]
        except Exception:
            # fallback to forward and take last_hidden_state
            out = esm_model(tokens)
            # some ESM models return a tuple, some dicts
            if isinstance(out, dict) and "representations" in out and 33 in out["representations"]:
                reps = out["representations"][33]
            elif isinstance(out, tuple) and len(out) > 0:
                # last_hidden_state often at index 0
                reps = out[0]
            else:
                raise RuntimeError("Unable to extract ESM representations from model forward pass.")
    # reps: [batch, seq_len, dim] ; take mean over residues excluding BOS/EOS if present
    reps = reps.cpu().numpy()
    # if sequence tokens include special tokens at ends, the ESM convention usually has them; try to remove first and last if length>2
    v = reps[0]
    if v.shape[0] > 2:
        v = v[1:-1].mean(axis=0)
    else:
        v = v.mean(axis=0)
    return v.astype(np.float32)

def embed_ligand(smiles: str, tokenizer, model, device):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        # transformer last_hidden_state is typical
        if hasattr(out, "last_hidden_state"):
            hid = out.last_hidden_state
        elif isinstance(out, tuple) and len(out) > 0:
            hid = out[0]
        else:
            raise RuntimeError("Could not extract last_hidden_state from SMILES model output.")
    v = hid.mean(dim=1).squeeze(0).cpu().numpy()
    return v.astype(np.float32)

# --- Cache writing/loading ---
def cache_embeddings_from_df(df: pl.DataFrame, esm_model, alphabet, esm_device, tokenizer, smiles_model, smiles_device, max_samples=None, force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df) if max_samples is None else min(len(df), int(max_samples))
    print(f"Embedding up to {n} samples (max_samples={max_samples})")
    for idx, row in enumerate(tqdm(df.iter_rows(named=True), total=n)):
        if idx >= n:
            break
        out_file = CACHE_DIR / f"pair_{idx}.npz"
        if out_file.exists() and not force:
            continue
        prot = row["Protein Sequence"]
        smi = row["Ligand SMILES"]
        lab = row["Affinity"]
        try:
            p_emb = embed_protein(prot, esm_model, alphabet, esm_device)
            l_emb = embed_ligand(smi, tokenizer, smiles_model, smiles_device)
            np.savez_compressed(str(out_file), protein=p_emb, ligand=l_emb, label=float(lab))
            if (idx + 1) % 100 == 0:
                print(f"Saved {idx+1}/{n} embeddings.")
        except Exception as e:
            print(f"[warn] failed embedding idx={idx}: {e}")
            traceback.print_exc()

def load_all_npz(cache_dir=Path("./cache")):
    files = sorted([p for p in Path(cache_dir).glob("pair_*.npz")])
    proteins, ligands, labels = [], [], []
    for p in files:
        data = np.load(p)
        proteins.append(data["protein"])
        ligands.append(data["ligand"])
        labels.append(float(data["label"].item()))
    return np.stack(proteins), np.stack(ligands), np.array(labels)

# --- Hugging Face upload ---
def upload_to_hf(hf_token, hf_username, repo_name, private=False, zip_first=False, zip_name="dataset_bundle.zip"):
    if not hf_token:
        raise ValueError("HF token required for upload. Set HF_TOKEN env or pass via --hf-token.")
    HfFolder.save_token(hf_token)
    api = HfApi()
    repo_id = f"{hf_username}/{repo_name}"
    print(f"Creating/ensuring dataset repo: {repo_id} (private={private})")
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=hf_token)
    except Exception as e:
        print("Repo create warning:", e)

    # Optionally zip the whole data/cache to one file for faster upload & download
    if zip_first:
        print("Zipping data + cache to", zip_name)
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            if PROCESSED_CSV.exists():
                zf.write(PROCESSED_CSV, arcname=PROCESSED_CSV.name)
            for p in sorted(CACHE_DIR.glob("pair_*.npz")):
                zf.write(p, arcname=f"cache/{p.name}")
        print("Uploading zip artifact to HF...")
        upload_file(path_or_fileobj=zip_name, path_in_repo=os.path.basename(zip_name),
                    repo_id=repo_id, repo_type="dataset", token=hf_token)
        print("Uploaded zip.")
        return

    # Upload CSV
    if PROCESSED_CSV.exists():
        print("Uploading processed CSV...")
        upload_file(path_or_fileobj=str(PROCESSED_CSV), path_in_repo=PROCESSED_CSV.name,
                    repo_id=repo_id, repo_type="dataset", token=hf_token)
    else:
        print("[warn] processed CSV not found; skipping CSV upload.")

    # Upload NPZs (in 'cache/' subfolder)
    files = sorted(CACHE_DIR.glob("pair_*.npz"))
    print(f"Uploading {len(files)} .npz files to dataset repo (under 'cache/' folder).")
    for p in tqdm(files):
        path_in_repo = f"cache/{p.name}"
        upload_file(path_or_fileobj=str(p), path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type="dataset", token=hf_token)
    print("HF upload complete.")

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force-download", action="store_true", help="Redownload BindingDB archive")
    p.add_argument("--force-extract", action="store_true", help="Re-extract and rewrite processed CSV")
    p.add_argument("--force-embed", action="store_true", help="Recompute embeddings even if cache exists")
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of pairs to process (for smoke tests)")
    p.add_argument("--local-esm", type=str, default=str(LOCAL_ESM_DEFAULT), help="Path to local ESM checkpoint (.pt). If absent, tries hub (may attempt download).")
    p.add_argument("--upload", action="store_true", help="Upload processed files to Hugging Face after processing")
    p.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""), help="Hugging Face token (or set HF_TOKEN env var)")
    p.add_argument("--hf-username", type=str, default=os.getenv("HF_USERNAME", ""), help="Hugging Face username (or set HF_USERNAME env var)")
    p.add_argument("--hf-repo", type=str, default="mini_ligand_binding_dataset", help="HF dataset repo name")
    p.add_argument("--hf-private", action="store_true", help="Make HF repo private")
    p.add_argument("--zip-upload", action="store_true", help="Zip data+cache before uploading (single file upload)")
    p.add_argument("--skip-embed", action="store_true", help="Skip embedding step (assume cache already exists)")
    p.add_argument("--no-smiles-model", action="store_true", help="Skip loading SMILES transformer (if you only want protein embeddings)")
    return p.parse_args()

def main():
    args = parse_args()
    print("Starting preprocessor with args:", args)
    # ensure dirs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: download & extract
    download_bindingdb(force=args.force_download)
    extract_relevant_data(force=args.force_extract)

    # Load DataFrame
    df = pl.read_csv(PROCESSED_CSV)
    if args.max_samples:
        df = df[: args.max_samples]

    # Step 2: load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    esm_model = None
    alphabet = None
    if not args.skip_embed:
        try:
            esm_model, alphabet, esm_device = load_esm_650m_model(local_path=Path(args.local_esm), device=device)
            print("Loaded ESM model on", esm_device)
        except Exception as e:
            print("[ERROR] Could not load ESM model:", e)
            print("If your environment blocks downloads, place the checkpoint at the path specified by --local-esm")
            raise

        # smiles transformer
        tokenizer = None
        smiles_model = None
        if not args.no_smiles_model:
            try:
                tokenizer, smiles_model, smiles_device = load_smiles_transformer(device=device)
            except Exception as e:
                print("[ERROR] Could not load SMILES transformer:", e)
                raise
        else:
            tokenizer = None
            smiles_model = None

        # Step 3: embed and cache
        cache_embeddings_from_df(df, esm_model, alphabet, esm_device, tokenizer, smiles_model, smiles_device,
                                 max_samples=args.max_samples, force=args.force_embed)
    else:
        print("[skip] embedding step as requested (--skip-embed).")

    # Optional: Upload to HF
    if args.upload:
        hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
        hf_username = args.hf_username or os.getenv("HF_USERNAME", "")
        if not hf_token or not hf_username:
            raise ValueError("To upload to HF, supply --hf-token and --hf-username or set HF_TOKEN and HF_USERNAME env vars.")
        upload_to_hf(hf_token=hf_token, hf_username=hf_username, repo_name=args.hf_repo, private=args.hf_private, zip_first=args.zip_upload)
    print("Done.")

if __name__ == "__main__":
    main()
