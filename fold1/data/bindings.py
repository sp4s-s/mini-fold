import os
import urllib.request
import polars as pl
import gzip

DATA_DIR = "./data"
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
        print("BindingDB dataset already downloaded.")
        
def extract_relevant_data():
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

    # Apply pick_affinity as a new column (using polars map)
    affinity = filtered.apply(pick_affinity, return_dtype=pl.Float64)
    filtered = filtered.with_columns(pl.Series("Affinity", affinity))
    filtered = filtered.filter(pl.col("Affinity").is_not_null())

    filtered.write_csv(PROCESSED_CSV)
    print(f"Filtered data saved to {PROCESSED_CSV}")

if __name__ == "__main__":
    download_bindingdb()
    extract_relevant_data()
