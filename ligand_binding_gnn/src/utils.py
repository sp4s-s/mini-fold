import os, torch, random, numpy as np, gc

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)

def free_mem():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
