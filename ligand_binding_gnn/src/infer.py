import torch, hydra
from omegaconf import DictConfig
from net import BindingPredictor
from dataset import CachedPairDataset
import numpy as np

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    model = BindingPredictor(cfg).to(device)
    ck = torch.load(cfg.checkpoint.resume, map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    ds = CachedPairDataset(cfg.data.cache_dir, mode=cfg.data.mode, max_samples=1)
    sample = ds[0]
    batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k,v in sample.items() if k!='label'}
    with torch.no_grad():
        out = model(batch)
    print('Predicted affinity:', out.item())

if __name__=='__main__':
    main()
