import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import CachedPairDataset, collate_fn
from net import BindingPredictor
from utils import load_checkpoint
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    ds = CachedPairDataset(cfg.data.cache_dir, mode=cfg.data.mode, max_samples=cfg.data.max_samples)
    dl = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    model = BindingPredictor(cfg).to(device)
    ck = torch.load(cfg.checkpoint.resume, map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dl:
            for k,v in batch.items():
                batch[k] = v.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    print(f'Eval MSE: {mse:.6f}, R2: {r2:.4f}')

if __name__=='__main__':
    main()
