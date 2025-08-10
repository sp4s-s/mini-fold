import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import CachedPairDataset, collate_fn
from net import BindingPredictor
from utils import seed_everything, save_checkpoint, free_mem
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print('Config:\n', OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    dataset = CachedPairDataset(cfg.data.cache_dir, mode=cfg.data.mode, max_samples=cfg.data.max_samples)
    dl = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True)
    model = BindingPredictor(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp and device.type=='cuda')
    criterion = nn.MSELoss()
    start_epoch = 0
    if cfg.checkpoint.resume:
        ck = torch.load(cfg.checkpoint.resume, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        start_epoch = ck.get('epoch', 0)
    global_step = 0
    if cfg.logging.wandb and WANDB_AVAILABLE:
        wandb.init(project=cfg.logging.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(dl), total=len(dl), desc=f'Train epoch {epoch}')
        optimizer.zero_grad()
        for step, batch in pbar:
            for k,v in batch.items():
                batch[k] = v.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                preds = model(batch)
                loss = criterion(preds, batch['label'])
                loss = loss / cfg.training.accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % cfg.training.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            epoch_loss += loss.item() * cfg.training.accumulation_steps
            pbar.set_postfix({'loss': epoch_loss/(step+1)})
            if global_step % 100 == 0:
                free_mem()
        avg_loss = epoch_loss / len(dl)
        print(f'Epoch {epoch} train loss: {avg_loss:.6f}')
        ck_path = os.path.join(cfg.checkpoint.dir, f'ck_epoch_{epoch}.pt')
        save_checkpoint({'epoch': epoch+1, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, ck_path)
    if cfg.logging.wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    main()
