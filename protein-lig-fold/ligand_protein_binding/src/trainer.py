import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import gc

class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            
            protein_emb = batch['protein_embedding'].to(self.device)
            mol_data = batch['mol_graph'].to(self.device)
            labels = batch['label'].float().to(self.device)
            
            outputs = self.model(protein_emb, mol_data).squeeze()
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        metrics = self.calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                protein_emb = batch['protein_embedding'].to(self.device)
                mol_data = batch['mol_graph'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                outputs = self.model(protein_emb, mol_data).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        metrics = self.calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def calculate_metrics(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        binary_preds = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(targets, binary_preds)
        auc = roc_auc_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, binary_preds, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, path)
