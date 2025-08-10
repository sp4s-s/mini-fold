#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, create_repo, upload_folder
from datasets import Dataset as HFDataset
import gc
import json

sys.path.append('src')

from configs.config import config, data_config, wandb_config, hf_config
from src.data_processor import DataProcessor
from src.model import ProteinLigandPredictor
from src.trainer import Trainer
from src.dataset import ProteinLigandDataset, collate_fn

def download_and_prepare_data():
    """Download and prepare sample dataset"""
    print("📊 Preparing sample dataset...")
    
    # Create sample data (replace with actual dataset download)
    sample_data = {
        'sequence': [
            'MKFLVNVALVFMVVYISYIYAARVFLLGGFRVDDAKVTGAAQSAIRSTNHAKVTGLPDVDLVRLMLQSFPFDPRGNKTDLQKVAYGQCSILLTSVDNV',
            'MALVGAGLAVLAVGAGPAPAPPAPPHRPPPPPPAPPPPPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAPAP',
            'MKTLLILTCLVAVALASPGETALAQVTQIVKQFNTVDGVQTFLVRGFVTDKLATNVPQKIKGTLVDAKMSKLGVKRTQPVVFVPPVVQKQKSRQKRNRN'
        ],
        'smiles': [
            'CCO',
            'CC(=O)OC1=CC=CC=C1C(=O)O',
            'C1=CC=C(C=C1)CCN'
        ],
        'label': [1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_data.csv', index=False)
    
    # Upload to HuggingFace Hub
    try:
        hf_dataset = HFDataset.from_pandas(df)
        hf_dataset.push_to_hub(hf_config.dataset_repo, private=False)
        print(f"✅ Dataset uploaded to {hf_config.dataset_repo}")
    except Exception as e:
        print(f"⚠️ Failed to upload dataset: {e}")
    
    return df

def process_data(df):
    """Process raw data into model-ready format"""
    print("🔧 Processing data...")
    
    processor = DataProcessor()
    processed_data = processor.process_batch(df)
    processor.cleanup()
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return processed_data

def train_model(train_data, val_data, test_data):
    """Train the model"""
    print("🚀 Starting training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.run_name,
        config=config.__dict__
    )
    
    # Create datasets
    train_dataset = ProteinLigandDataset(train_data)
    val_dataset = ProteinLigandDataset(val_data)
    test_dataset = ProteinLigandDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                          shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = ProteinLigandPredictor(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_gnn_layers=config.gnn_layers
    )
    
    trainer = Trainer(model, device, config)
    
    best_val_auc = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{config.num_epochs}")
        
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_accuracy': val_metrics['accuracy']
        })
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            trainer.save_checkpoint('models/best_model.pt', epoch, val_metrics)
    
    # Final evaluation
    test_metrics = trainer.evaluate(test_loader)
    print(f"\n🎯 Test Results: AUC={test_metrics['auc']:.4f}, Acc={test_metrics['accuracy']:.4f}")
    
    wandb.log({'test_auc': test_metrics['auc'], 'test_accuracy': test_metrics['accuracy']})
    wandb.finish()
    
    return model, test_metrics

def upload_model_to_hf(model, metrics):
    """Upload trained model to Hugging Face Hub"""
    print("📤 Uploading model to Hugging Face Hub...")
    
    try:
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'metrics': metrics
        }, 'models/pytorch_model.bin')
        
        # Create model card
        model_card = f"""
# Ligand-Protein Binding Prediction Model

## Model Description
This model predicts ligand-protein binding affinity using:
- ESM-2 protein embeddings
- Graph Neural Networks for molecular representation
- Multi-head attention fusion

## Performance
- Test AUC: {metrics['auc']:.4f}
- Test Accuracy: {metrics['accuracy']:.4f}

## Usage
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("{hf_config.model_repo}")
```
"""
        
        with open('models/README.md', 'w') as f:
            f.write(model_card)
        
        # Create repository and upload
        api = HfApi()
        create_repo(hf_config.model_repo, exist_ok=True)
        upload_folder(
            folder_path='models/',
            repo_id=hf_config.model_repo,
            repo_type='model'
        )
        
        print(f"✅ Model uploaded to {hf_config.model_repo}")
        
    except Exception as e:
        print(f"⚠️ Failed to upload model: {e}")

def run_inference():
    """Run inference on sample data"""
    print("🔍 Running inference...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load('models/best_model.pt', map_location=device)
    model = ProteinLigandPredictor()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Sample inference data
    sample_data = pd.DataFrame({
        'sequence': ['MKTLLILTCLVAVALASPGETAL'],
        'smiles': ['CCO'],
        'label': [1]  # Ground truth for comparison
    })
    
    processor = DataProcessor()
    processed_sample = processor.process_batch(sample_data)
    processor.cleanup()
    
    with torch.no_grad():
        protein_emb = processed_sample[0]['protein_embedding'].unsqueeze(0).to(device)
        mol_data = processed_sample[0]['mol_graph'].to(device)
        
        prediction = model(protein_emb, mol_data)
        
        print(f"🎯 Prediction: {prediction.item():.4f}")
        print(f"📊 Ground Truth: {processed_sample[0]['label']}")

def main():
    parser = argparse.ArgumentParser(description='Ligand-Protein Binding Prediction Pipeline')
    parser.add_argument('--mode', choices=['data', 'train', 'eval', 'inference', 'full'], 
                       default='full', help='Pipeline mode')
    args = parser.parse_args()
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.mode in ['data', 'full']:
        df = download_and_prepare_data()
        processed_data = process_data(df)
        
        # Split data
        train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Save splits
        torch.save({'train': train_data, 'val': val_data, 'test': test_data}, 'data/processed_data.pt')
    
    if args.mode in ['train', 'full']:
        data_splits = torch.load('data/processed_data.pt')
        model, metrics = train_model(data_splits['train'], data_splits['val'], data_splits['test'])
        upload_model_to_hf(model, metrics)
    
    if args.mode in ['inference', 'full']:
        run_inference()
    
    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
