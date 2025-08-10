#!/usr/bin/env python3

import sys
import torch
import pandas as pd
sys.path.append('../src')

from src.data_processor import DataProcessor
from src.model import ProteinLigandPredictor

def run_inference(sequence, smiles, model_path='../models/best_model.pt'):
    """Run inference on a single protein-ligand pair"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    checkpoint = torch.load(model_path, map_location=device)
    model = ProteinLigandPredictor()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process input
    df = pd.DataFrame({
        'sequence': [sequence],
        'smiles': [smiles],
        'label': [0]  # Dummy label
    })
    
    processor = DataProcessor()
    processed_data = processor.process_batch(df)
    processor.cleanup()
    
    if not processed_data:
        print("❌ Failed to process input data")
        return None
    
    # Run inference
    with torch.no_grad():
        protein_emb = processed_data[0]['protein_embedding'].unsqueeze(0).to(device)
        mol_data = processed_data[0]['mol_graph'].to(device)
        
        prediction = model(protein_emb, mol_data)
        binding_probability = prediction.item()
    
    return binding_probability

def main():
    # Demo examples
    examples = [
        {
            'name': 'Aspirin-Protein',
            'sequence': 'MKTLLILTCLVAVALASPGETALAQVTQIVKQFNTVDGVQTFLVRGFVTDKLATNVPQKIKGTLVDAKMSKLGVKRTQPVVFVPPVVQKQKSRQKRNRN',
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'
        },
        {
            'name': 'Ethanol-Protein',
            'sequence': 'MKFLVNVALVFMVVYISYIYAARVFLLGGFRVDDAKVTGAAQSAIRSTNHAKVTGLPDVDLVRLMLQSFPFDPRGNKTDLQKVAYGQCSILLTSVDNV',
            'smiles': 'CCO'
        }
    ]
    
    print("🧪 Running inference on demo examples...")
    
    for example in examples:
        print(f"\n📋 Example: {example['name']}")
        print(f"🧬 Protein: {example['sequence'][:50]}...")
        print(f"🧪 SMILES: {example['smiles']}")
        
        binding_prob = run_inference(example['sequence'], example['smiles'])
        
        if binding_prob is not None:
            binding_strength = "Strong" if binding_prob > 0.5 else "Weak"
            print(f"🎯 Binding Probability: {binding_prob:.4f} ({binding_strength})")
        else:
            print("❌ Inference failed")

if __name__ == "__main__":
    main()
