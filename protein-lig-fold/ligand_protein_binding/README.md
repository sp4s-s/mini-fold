# 🔬 Ligand-Protein Binding Prediction Pipeline

A comprehensive machine learning pipeline for predicting ligand-protein binding affinity using Graph Neural Networks and protein language models.

## 🚀 Quick Start

```bash
# Setup environment
python setup.py

# Authenticate services
wandb login
huggingface-cli login

# Run full pipeline
python run.py

# Run specific components
python run.py --mode data    # Data processing only
python run.py --mode train   # Training only
python run.py --mode inference  # Inference only
```

## 🏗️ Architecture

- **Protein Encoding**: ESM-2 transformer model (8M parameters)
- **Molecular Encoding**: Graph Neural Networks with GAT layers
- **Fusion**: Multi-head attention mechanism
- **Prediction**: Binary classification (strong/weak binding)

## 📊 Features

- ✅ Automated dataset preparation and upload to HuggingFace Hub
- ✅ Memory-efficient processing with automatic cache clearing
- ✅ Comprehensive experiment tracking with Weights & Biases
- ✅ Model versioning and deployment to HuggingFace Model Hub
- ✅ Independent inference capabilities
- ✅ GPU memory optimization (<20GB VRAM)

## 🔧 Configuration

Edit `configs/config.py` to modify:
- Model hyperparameters
- Training settings
- Dataset parameters
- Logging configuration

## 📁 Project Structure

```
ligand_protein_binding/
├── configs/           # Configuration files
├── src/              # Source code
├── data/             # Dataset storage
├── models/           # Trained models
├── scripts/          # Utility scripts
├── run.py           # Main pipeline script
└── README.md        # This file
```

## 🎯 Performance

The model achieves competitive performance on binding affinity prediction tasks with minimal computational requirements.

## 📄 License

MIT License
