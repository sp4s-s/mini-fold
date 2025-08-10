# Mini Ligand–Protein Binding GNN Project
- [Quickstart](#quickstart)
- [Smoke-Test](#smoke-test)


### Quickstart
1. Place your cached `.npz` files in `./cache` (files produced by your caching script: pair_*.npz).
2. Edit `configs/config.yaml` to point to your `cache_dir` and set `data.mode` to `embed` (default) or `graph`.
3. Run training:
   ```bash
   python3 src/train.py
   ```

Tips to keep under 24GB VRAM
- Use smaller `batch_size` (8,16) and increase `accumulation_steps`.
- Ensure `training.amp = true` for mixed precision.
- Use `graph` mode with smaller `graph_hidden` or fewer layers.
- Set `max_samples` in configs for quick smoke tests.


### SMoke-test
Quick smoke test: edit configs/config.yaml and set data.max_samples: 100 and training.epochs: 2. Then run:
python3 src/train.py
or inside the container (if you build it):
docker build -t binding-gnn .
docker run --gpus all -v $(pwd):/app -it binding-gnn
