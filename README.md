# st2protein — core model training

Minimal training snapshot of the active SpatialChord regression model, taken from the server on 2026-09-22. Source code is preserved from `code/scChord`; historical models, comparison methods, preprocessing, datasets and experiment results are excluded.

## Files

- `dataset.py`: feature loading, splits and spatial neighbors.
- `metrics.py`: metrics used during validation.
- `spatialchord/train_reg.py`: model training entry point.
- `spatialchord/reg_models.py`: model architecture and fusion variants.
- `spatialchord/config.py`, `experiment.py`, `engine.py`, `losses.py`: training configuration, model construction, training loop and objectives.
- `spatialchord/artifacts.py`, `evaluation.py`: checkpoint handling and validation required by training.
- `spatialchord/evaluate_reg.py`: evaluation of saved checkpoints.
- `spatialchord/train_scvi_feature.py`: training the RNA feature encoder.

## Run

Use a Python environment with the appropriate PyTorch/CUDA installation. `requirements.txt` lists dependencies; it is not an environment lockfile. Spatial graph sampling may additionally require a compatible PyG sampling backend. `scvi-tools` is used by the RNA encoder.

Run from this repository root with your existing preprocessed data:

```bash
python spatialchord/train_reg.py \
  --experiment fusion_pc \
  --h5ad_path /path/to/processed/xenium_rna_prot.h5ad \
  --feature_dir /path/to/processed \
  --output_dir ./outputs/fusion_pc \
  --device cuda:0
```

The model consumes precomputed RNA and H&E features. Consult `python spatialchord/train_reg.py --help` for feature filenames and training options. Original server defaults remain in the source; override paths for your environment.

```bash
python spatialchord/evaluate_reg.py --run_dir ./outputs/fusion_pc
python spatialchord/train_scvi_feature.py --help
```

No data or weights are included. This upload does not rerun model training.
