# Spatial Continuity Filling Network

This module demonstrates the repair component used by SCF-3DMask. It learns to
predict local repair patches from sparse or damaged object proxies.

## Files

- `model.py`: network architecture wrapper.
- `dataset.py`: self-supervised damage-and-repair dataset.
- `train.py`: training entry point.
- `inference.py`: inference entry point for a sparse proxy.

The implementation is imported from `src/scf3dmask/repair` to avoid duplicate
code.

Install the package from the repository root first (`python -m pip install -e .`).
The commands below assume this module directory is the current directory.

## Train

```bash
python train.py --data-dir ../../data/repair_train --output-dir ../../checkpoints
```

The training directory can contain `.off` meshes or `.ply` point clouds. The
dataset samples a clean local point set, applies synthetic filtering/slicing
damage, and uses the removed points as repair labels.

## Inference

```bash
python inference.py \
  --input ../../outputs/example/01_sparse_proxy.ply \
  --ckpt ../../checkpoints/best_model.pth \
  --output ../../outputs/example/repair_vis.ply \
  --patch-output ../../outputs/example/03_repair_patch.ply
```

The colored output uses gray for the input sparse proxy and red for the
predicted repair patch.
