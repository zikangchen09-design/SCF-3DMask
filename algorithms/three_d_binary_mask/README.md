# 3D Binary Mask Generator

This module demonstrates the normal-direction-guided adaptive tomographic mask
used by SCF-3DMask.

Given a repaired sparse reference and an original dense reconstructed point
cloud, the generator:

1. Estimates normals on the reference point cloud.
2. Splits the reference into thin Z-axis slices.
3. Expands each slice window according to local normal verticality.
4. Builds cluster-level 2D AlphaShape polygons.
5. Extracts source points whose projected coordinates fall inside the slice
   mask.

Install the package from the repository root first (`python -m pip install -e .`).
The command below assumes this module directory is the current directory.

## Inference

```bash
python infer_mask.py \
  --reference ../../outputs/example/03_repaired_sparse_reference.ply \
  --source ../../data/raw_scene.ply \
  --output ../../outputs/example/04_mask_extracted_dense.ply \
  --frame-output ../../outputs/example/02_extraction_frame.ply
```

`frame-output` is a colored LineSet PLY containing the slice-wise extraction
volume.
