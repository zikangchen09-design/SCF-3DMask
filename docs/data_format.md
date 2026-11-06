# Data Format

SCF-3DMask uses point clouds readable by Open3D.

Recommended input formats:

- `.ply`
- `.pcd`
- `.xyz`
- `.pts`

Training the repair network also supports `.off` meshes.

## Suggested Directory Layout

```text
data/
├── raw_scene.ply
└── repair_train/
    ├── object_001.ply
    ├── object_002.ply
    └── category_name/
        ├── train/
        │   └── sample.off
        └── test/
            └── sample.off
```

The dataset loader recursively searches for `.ply` and `.off` files. When a
directory follows a `train/` and `test/` split, the loader skips `test` during
training and skips `train` during validation.

## Data and reconstruction guides

- [Included point clouds and example weight](../data/README.md).
- [Public datasets cited by the paper](datasets.md).
- [Reconstruct multi-view images with Nerfstudio](nerfstudio_workflow.md).

The supplied public-dataset names and upstream links do not imply that their
full data is stored under this example directory.
