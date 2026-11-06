# Usage and training

Run commands from the repository root after installing the package with
`python -m pip install -e ".[visualization]"`.

## Full pipeline

Use an XYZ/RGB reconstruction and a compatible repair checkpoint:

```bash
python scripts/run_pipeline.py --input path/to/scene.ply --output-dir outputs/example --ckpt path/to/repair_model.pth
```

The included `checkpoints/examples/continuity_filling_demo.pth` can be used to
try the interface. It is fitted on a small amount of data. Use a trained repair
checkpoint for experiments on new objects.

Without the repair stage:

```bash
python scripts/run_pipeline.py --input path/to/scene.ply --output-dir outputs/example_no_repair --skip-repair
```

`--voxel-size` defaults to 0.01 and `--num-layers` defaults to 70. Geometric
thresholds depend on scale, and the mask slices along Z. Keep raw geometry,
references, and patches in one coordinate frame.

## Outputs

| File | Meaning |
| --- | --- |
| `01_sparse_proxy.ply` | Extracted sparse proxy |
| `02_extraction_frame.ply` | Colored Open3D LineSet for valid mask slices |
| `03_repair_patch.ply` | Predicted patch, when repair is enabled |
| `03_repaired_sparse_reference.ply` | Proxy plus patch, when repair is enabled |
| `04_mask_extracted_dense.ply` | Original dense points accepted by the mask |
| `05_final_object.ply` | Fused recovered object |
| `metadata.json` | Point counts and selected filtering branch |

View the final result:

```bash
python scripts/view_results.py --final outputs/example/05_final_object.ply
```

Or overlay the intermediate geometry:

```bash
python scripts/view_results.py --input path/to/scene.ply --sparse-proxy outputs/example/01_sparse_proxy.ply --frame outputs/example/02_extraction_frame.ply --dense outputs/example/04_mask_extracted_dense.ply --final outputs/example/05_final_object.ply
```

The frame file contains lines; use the supplied viewer or an Open3D LineSet reader.

## Train continuity filling

Supply clean `.ply` point clouds or `.off` meshes:

```bash
python scripts/train_repair.py --data-dir data/repair_train --output-dir checkpoints --epochs 200 --batch-size 4 --amp
```

The loader constructs self-supervised incomplete inputs and removed-point targets.
See [data formats](data_format.md) for training directories and split handling.
Provide separate `train/` and `test/` object directories for held-out validation.
An unsplit directory is reused by both loaders and does not provide an
independent validation set.
GPU memory usage depends on batch size; install an appropriate PyTorch CUDA build
before training on a GPU.

## Repair inference only

```bash
python scripts/infer_repair.py --input outputs/example/01_sparse_proxy.ply --ckpt checkpoints/best_model.pth --output outputs/example/repair_vis.ply --patch-output outputs/example/03_repair_patch.ply
```

The colored output combines gray input points with a red patch. For bundled
inputs, use the [repair example runner](repair_examples.md).

## 3D mask only

```bash
python algorithms/three_d_binary_mask/infer_mask.py --reference outputs/example/03_repaired_sparse_reference.ply --source path/to/scene.ply --output outputs/example/04_mask_extracted_dense.ply --frame-output outputs/example/02_extraction_frame.ply
```

Without repair, provide `01_sparse_proxy.ply` as the reference. Use the full
original reconstruction as the source to retain dense geometry and colors.
For supplied object examples, see the [clipping example guide](mask_examples.md).
