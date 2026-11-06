# 3D clipping mask examples

Test the project's existing clipping-mask algorithm with the `.ply` files in
`data/examples/3D clipping mask/`:

```bash
pip install -e ".[visualization]"
python scripts/test_mask_examples.py
```

The default command runs **one example, `basket.ply`**. Use `--case` to select a
different file. Testing every file requires the explicit `--all` option.

The script calls `scf3dmask.mask3d.generate_3d_binary_mask()` directly. It does
not copy or replace DBSCAN, AlphaShape, slicing, or point extraction. The
interactive view follows the white background, colored slice wireframes, and
extracted-cloud overlay used by the supplied visualization reference. Both the
preview and interactive view consume the frame returned by the core algorithm.

## Input interpretation

The supplied directory contains eight individual object point clouds, rather
than pairs of noisy sources and sparse/repaired references. For the default
demonstration, the script deterministically samples up to 8192 points from each
cloud as the reference and clips the original full cloud. It does not add noise,
apply the filling network, or modify the supplied files. This demonstrates the
sparse-to-dense clipping interface; retention is not a denoising accuracy score.

The included filenames are `basket.ply`, `long_deer_norm.ply`, `scan2-Cloud.ply`,
`scan3 - Cloud.ply`, `scan4 - Cloud.ply`, `scan5 - Cloud.ply`, `ship_norm.ply`,
and `statue_norm.ply`. The two largest clouds have more than six million points
each, so full batch processing and exported clouds can be substantial.

For a genuine sparse/repaired reference and a noisy original source, supply
both explicitly:

```bash
python scripts/test_mask_examples.py --reference path/to/repaired_sparse.ply --source path/to/noisy_scene.ply
```

Both point clouds must share a coordinate system. The core algorithm slices
along Z, using the normal-guided adaptive window already implemented in
`src/scf3dmask/mask3d.py`. Defaults are 70 slices, EMA 0.8, and Alpha multiplier
6.0. The algorithm module is unchanged by these example runners.

## Visualization

A four-panel `preview.png` is saved automatically for every case: original
source, sparse reference, colored mask over extraction, and extraction alone.
The panels share an equal-scale camera view. Original cloud colors are kept,
the reference is blue, and the mask has the existing height-dependent colors.
Only preview drawing is limited to 30000 points and 25000 frame lines; clipping
and exported outputs use the full source and full generated mask.

To run one case and open an interactive Open3D window:

```bash
python scripts/test_mask_examples.py --case basket --visualize
```

The window uses a white background, point size 4, and the full generated colored
wireframe. Rotate and zoom with the mouse; close the window to continue.
The default command processes only `basket` and saves a preview without opening
windows. With `--all --visualize`, cases are shown in sequence; this is optional
and can take substantially longer.

Other examples:

```bash
python scripts/test_mask_examples.py --case "scan3 - Cloud"
python scripts/test_mask_examples.py --case basket --num-layers 30
python scripts/test_mask_examples.py --reference-points 4096 --max-display-points 15000
```

## Outputs and checks

Each `outputs/mask_examples/<case>/` contains `reference.ply`, `extracted.ply`,
`mask_frame.ply`, `slices.json`, and `preview.png`. The batch report is
`outputs/mask_examples/test_report.json` and is updated after each case.

Checks cover finite coordinates, a nonempty extraction and mask, saved point
coordinates, source color preservation, and wireframe line counts. Per-case
reports contain input/reference/extracted counts, retained fractions, valid
slice counts, and algorithm time. Failed cases are reported and cause a
nonzero final exit code. No ground-truth CD or AP is computed.

## 中文说明

脚本直接调用项目已有的 `generate_3d_binary_mask()`，没有重新实现裁剪算法。
默认只测试 `basket.ply`，从原始点云采样最多 8192 点作为稀疏参考，再裁剪
完整原始点云；每份
结果自动保存四视图预览。添加 `--case basket --visualize` 可打开白底、彩色
切片线框与提取结果叠加的 Open3D 交互窗口。

目录中的数据没有提供“含噪原始点云—稀疏参考”配对，因此默认测试用于展示
算法接口和稀疏到稠密的裁剪流程，不将保留比例解释为去噪精度。
