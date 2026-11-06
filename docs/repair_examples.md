# Spatial continuity filling examples

The project provides an example checkpoint fitted on a small amount of data,
together with five incomplete point clouds, for quickly trying and testing the
spatial continuity filling module. No training or full reconstruction dataset
is needed to run this demonstration.

## Included assets

- Inputs: `data/examples/continuity_filling/*input*.ply` (five files, 8192 points
  each; approximately 1.06 MiB in total).
- Checkpoint: `checkpoints/examples/continuity_filling_demo.pth` (78.7 MiB,
  unchanged FP32 state dictionary, approximately 20.62 million parameters).
- Manifest: `data/examples/continuity_filling/manifest.json` records filenames,
  point counts, file sizes, and SHA-256 checksums.

The supplied clouds are already incomplete inputs for repair evaluation; the
demo does not apply another filtering or damage operation. Original filenames
and coordinates are preserved. Ground-truth and previous repaired clouds are
not bundled. The example checkpoint is supplied for demonstration and interface
testing; it is not the full benchmark model. The bundle is lightweight in setup
and data volume; the network has not been compressed, pruned, or quantized.

## Run the batch smoke test

Install the project dependencies in your Python environment (see the main
README). From the repository root:

```bash
python scripts/test_repair_examples.py
```

The script also works from any directory when invoked by its absolute path.
Its default data, checkpoint, and output paths are resolved from the script
location. An editable installation of this source checkout is not required for
this script, but the runtime dependencies must be installed.

CPU is supported; CUDA is selected automatically when available. To select a
device explicitly:

```bash
python scripts/test_repair_examples.py --device cpu
python scripts/test_repair_examples.py --device cuda
```

To test your own inputs and compatible checkpoint:

```bash
python scripts/test_repair_examples.py --input-dir path/to/examples --checkpoint path/to/model.pth --output-dir outputs/custom_repair
```

Custom input filenames must contain `input` and end in `.ply`. Inputs are
resampled to 8192 points for inference; all original input points are retained
in the repaired output. A compatible default-architecture checkpoint is
required (64 anchors, 256 feature channels, KNN k=16).

## Outputs and checks

For each input, `outputs/repair_examples/` receives:

- `<name>_patch.ply`: 1024 predicted repair points, colored red.
- `<name>_repaired.ply`: original input points in gray plus the predicted patch
  in red (9216 points for the bundled examples).
- `test_report.json`: one batch report with per-input checks and timings,
  runtime versions, checkpoint checksum, and pass/fail results.

The script strictly loads the checkpoint, verifies bundled asset checksums,
checks nonempty finite inputs, runs the existing `infer_patch` implementation,
checks the 1024-point output, and reads saved PLY files back to verify point
counts and coordinate preservation. A failed check returns a nonzero exit code.
The model is loaded once for the batch. A seeded NumPy generator controls input
sampling; bitwise equality across different devices or PyTorch versions is not
guaranteed.

This is an inference smoke test, not an accuracy benchmark. There are no
ground-truth targets in this bundle, so the report does not claim Chamfer
Distance, AP, or generalization performance. It runs only the continuity filling
module, not sparse proxy extraction, 3D clipping, or dense geometry fusion.

Inspect one output using the existing viewer:

```bash
python scripts/view_results.py --final "outputs/repair_examples/scan1 - Cloud_repaired.ply"
```

The five inputs, manifest, and example checkpoint are explicitly allowed by
`.gitignore` for source-repository distribution. Other local datasets, training
checkpoints, and generated outputs remain ignored.

## 中文说明

本项目提供基于少量拟合数据的连续性填充网络示例权重，以及 5 份残缺点云，
用于快速体验和测试修复补丁生成流程。安装依赖后运行
`python scripts/test_repair_examples.py` 即可批量生成修复补丁、拼接后的点云和
检查报告。每份输入 8192 点，预测补丁 1024 点，拼接结果 9216 点。

示例权重用于功能演示和接口测试，不作为论文完整基准模型。轻量示例包含少量
数据和单个推理权重；权重原文件约 78.7 MiB，未做压缩、剪枝或量化。输入点云
和权重均原样复制。
