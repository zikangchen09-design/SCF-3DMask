# Command-line scripts

The algorithm implementations live in `src/scf3dmask/`. Scripts in this directory
are executable entry points and example runners, not additional copies of the
algorithms. Run commands from the repository root unless using absolute paths.

| Script | Purpose | Implementation called |
| --- | --- | --- |
| `run_pipeline.py` | Run all four recovery stages | `scf3dmask.pipeline.main` |
| `train_repair.py` | Train the continuity filling network | `scf3dmask.repair.train.main` |
| `infer_repair.py` | Predict a repair patch for one cloud | `scf3dmask.repair.inference.main` |
| `view_results.py` | Open a viewer for existing PLY results | `scf3dmask.visualization.main` |
| `test_repair_examples.py` | Batch-test the five repair examples and example weight | Existing model loading, `infer_patch`, and output writing |
| `test_mask_examples.py` | Test one 3D clipping example (default: basket), save a preview, and optionally open an interactive viewer | `generate_3d_binary_mask` and `scf3dmask.mask_visualization` |

The first four files are intentionally short wrappers that import `main()`.
Install the package first so that Python can resolve `scf3dmask`:

```bash
pip install -e ".[visualization]"
```

The two example runners also support execution directly from a source checkout
without editable installation, provided that all runtime dependencies are
installed. They resolve bundled assets relative to their own file locations.

```bash
python scripts/test_repair_examples.py
python scripts/test_mask_examples.py
python scripts/test_mask_examples.py --case basket --visualize
```

See `docs/repair_examples.md` and `docs/mask_examples.md` for inputs, outputs,
checks, and parameter details.

中文：前四个短脚本是命令行入口，实际功能位于 `src/scf3dmask`；两个示例测试
脚本负责读数据、调用已有实现、验证和保存结果。新增可视化函数只展示算法返回
的点云及线框，不重新计算裁剪算法。
