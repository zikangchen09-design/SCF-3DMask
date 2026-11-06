# From multi-view images to recovered object point clouds

SCF-3DMask starts from an existing reconstructed point cloud. To start from
photographs, first reproduce a Nerfstudio reconstruction, export the geometry,
and then run the SCF-3DMask pipeline.

```text
Images -> camera estimation and data processing -> Nerfstudio training
       -> RGB point_cloud.ply -> SCF-3DMask recovery -> final object cloud
```

## Reconstruction source and environment

The supplied upstream checkout is at
[`nerfstudio/nerfstudio-main/`](../nerfstudio/nerfstudio-main/README.md). Its
`pyproject.toml` declares version 1.1.5. Use its
[installation guide](../nerfstudio/nerfstudio-main/docs/quickstart/installation.md)
to set up CUDA, PyTorch, and its reconstruction dependencies. The image-data
workflow also needs the external tools described in the
[custom-data guide](../nerfstudio/nerfstudio-main/docs/quickstart/custom_dataset.md),
including COLMAP and FFmpeg.

Nerfstudio and SCF-3DMask can use separate environments. Their interface is the
exported PLY file. Commands below assume the working directory is the
SCF-3DMask repository root; switch environments at the indicated step.

In the prepared **Nerfstudio environment**:

```bash
pip install -e ./nerfstudio/nerfstudio-main
```

## Process and reconstruct your images

Put overlapping object photographs in `data/my_capture/images/`, then run:

```bash
ns-process-data images --data data/my_capture/images --output-dir data/my_capture/processed
ns-train nerfacto --data data/my_capture/processed
```

The first command estimates camera information and prepares Nerfstudio data.
The second trains the reconstruction. Keep the actual `config.yml` path printed
by the training command; the scene and timestamp directories depend on your run.

## Export a point cloud

Replace `path/to/config.yml` with that training configuration:

```bash
ns-export pointcloud --load-config path/to/config.yml --output-dir data/my_capture/reconstruction --normal-method open3d --num-points 1000000
```

The bundled `ExportPointCloud` implementation writes `point_cloud.ply` under
the output directory. `--normal-method open3d` estimates normals from the point
cloud, so this example does not depend on a model-predicted normal output.
The requested point count is a sampling setting; the exported cloud may contain
fewer points after upstream filtering.

See the bundled [geometry export guide](../nerfstudio/nerfstudio-main/docs/quickstart/export_geometry.md)
and `ns-export pointcloud --help` for exporter options. The command describes a
NeRF point-cloud export; Gaussian-splat parameter PLY files use another exporter
and are not interchangeable with ordinary XYZ/RGB geometry.

## Run SCF-3DMask

Switch to the **SCF-3DMask environment** and install its package:

```bash
pip install -e ".[visualization]"
python scripts/run_pipeline.py --input data/my_capture/reconstruction/point_cloud.ply --output-dir outputs/my_capture --ckpt checkpoints/examples/continuity_filling_demo.pth
```

This runs sparse proxy extraction, continuity filling, direct 3D clipping, and
dense fusion. The final output is `outputs/my_capture/05_final_object.ply`.
The bundled checkpoint is an example weight fitted on a small amount of data;
replace it with your trained repair checkpoint for experiments on new objects.
To try only the geometry stages, use `--skip-repair` in place of `--ckpt ...`.

Nerfstudio export normally uses its reconstructed coordinate frame. Keep the
raw scene and every derived reference/patch in the same frame. The clipping
module slices along Z, and several geometric filtering thresholds depend on
scale. Choose a suitable orientation and scale for your capture; adjust
`--voxel-size` when appropriate rather than assuming that all captures have the
same coordinate units.

## View the recovered result

```bash
python scripts/view_results.py --final outputs/my_capture/05_final_object.ply
```

The supplied PLY examples can be tested directly without performing any of the
Nerfstudio steps. See the [repair guide](repair_examples.md) and
[clipping guide](mask_examples.md).

These commands were checked against the bundled Nerfstudio documentation and
exporter implementation. A new Nerfstudio reconstruction has not been trained
as part of preparing this documentation.

## Attribution

[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) is a third-party
project, licensed under Apache-2.0 in the bundled checkout. Its source and
license are retained. Cite the Nerfstudio project when using its reconstruction
workflow, in addition to citing SCF-3DMask for object recovery.
