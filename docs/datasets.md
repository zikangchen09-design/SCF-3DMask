# Datasets used in the paper

The paper evaluates SCF-3DMask on four public datasets covering different
reconstruction conditions. The links below point to the upstream projects or
the original publication, rather than copies of their complete data inside this
repository.

| Dataset | Reference | Upstream source | Use in SCF-3DMask |
| --- | --- | --- | --- |
| CO3D-V2 / Common Objects in 3D | Reizenstein et al., ICCV 2021 | [Official CO3D repository and download instructions](https://github.com/facebookresearch/co3d) | Object reconstruction and recovery across real-world captures |
| BlendedMVS | Yao et al., CVPR 2020 | [Official dataset repository and download links](https://github.com/YoYo000/BlendedMVS) | Recovery under diverse multi-view reconstruction conditions |
| DTU MVS | Jensen et al., CVPR 2014 | [DTU benchmark and dataset page](https://roboimagedata.compute.dtu.dk/?page_id=36) | Controlled multi-view reconstruction and geometric evaluation |
| Dense Light Field dataset of Yücer et al. | *Efficient 3D Object Segmentation from Densely Sampled Light Fields with Applications to 3D Reconstruction*, ACM TOG 2016 | [Original publication and associated resources](https://doi.org/10.1145/2876504) | Recovery from densely sampled viewpoints |

For CO3D, follow the current upstream instructions for the V2 release. The light
field entry identifies the exact paper cited by the manuscript; it is a
publication/resource landing page, not a direct archive download URL. Consult
each upstream source for its available data, release version, and use terms.

## Using the public data

SCF-3DMask consumes XYZ/RGB point clouds. Starting with the upstream images,
reconstruct a scene using the [Nerfstudio workflow](nerfstudio_workflow.md) or
another reconstruction pipeline and export a geometry PLY. Then run:

```bash
python scripts/run_pipeline.py --input path/to/reconstructed_scene.ply --output-dir outputs/public_dataset_scene --ckpt path/to/repair_model.pth
```

Use a trained repair weight for quantitative experiments. The bundled
small-data-fitted weight is intended for the [quick repair demo](repair_examples.md).
The paper describes automatically constructing repair targets from DTU and
CO3D-V2 point clouds; the current training loader accepts `.ply` and `.off` files.

## Examples already included in this repository

The [data index](../data/README.md) lists the five incomplete repair inputs,
eight clipping-example point clouds, and example checkpoint. These assets let
users try the core modules immediately. Their file inventory does not describe
the complete public datasets or paper evaluation splits.

The manuscript also discusses SCF-Synth, its separately constructed synthetic
benchmark. The four entries above are the external public datasets.
