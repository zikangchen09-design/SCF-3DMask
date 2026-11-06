# Data and example assets

## Public datasets from the paper

See [the public dataset index](../docs/datasets.md) for CO3D-V2, BlendedMVS,
DTU MVS, and the Dense Light Field dataset of Yücer et al., with original source
links and the image-to-point-cloud workflow. Their full datasets are obtained
from their upstream providers.

## Included point-cloud examples

The source repository supplies five incomplete inputs for continuity filling
and eight point clouds for the 3D clipping demo. No reconstruction or training
is needed to try these examples. [manifest.json](manifest.json) records every
file's point count, size, and SHA-256 checksum.

| Point cloud | Points | MiB | Storage |
| --- | ---: | ---: | --- |
| [scan1 - Cloud_input.ply](examples/continuity_filling/scan1%20-%20Cloud_input.ply) | 8,192 | 0.21 | Git |
| [scan2-Cloud_input.ply](examples/continuity_filling/scan2-Cloud_input.ply) | 8,192 | 0.21 | Git |
| [scan3 - Cloud_input.ply](examples/continuity_filling/scan3%20-%20Cloud_input.ply) | 8,192 | 0.21 | Git |
| [scan4 - Cloud_input.ply](examples/continuity_filling/scan4%20-%20Cloud_input.ply) | 8,192 | 0.21 | Git |
| [scan5 - Cloud_input.ply](examples/continuity_filling/scan5%20-%20Cloud_input.ply) | 8,192 | 0.21 | Git |
| [basket.ply](examples/3D%20clipping%20mask/basket.ply) | 220,529 | 10.73 | Git |
| [long_deer_norm.ply](examples/3D%20clipping%20mask/long_deer_norm.ply) | 1,384,402 | 35.65 | Git |
| [scan2-Cloud.ply](examples/3D%20clipping%20mask/scan2-Cloud.ply) | 100,302 | 4.88 | Git |
| [scan3 - Cloud.ply](examples/3D%20clipping%20mask/scan3%20-%20Cloud.ply) | 154,044 | 7.49 | Git |
| [scan4 - Cloud.ply](examples/3D%20clipping%20mask/scan4%20-%20Cloud.ply) | 383,089 | 18.63 | Git |
| [scan5 - Cloud.ply](examples/3D%20clipping%20mask/scan5%20-%20Cloud.ply) | 88,610 | 4.31 | Git |
| [ship_norm.ply](examples/3D%20clipping%20mask/ship_norm.ply) | 6,599,227 | 169.93 | Git LFS |
| [statue_norm.ply](examples/3D%20clipping%20mask/statue_norm.ply) | 6,557,170 | 168.84 | Git LFS |

Run from the repository root:

```bash
python scripts/test_repair_examples.py
python scripts/test_mask_examples.py
```

The repair command checks the five inputs. The clipping command defaults to
one case, `basket.ply`, with up to 8192 sampled reference points. It saves a
preview without opening a window. Use `--case basket --visualize` to also open
an interactive viewer. These example inputs do not provide complete paired
benchmark ground truth or public evaluation splits.

## Example weight

The bundled [continuity_filling_demo.pth](../checkpoints/examples/continuity_filling_demo.pth)
is an example checkpoint fitted on a small amount of data. Its size is
78.7 MiB. See [the repair example guide](../docs/repair_examples.md) for the
model interface and output details.

## Downloading the source examples

Two large example clouds and the example checkpoint are configured in the root
`.gitattributes` for Git LFS: `ship_norm.ply`, `statue_norm.ply`, and
`continuity_filling_demo.pth`. With Git LFS installed, clone the
published repository and run these commands inside that clone:

```bash
git lfs install
git lfs pull
```

Retrieve the real PLY objects rather than using LFS pointer text as geometry.
Other example clouds use ordinary Git.

See [data formats](../docs/data_format.md), [repair examples](../docs/repair_examples.md),
and [clipping examples](../docs/mask_examples.md) for further details.
