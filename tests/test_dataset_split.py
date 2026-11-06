"""Ensure object subdirectories cannot bypass the held-out split boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scf3dmask.repair.dataset import PointCloudRepairDataset


class DatasetSplitTests(unittest.TestCase):
    def test_nested_objects_stay_in_their_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ["category/train/a.ply", "category/train/nested/b.off",
                             "category/test/c.ply", "category/test/nested/d.off"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            # Collection is independent of geometry decoding/preloading.
            with patch.object(PointCloudRepairDataset, "_preload"):
                train = PointCloudRepairDataset(root, split="train")
                val = PointCloudRepairDataset(root, split="val")
            self.assertEqual({p.relative_to(root).as_posix() for p, _ in train.file_paths},
                             {"category/train/a.ply", "category/train/nested/b.off"})
            self.assertEqual({p.relative_to(root).as_posix() for p, _ in val.file_paths},
                             {"category/test/c.ply", "category/test/nested/d.off"})


if __name__ == "__main__":
    unittest.main()
