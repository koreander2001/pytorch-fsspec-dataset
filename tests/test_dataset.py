from pathlib import Path

from fsspec.implementations.local import LocalFileSystem

from pytorch_fsspec_dataset.dataset import ImageStorageDataset


class TestImageStorageDataset:
    def test_normal(self, tmp_path: Path):
        dataset = ImageStorageDataset(
            storage=LocalFileSystem(),
            root=str(tmp_path),
        )
        assert len(dataset) == 0
