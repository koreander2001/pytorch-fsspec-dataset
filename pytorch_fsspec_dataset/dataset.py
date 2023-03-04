from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Sequence, Tuple

from fsspec import AbstractFileSystem
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms.functional import pil_to_tensor


class BaseStorageDataset(ABC, Dataset, Sequence):
    def __init__(
        self,
        file_system: AbstractFileSystem,
        root: str,
        extensions: Tuple[str, ...],
        transform: nn.Module = nn.Identity(),
        device: str = "cpu",
    ):
        self.file_system = file_system
        self.device = device
        self.transform = transform.to(self.device)

        all_paths: List[str] = self.file_system.ls(root, detail=False)
        self.paths = [p for p in all_paths if p.lower().endswith(extensions)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Tuple[str, Tensor]:
        path = self.paths[index]

        raw_data = self.read(path).to(self.device)
        data = self.transform(raw_data)

        return path, data

    @abstractmethod
    def read(self, path: str) -> Tensor:
        pass


class ImageStorageDataset(BaseStorageDataset):
    def __init__(
        self,
        file_system: AbstractFileSystem,
        root: str,
        extensions: Tuple[str, ...] = IMG_EXTENSIONS,
        transform: nn.Module = nn.Identity(),
        device: str = "cpu",
    ):
        super().__init__(file_system, root, extensions, transform, device)

    def read(self, path: str) -> Tensor:
        with TemporaryDirectory() as temp_dir:
            local_path = str(Path(temp_dir) / Path(path).name)
            self.file_system.get_file(path, local_path)
            return pil_to_tensor(Image.open(local_path))
