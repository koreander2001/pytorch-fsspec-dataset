from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from fsspec import AbstractFileSystem
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms.functional import pil_to_tensor


class BaseStorageDataset(ABC, Dataset):
    def __init__(
        self,
        storage: AbstractFileSystem,
        root: str,
        extensions: Tuple[str, ...],
        transform: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.storage = storage
        self.device = device

        self.transform = (transform if transform else nn.Sequential()).to(self.device)

        all_paths: List[str] = storage.ls(root, detail=False)
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
        storage: AbstractFileSystem,
        root: str,
        extensions: Tuple[str, ...] = IMG_EXTENSIONS,
        transform: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__(
            storage=storage,
            root=root,
            extensions=extensions,
            transform=transform,
            device=device,
        )

    def read(self, path: str) -> Tensor:
        with self.storage.open(path) as f, Image.open(f) as pil_image:
            return pil_to_tensor(pil_image).to(self.device)
