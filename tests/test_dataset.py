from pathlib import Path
from typing import List

from fsspec.implementations.local import LocalFileSystem
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

from pytorch_fsspec_dataset.dataset import ImageStorageDataset


class TestImageStorageDataset:
    def test_normal(self, tmp_path: Path):
        """代表的な画像チャンネルと拡張子でのテスト"""
        # 画像ファイルの作成
        input_images: List[Image.Image] = [
            Image.new(mode="RGB", size=(101, 201)),
            Image.new(mode="RGBA", size=(102, 202)),
            Image.new(mode="L", size=(103, 203)),
            Image.new(mode="RGB", size=(104, 204)),
            Image.new(mode="RGB", size=(105, 205)),
        ]
        input_paths = [
            tmp_path / "0.jpg",
            tmp_path / "1.png",
            tmp_path / "2.bmp",
            tmp_path / "3.tiff",
            tmp_path / "4.webp",
        ]
        for image, path in zip(input_images, input_paths):
            image.save(path)

        # テスト対象クラスのインスタンス生成
        dataset = ImageStorageDataset(
            file_system=LocalFileSystem(),
            root=str(tmp_path),
        )

        # データ数の一致
        assert len(dataset) == len(input_images)

        for i, (path, image) in enumerate(dataset):
            # パスの一致
            input_path = input_paths[i]
            assert path == input_path.as_posix()

            # データサイズの一致
            input_image = input_images[i]
            n_channels = len(input_image.getbands())
            assert image.shape == (n_channels, input_image.height, input_image.width)

            # 可逆圧縮画像のデータの一致
            if str(input_path).endswith((".png", ".bmp", ".tiff", ".webp")):
                assert to_pil_image(image) == input_image

    def test_empty(self, tmp_path: Path):
        """rootが空の場合"""
        dataset = ImageStorageDataset(
            file_system=LocalFileSystem(),
            root=str(tmp_path),
        )

        assert len(dataset) == 0

    def test_extensions(self, tmp_path: Path):
        """extensions引数のテスト"""
        # 画像ファイルの作成
        input_images: List[Image.Image] = [
            Image.new(mode="RGB", size=(101, 201)),
            Image.new(mode="RGB", size=(102, 202)),
            Image.new(mode="RGB", size=(103, 203)),
            Image.new(mode="RGB", size=(104, 204)),
            Image.new(mode="RGB", size=(105, 205)),
            Image.new(mode="RGB", size=(106, 206)),
        ]
        input_paths = [
            tmp_path / "0.jpg",
            tmp_path / "1.png",
            tmp_path / "2.bmp",
            tmp_path / "3.tiff",
            tmp_path / "4.webp",
            tmp_path / "5.jpg",
        ]
        for image, path in zip(input_images, input_paths):
            image.save(path)
        target_input_paths = [
            input_paths[0],
            input_paths[1],
            input_paths[5],
        ]

        # テスト対象クラスのインスタンス生成
        dataset = ImageStorageDataset(
            file_system=LocalFileSystem(),
            root=str(tmp_path),
            extensions=(".jpg", ".png"),
        )

        # データ数の、対象拡張子のファイル数との一致
        assert len(dataset) == len(target_input_paths)

        for (path, _), input_path in zip(dataset, target_input_paths):
            assert path == input_path.as_posix()

    def test_transform(self, tmp_path: Path):
        """transform引数のテスト"""
        input_images: List[Image.Image] = [
            Image.new(mode="RGB", size=(256, 256), color=0),
            Image.new(mode="RGB", size=(1920, 1080), color=255),
        ]
        input_paths = [
            tmp_path / "0.jpg",
            tmp_path / "1.jpg",
        ]
        for image, path in zip(input_images, input_paths):
            image.save(path)

        dataset = ImageStorageDataset(
            file_system=LocalFileSystem(),
            root=str(tmp_path),
            transform=nn.Sequential(
                T.Resize((224, 224)),
                T.ConvertImageDtype(float),
                T.Normalize(mean=(0, 0, 0), std=(255, 255, 255)),
            ),
        )

        assert len(dataset) == len(input_images)
        for _, image in dataset:
            # Resizeの確認
            assert image.shape == (3, 224, 224)

            # Normalizeの確認
            assert (image >= 0).all()
            assert (image <= 1).all()
