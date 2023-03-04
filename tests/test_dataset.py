from pathlib import Path
from typing import List

from fsspec.implementations.local import LocalFileSystem
from PIL import Image
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

    # TODO: extensionsのテスト

    # TODO: transformのテスト
