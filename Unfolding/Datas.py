import torch
import torch.nn.functional
import torch.utils.data
import torchvision

import os
import pathlib

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: pathlib.Path,  config: dict = {}) -> None:
        """
        Args:
            root_dir (pathlib.Path): Directory with all the images.
        """
        super(ImageDataset, self).__init__()
        self.data_path: pathlib.Path = data_path
        self.image_names: list[pathlib.Path] = [ 
            filename.stem for filename in map(lambda e : pathlib.Path(e), os.listdir(self.data_path / 'Artifacts'))
        ]

        self.items: list[tuple[torch.Tensor, torch.Tensor]] = [] 

        device = config.get('device', 'cpu')
        size_pool = config.get('max_pool2d', (2, 2))

        for index in range(0, len(self.image_names)):

            filename_artifact: pathlib.Path = \
                self.data_path / 'Artifacts' / (self.image_names[index] + '.jpg')

            image_artifact_string: torch.Tensor = \
                torchvision.io.read_file(str(filename_artifact))

            image_artifact_decoded: torch.Tensor = \
                torchvision.io.decode_jpeg(
                    input=image_artifact_string, 
                    mode=torchvision.io.ImageReadMode.GRAY
                ) / 255.0

            filename_result: pathlib.Path = \
                self.data_path / 'Results' / (self.image_names[index] + '.png')

            image_result_string: torch.Tensor = \
                torchvision.io.read_file(str(filename_result))

            image_result_decoded: torch.Tensor = \
                torchvision.io.decode_png(
                    input=image_result_string, 
                    mode=torchvision.io.ImageReadMode.GRAY
                ) / 255.0

            image_artifact_decoded = torch.nn.functional.max_pool2d(image_artifact_decoded, size_pool)
            image_result_decoded = torch.nn.functional.max_pool2d(image_result_decoded, size_pool)

            self.items.append((image_artifact_decoded.to(device=device), image_result_decoded.to(device=device)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image_artifact_decoded, image_result_decoded = self.items[index]
        return image_artifact_decoded, image_result_decoded

    def to(self, device: str = 'cpu') -> None:
        for i in range(self.items):
            artifact, result = self.items[i]
            self.items[i] = artifact.to(device), result.to(device)


def split_dataset(dataset: ImageDataset, train_size: float) -> tuple[ImageDataset, ImageDataset]:
    n = len(dataset)
    train_size = int(0.8*n)
    test_size = n-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_dataloaders(config: dict[str, str|int]) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset_full = ImageDataset(
        data_path=pathlib.Path(config['dataset_path']),
        config=config
    )
    train_dataset, test_dataset = split_dataset(dataset=dataset_full, train_size=config.get('train_size', 0.8))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 32),
        shuffle=config.get('shuffle', False)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.get('batch_size', 32))
    return train_loader, test_loader