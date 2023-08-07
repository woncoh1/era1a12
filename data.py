import torch
import torchvision
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Pixel statistics of all (train + test) CIFAR-10 images
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
AVG = (0.4914, 0.4822, 0.4465) # Mean
STD = (0.2023, 0.1994, 0.2010) # Standard deviation
# Image dimensions of CIFAR-10
CHW = (3, 32, 32) # Channel, height, width
# List of all classes in CIFAR-10
CLASSES = [ # Class labels (list index = class value)
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]
BATCH_SIZE = 512
EPOCHS = 24


def get_transforms(
    padding:int=40, # size after padding before cropping (unit: pixels)
    crop:int=32, # size after cropping (unit: pixels)
    cutout:int=8, # size of cutout box (unit: pixels)
) -> dict[str, A.Compose]:
    """Create image transformation pipeline for training and test datasets.
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L30-L34
    https://github.com/davidcpage/cifar10-fast/blob/master/core.py#L98
    """
    return {
        'train': A.Compose([
            A.Normalize(mean=AVG, std=STD, always_apply=True), # Cutout boxes should be grey, not black
            A.PadIfNeeded(min_height=padding, min_width=padding, always_apply=True), # Pad before cropping to achieve translation
            A.RandomCrop(height=crop, width=crop, always_apply=True),
            A.HorizontalFlip(),
            A.CoarseDropout( # Cutout
                max_holes=1, max_height=cutout, max_width=cutout,
                min_holes=1, min_height=cutout, min_width=cutout,
                fill_value=AVG
            ),
            ToTensorV2(),
        ]),
        'test': A.Compose([
            A.Normalize(mean=AVG, std=STD, always_apply=True),
            ToTensorV2(),
        ]),
    }


class CIFAR10DataSet(torch.utils.data.Dataset):
    """Pytorch dataset + custom data augmentation (Albumentations image transformation)
    https://github.com/parrotletml/era_session_seven/blob/main/mnist/dataset.py
    https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/
    """
    def __init__(self,
        dataset:torchvision.datasets,
        transform:A.Compose|None=None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        if self.transform: image = self.transform(image=np.array(image))['image']
        return image, label


class CIFAR10DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CIFAR-10
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """
    def __init__(self,
        batch_size:int=128,
        num_workers:int=0,
        shuffle:bool=True,
        pin_memory:bool=True,
    ) -> None:
        super().__init__()
        self.data_dir = './'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.transform = get_transforms()

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.cifar_train = CIFAR10DataSet(
                torchvision.datasets.CIFAR10(self.data_dir, train=True),
                transform=self.transform['train'],
            )
            self.cifar_valid = CIFAR10DataSet(
                torchvision.datasets.CIFAR10(self.data_dir, train=False),
                transform=self.transform['test'],
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )