import torchvision
import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from ddpm_config import TrainingConfig as TConfig
from ddpm_config import BaseConfig as BConfig

def get_dataset(dataset_name='BConfig.DATASET'):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32),
                      interpolation=TF.InterpolationMode.BICUBIC,
                      antialias=True),
            TF.RandomHorizontalFlip()
            TF.Lambda(lambda t: (t * 2) - 1)
        ]
    )

    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)

    return dataset

def inverse_transform(tensors):
    return ((tensors.clamp(-1,1) + 1.0) / 2.0) * 255.0

def get_dataloader(dataset_name=BConfig.DATASET,
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device=BConfig.DEVICE):
    dataset = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers
                            shuffle=shuffle
                            )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader



