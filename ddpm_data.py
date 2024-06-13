import torchvision
import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from ddpm_config import TrainingConfig as TConfig
from ddpm_config import BaseConfig as BConfig

def lam_func(t):
    return (t * 2) -1

def get_dataset(dataset_name='BConfig.DATASET'):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32),
                      interpolation=TF.InterpolationMode.BICUBIC,
                      antialias=True),
            TF.RandomHorizontalFlip(),
            TF.Lambda(lam_func)
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
                            num_workers=num_workers,
                            shuffle=shuffle
                            )

    return dataloader

#Visualize the data
# loader = get_dataloader(dataset_name=BConfig.DATASET,
#                         batch_size=128,
#                         device=BConfig.DEVICE)

# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,4), facecolor='grey')
# for b_image, _ in loader:
#     b_image = inverse_transform(b_image)
#     grid_img = make_grid(b_image / 255.0, nrow = 16, padding=True)
#     plt.imshow(grid_img.permute(1,2,0))
#     plt.axis("off")
#     break
# plt.show()


