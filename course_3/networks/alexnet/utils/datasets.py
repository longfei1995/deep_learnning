"""
数据集加载
"""

import torch.utils.data
import torchvision.transforms as transforms

import torchvision.datasets as datasets


def load_datasets(name, root, batch_size):
    if name == "mnist":
        train_dataset = datasets.MNIST(
            root=root,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_dataset = datasets.MNIST(
            root=root,
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, test_dataloader

    elif name == "fmnist":
        train_dataset = datasets.FashionMNIST(
            root=root,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_dataset = datasets.FashionMNIST(
            root=root,
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, test_dataloader

    elif name == "kmnist":
        train_dataset = datasets.KMNIST(
            root=root,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_dataset = datasets.KMNIST(
            root=root,
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, test_dataloader

    elif name == "qmnist":
        train_dataset = datasets.QMNIST(
            root=root,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_dataset = datasets.QMNIST(
            root=root,
            download=True,
            what="test50k",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, test_dataloader

    elif name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=root,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_dataset = datasets.CIFAR10(
            root=root,
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, test_dataloader

    elif name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=root,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_dataset = datasets.CIFAR100(
            root=root,
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_dataloader, test_dataloader
    elif name == "food101":
        train_dataset = datasets.Food101(
            root=root,
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        test_dataset = datasets.Food101(
            root=root,
            split="test",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return train_dataloader, test_dataloader
