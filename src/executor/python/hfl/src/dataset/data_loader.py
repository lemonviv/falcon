from torch import tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        print("self.data shape:", self.data.shape)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, -1]
        # print(f"label: {label}")
        # print(f"type of label: {type(label)}")
        sample = self.data.iloc[idx, :-1].values
        # print(f"sample before transform: {sample}")
        # print(f"type of sample: {type(sample)}")
        sample_float = sample.astype(np.float32)
        # print(f"sample_float before transform: {sample_float}")
        # print(f"type of sample_float: {type(sample_float)}")
        if self.transform:
            sample_float = sample_float.reshape(1, -1)
            sample_float = self.transform(sample_float)
        return (sample_float, label)


class MinMaxScaleTransform:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def __call__(self, sample):
        # sample is assumed to be a NumPy array or a PyTorch tensor
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()  # Convert tensor to NumPy array
        scaled_sample = self.scaler.fit_transform(sample)
        return torch.from_numpy(scaled_sample)  # Convert NumPy array back to tensor


def get_dataset(data, data_dir, client_id):
    """
    Returns train and test datasets.
    """

    if data == 'bank':
        apply_transform = MinMaxScaleTransform()
        # apply_transform = transforms.Compose([
        #    transforms.ToTensor(), MinMaxScaleTransform()])
        train_dataset = CustomDataset(data_dir + "/bank_train_" + str(client_id) + ".csv",
                                      transform=apply_transform)
        test_dataset = CustomDataset(data_dir + "/bank_test_" + str(client_id) + ".csv",
                                     transform=apply_transform)

    elif data == 'mnist' or 'fmnist':
        # if data == 'mnist':
        #    data_dir = '../data/mnist/'
        # else:
        #    data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    return train_dataset, test_dataset

