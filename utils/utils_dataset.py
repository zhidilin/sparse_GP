import pandas as pd
from scipy.io import loadmat
import torch
from .setup import dtype
from .setup import reset_seed

# data normalization, define a class which can normalize the data and de-normalize the data
class DataNormalizer:
    def __init__(self, data):
        """
        Initialize the DataNormalizer with the input data.

        Args:
            data (torch.Tensor): Input data tensor.
        """
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def normalize(self, data):
        """
        Normalize the input data using the stored mean and std.

        Args:
            data (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Normalized data tensor.
        """
        return (data - self.mean) / self.std

    def denormalize(self, normalized_data):
        """
        Denormalize the input normalized data using the stored mean and std.

        Args:
            normalized_data (torch.Tensor): Normalized data tensor.

        Returns:
            torch.Tensor: Denormalized data tensor.
        """
        return normalized_data * self.std + self.mean



def load_Snelson(file_path):
    """
    Load dataset from a .mat file and preprocess it.

    Args:
        file_path (str): Path to the .mat file containing the dataset.

    Returns:
        tuple: Preprocessed training and testing data as tensors.
    """
    # Load the data from the .mat file
    data = loadmat(file_path)

    train_x = torch.tensor(data['Xtrain'], dtype=dtype)
    train_y = torch.tensor(data['Ytrain'], dtype=dtype)
    test_x = torch.tensor(data['Xtest'], dtype=dtype)

    # randomly choose 40 points from the training set
    if train_x.size(0) > 40:
        indices = torch.randperm(train_x.size(0))[:40]
        train_x = train_x[indices]
        train_y = train_y[indices].squeeze()
    else:
        print("Warning: Training set has less than 40 points, using all available points.")
        train_x = train_x.contiguous()
        train_y = train_y.contiguous().squeeze()

    test_x = test_x.contiguous()
    return train_x, train_y, test_x


