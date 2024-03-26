# Containg the scripts nescessary to geerate the 2d datasets for benchmark tests
import numpy as np
import torch
from torch.utils.data import Dataset

# A reminder this code was tajen from: (https://github.com/jraine/flows4flows/blob/main/ffflows/data/plane.py)
class PlaneDataset(Dataset):
    def __init__(self, num_points, flip_axes=False):
        self.num_points = num_points
        self.flip_axes = flip_axes
        self.data = None
        self.reset()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def reset(self):
        self._create_data()
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()

    def _create_data(self):
        raise NotImplementedError

class CheckerboardDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        self.data = torch.stack([x1, x2]).t() * 2

class FourCircles(PlaneDataset):
    def __init__(self, num_points, flip_axes=False):
        if num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        super().__init__(num_points, flip_axes)

    @staticmethod
    def create_circle(num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        self.data = torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )