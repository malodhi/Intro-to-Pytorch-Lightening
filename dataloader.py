import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, num_samples=5, img_size=400, channels=1, case=2):
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.img_channels = channels
        self.case = case
        self.sample = self.generate_sample()

    def __getitem__(self, item):
        if item <= self.num_samples:
            return self.sample

    def __len__(self):
        return self.num_samples

    def generate_sample(self):
        img_height = self.img_size[0]
        if self.case == 1:
            sample = torch.zeros(self.img_channels+3, *self.img_size)
            # horizontal and vertical bar: 3/4 area covered.
            mask_start, mask_end_x = int(img_height * 0.062), int(img_height * 0.9375)
            mask_end_y = int(img_height / 2)
            for x in range(img_height):
                for y in range(img_height):
                    if x in range(mask_start, mask_end_x) and y in range(mask_start, mask_end_y):
                        sample[1, x, y] = 1
                        sample[2, x, y] = x
                        sample[3, x, y] = y
        if self.case == 2:
            sample = torch.zeros(self.img_channels, *self.img_size)
            mask_start, mask_end_x = int(img_height * 0.062), int(img_height / 2)
            mask_end_y = int(img_height / 2)
            mask = torch.ones(mask_end_y - mask_start, mask_end_x - mask_start)
            sample[:, mask_start:mask_end_x, mask_start:mask_end_y] = mask.T
            sample[:, mask_start:mask_end_y, mask_start:mask_end_x] = mask
        return sample

    def plot_sample(self):
        plt.imshow(self.sample[0].numpy())
        plt.show()


if __name__ == "__main__":
    dataloader = DataGenerator()
    dataloader.plot_sample()
