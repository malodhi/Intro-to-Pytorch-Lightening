from dataloader import DataGenerator
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


class Inference(object):
    def __init__(self, model_path, device='cuda', num_samples=1, case=1):
        self.model_path = model_path
        self.device = device
        self.case = case
        self.num_samples = num_samples

    def load_model(self):
        self.model = torch.load(self.model_path, map_location=torch.device(self.device))
        self.model.eval()
        self.model.to(self.device)

    def measure_accu(self, img, output, threshold):
        pass

    @staticmethod
    def plot_results(output):
        img = output.detach().cpu().numpy()
        plt.imshow(img[0][0])
        plt.show()

    def test_loader(self):
        return DataLoader(DataGenerator(self.num_samples, case=self.case), batch_size=self.num_samples, num_workers=8)

    def test(self):
        self.load_model()
        for data in self.test_loader():
            data = data.to(self.device)
            output = self.model(data)
            self.plot_results(output)


if __name__=="__main__":
    infer = Inference('model_snapshots/snapshot_model_epoch_19.pt', num_samples=1, case=1)
    infer.test()