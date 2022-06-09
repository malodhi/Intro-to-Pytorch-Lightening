import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import UNetMini
from dataloader import DataGenerator


Path('Summary').mkdir(exist_ok=True, parents=True)
writer = SummaryWriter('Summary/')


class Trainer(object):
    def __init__(self, model_path: str, device='cuda', lr=0.0001, batch_size=2,
                 num_samples=2, epochs=20, case=1):
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.case = case
        self.num_samples = num_samples
        self.model_path = Path(model_path)
        self.setup()

    def setup(self):
        self.set_model()
        self.set_optimizer()
        self.set_criterion()

    def set_model(self):
        self.model = UNetMini().to(self.device)#.double().

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def set_criterion(self):
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()

    @staticmethod
    def check_patience(losses, patience=3, min_loss=None):
        losses_len = len(losses)
        if losses_len == 1:
            return True
        min_loss = min(losses[0: -1]) > losses[-1]
        losses = losses[-patience:]
        decreasing = all(y < x for x, y in zip(losses, losses[1:]))
        if decreasing or min_loss:
            return True
        return False

    def train_loader(self):
        dataset = DataGenerator(self.num_samples, case=self.case)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=8)

    def save_model(self, path):
        path = Path(path)
        if path.suffixes:
            dir_path = path.parent
        else:
            dir_path = path
        dir_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model, path)

    def train(self):
        validation_loss = []
        for epoch in range(self.epochs):
            train_loss = list()
            self.model.train(True)
            for i, data in enumerate(self.train_loader()):
                self.optimizer.zero_grad()
                img = data.to(self.device)
                output = self.model(img)
                loss = self.criterion(output, img)
                loss.backward()
                self.optimizer.step()
                z = loss.detach().cpu().numpy()
                train_loss.append(z)
                # print("Train Step Loss :  ", z)
            train_loss = np.mean(train_loss)
            val_loss = list()
            self.model.train(False)
            with torch.no_grad():
                for i, data in enumerate(self.train_loader()):
                    img = data.to(self.device)
                    output = self.model(img)
                    loss = self.criterion(output, img)
                    z = loss.detach().cpu().numpy()
                    val_loss.append(z)
                    # print("Val Step Loss :  ", z)

            val_loss = np.mean(val_loss)
            validation_loss.append(val_loss)
            print(f"Epoch:  {epoch},  train/loss={train_loss},  valid/loss={val_loss}")

            if self.check_patience(validation_loss):
                path = "snapshot_model_epoch_" + str(epoch) + ".pt"
                Path(self.model_path).mkdir(exist_ok=True, parents=True)
                model_path = Path(self.model_path) / path
                print("Saving model  ===>>>   ", model_path.as_posix())
                self.save_model(model_path)

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.flush()

        writer.close()


if __name__ == "__main__":
    trainer = Trainer('model_snapshots/')
    trainer.train()
