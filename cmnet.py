import torch.nn.functional as func
import settings
import torch
import pytorch_lightning as pl
import pdb
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from custom_dataloader import LandmarkDataset, CollateFn, ToTensor, Normalise, ZeroPadding
from circular_motion_functions import CircularMotionEstimationBase
from custom_dataloader import LandmarksDataModule
from argparse import ArgumentParser


class CMNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.cme = CircularMotionEstimationBase()
        self.fc1 = torch.nn.Linear(in_features=settings.K_MAX_MATCHES * 4, out_features=settings.K_MAX_MATCHES * 2)
        # self.loss = None  # probably an L1 but do this later

    def forward(self, x):
        b, n, c = x.shape
        predictions = self.fc1(x.float().flatten(1)).view(b, n, 2)  # only predicting corrections on x2 and y2 positions
        padded_predictions = func.pad(predictions, pad=(0, 2, 0, 0))
        x = x + padded_predictions
        # Scale landmark positions back up to metres (after being between [-1, -1] for predictions)
        x = x * settings.MAX_LANDMARK_RANGE_METRES

        return self.cme(x)

    def training_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = func.mse_loss(self.forward(x), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = func.mse_loss(self.forward(x), y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parent_parser.default_root_dir = settings.MODEL_DIR

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=settings.LEARNING_RATE, type=float)
        parser.add_argument('--batch_size', default=settings.BATCH_SIZE, type=int)
        parser.add_argument('--dropout', default=0, type=float)

        # training specific (for this model)
        parser.add_argument('--max_num_epochs', default=settings.MAX_EPOCHS, type=int)

        return parser


if __name__ == "__main__":
    print("Running :)")
    # input = torch.randn(2, 10, 4)  # batches, number of pairs, 4.
    dm = LandmarksDataModule()
    model = CMNet({})
    trainer = pl.Trainer()
    trainer.fit(model, dm)
    model(input)
