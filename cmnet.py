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
        self.fc1 = torch.nn.Linear(in_features=1200 * 4, out_features=1200 * 4)
        # self.loss = None  # probably an L1 but do this later

    def forward(self, x):
        # pdb.set_trace()
        b, n, c = x.shape
        x = x + self.fc1(x.float().flatten(1)).view(b, n, c)
        # Still need to account for the correction additions here. The network predicts corrections,
        # which are then added on to the original inputs (but on only 2 of the 4)
        return self.cme(x)

    def training_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = func.mse_loss(self.forward(x), y)
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
