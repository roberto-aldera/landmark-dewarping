import torch.nn.functional as func
import settings
import torch
import pytorch_lightning as pl
import pdb
from circular_motion_functions import CircularMotionEstimationBase
from custom_dataloader import LandmarksDataModule
from argparse import ArgumentParser


class CMNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.cme = CircularMotionEstimationBase()
        self.fc1 = torch.nn.Linear(in_features=settings.K_MAX_MATCHES * 4, out_features=settings.K_MAX_MATCHES * 6)
        self.fc2 = torch.nn.Linear(in_features=settings.K_MAX_MATCHES * 6, out_features=settings.K_MAX_MATCHES * 4)
        self.fc3 = torch.nn.Linear(in_features=settings.K_MAX_MATCHES * 4, out_features=settings.K_MAX_MATCHES * 2)

    def forward(self, x):
        b, n, c = x.shape
        predictions = self.fc1(x.float().flatten(1))
        predictions = func.relu(self.fc2(predictions))
        predictions = self.fc3(predictions).view(b, n, 2)

        # Apply predictions to latest landmark set (leaving previous landmarks unaltered)
        prediction_set = torch.zeros(b, n, c)
        prediction_set[:, :, 1] = predictions[:, :, 0]
        prediction_set[:, :, 3] = predictions[:, :, 1]

        vanilla_x = torch.tensor(x)
        # pdb.set_trace()
        mask = x != 0  # get mask to recall which elements where zero-padded
        prediction_set = prediction_set * mask  # ignore predicted corrections that were made on zero-padded entries
        x = x.add(prediction_set)

        # Quick check
        # pdb.set_trace()
        do_plot = False
        if do_plot:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(figsize=(10, 10))
            plt.grid()
            plt.plot(np.array(vanilla_x[0, :, 1].detach().numpy()), np.array(vanilla_x[0, :, 3].detach().numpy()), ',')
            plt.plot(np.array(x[0, :, 1].detach().numpy()), np.array(x[0, :, 3].detach().numpy()), ',')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("%s%s" % (settings.RESULTS_DIR, "landmarks-and-corrections.pdf"))
            plt.close()
            print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "landmarks-and-corrections.pdf"))
            pdb.set_trace()

        # Scale landmark positions back up to metres (after being between [-1, 1] for predictions)
        x = x * settings.MAX_LANDMARK_RANGE_METRES

        return self.cme(x)

    def training_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = func.mse_loss(self.forward(x).to(self.device), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = func.mse_loss(self.forward(x).to(self.device), y)
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
