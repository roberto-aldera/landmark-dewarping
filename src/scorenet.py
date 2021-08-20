import torch.nn.functional as func
from torch import nn
import settings
import torch
import pytorch_lightning as pl
import pdb
from circular_motion_functions import CircularMotionEstimationBase
from loss_functions import LossFunctionFinalPoseVsCmeGt
from argparse import ArgumentParser


class ScoreNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.cme = CircularMotionEstimationBase()
        self.fc1 = nn.Linear(in_features=settings.K_MAX_MATCHES * 1, out_features=settings.K_MAX_MATCHES * 6)
        self.fc2 = nn.Linear(in_features=settings.K_MAX_MATCHES * 6, out_features=settings.K_MAX_MATCHES * 3)
        self.fc3 = nn.Linear(in_features=settings.K_MAX_MATCHES * 3, out_features=settings.K_MAX_MATCHES * 1)
        self.dropout = nn.Dropout(0.5)
        self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, self.dropout, self.fc3, nn.Softmax(dim=-1))

        self.cme = CircularMotionEstimationBase()
        self.loss = LossFunctionFinalPoseVsCmeGt(self.device)

    def forward(self, x):
        # Get CME parameters here so network feeds directly on CME data (possibly just thetas for now)
        x_cme_parameters = self.cme(x)
        x_thetas = x_cme_parameters[:, :, 0].to(self.device).type(torch.FloatTensor)
        scores = self.net(x_thetas.to(self.device))

        # Get poses, and then weight each match by the score
        thetas = x_cme_parameters[:, :, 0].type(torch.FloatTensor)
        curvatures = x_cme_parameters[:, :, 1].type(torch.FloatTensor)
        radii = 1 / curvatures.type(torch.FloatTensor)

        phi = thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[radii == float('inf')] = 0
        d_y[radii == float('inf')] = 0

        # Weight all dx, dy, dth by the (normalised) scores, and sum to get final pose
        d_x = d_x.to(self.device) * scores
        d_y = d_y.to(self.device) * scores
        d_th = thetas.to(self.device) * scores

        final_pose = torch.cat((torch.sum(d_x, dim=1).unsqueeze(1), torch.sum(d_y, dim=1).unsqueeze(1),
                                torch.sum(d_th, dim=1).unsqueeze(1)), dim=1)

        # Do some plotting
        sorted_thetas, sorted_indices = torch.sort(x_cme_parameters[:, :, 0])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].grid()
        ax[0].plot(scores.detach().numpy()[0], '.', label="Scores")
        ax[0].plot(torch.sort(scores)[0].detach().numpy()[0], '.', label="Sorted scores by magnitude")
        ax[0].plot(sorted_indices[0], scores.detach().numpy()[0], '.', label="Sorted scores by sorted theta indices")
        ax[0].legend()

        ax[1].grid()
        # ax[1].plot(sorted_thetas.detach().numpy()[0], '.')
        ax[1].plot(scores.detach().numpy()[0], '.', label="Scores")
        ax[1].plot(x_cme_parameters[:, :, 0].detach().numpy()[0], '.', label="Thetas")
        ax[1].set_ylim(-0.01, 0.01)
        # ax[1].set_xlim(0, 5)
        ax[1].legend()

        fig.suptitle("Debugging scores and their corresponding thetas")
        fig.savefig("%s%s" % (settings.RESULTS_DIR, "debugging.pdf"))
        plt.close()
        print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "debugging.pdf"))
        pdb.set_trace()

        return final_pose

    def training_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = self.loss(self.forward(x), y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        loss = self.loss(self.forward(x), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.default_root_dir = settings.MODEL_DIR

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=settings.LEARNING_RATE, type=float)
        parser.add_argument('--batch_size', default=settings.BATCH_SIZE, type=int)

        # training specific (for this model)
        parser.add_argument('--max_num_epochs', default=settings.MAX_EPOCHS, type=int)

        return parser


if __name__ == "__main__":
    print("Running...")
    x = torch.randn(2, settings.K_MAX_MATCHES, 4)  # batches, number of pairs, 4.
    hparams = {}
    net = ScoreNet(hparams)
    y = net.forward(x)[0]
