import torch.nn.functional as func
import settings
import torch
import pytorch_lightning as pl
import pdb

from argparse import ArgumentParser


class MyPlaceholderNetwork(torch.nn.Module):
    def forward(self, x):
        return 0 * x


class CircularMotionEstimationBase(torch.nn.Module):

    def forward(self, x):
        # get range, bearing for each match
        ranges, bearings = self.get_ranges_and_bearings_from_cartesian(x)
        # get theta, curvature for each range, bearing pair
        thetas, curvatures = self.get_thetas_and_curvatures_from_ranges_and_bearings(ranges, bearings)
        # pick theta (and its curvature) based on some heuristic (median)
        theta_estimate, curvature_estimate = self.select_theta_and_curvature(thetas, curvatures)
        return theta_estimate, curvature_estimate

    def get_ranges_and_bearings_from_cartesian(self, x):
        y2, y1, x2, x1 = torch.split(x, 1, dim=2)
        r1 = torch.norm(torch.cat([x1, y1], dim=2), dim=2, keepdim=True)
        r2 = torch.norm(torch.cat([x2, y2], dim=2), dim=2, keepdim=True)
        a1 = torch.atan2(y1, x1)
        a2 = torch.atan2(y2, x2)
        return torch.cat([r1, r2], dim=2), torch.cat([a1, a2], dim=2)  # we want 2, 10, 2

    def get_thetas_and_curvatures_from_ranges_and_bearings(self, ranges, bearings):
        r1, r2 = torch.split(ranges, 1, dim=2)
        a1, a2 = torch.split(bearings, 1, dim=2)
        thetas = 2 * torch.atan(
            (-torch.sin(a2) + (r1 / r2) * torch.sin(a1)) / ((r1 / r2) * torch.cos(a1) + torch.cos(a2)))
        radii = (r2 * torch.sin(a1 - a2 - thetas)) / (2 * torch.sin(thetas / 2) * torch.sin(-a1 + (thetas / 2)))
        curvatures = 1 / radii  # division by zero becomes inf, that's what we expect
        return thetas, curvatures

    def select_theta_and_curvature(self, thetas, curvatures):
        # Might replace this going forward and use the mask (or some sort of weighted score) to pick theta
        theta_estimate, median_indices = torch.median(thetas, dim=1)
        # Pick the curvature that corresponds to the median theta in each batch
        curvature_estimate = torch.gather(curvatures, 1, median_indices.unsqueeze(2))
        return theta_estimate, curvature_estimate


class CMNet(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.l1 = MyPlaceholderNetwork()
        self.cme = CircularMotionEstimationBase()
        self.loss = None  # probably an L1 but do this later

    def forward(self, x):
        x = x + self.l1(x)
        return self.cme(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(y, self.forward(x))
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
    input = torch.randn(2, 10, 4)  # batches, number of pairs, 4.
    model = CMNet({})
    model(input)
