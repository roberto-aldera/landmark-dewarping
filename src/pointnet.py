# Source of original template compiled from here: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import pytorch_lightning as pl
import settings
from circular_motion_functions import CircularMotionEstimationBase
from argparse import ArgumentParser
import pdb


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for name, parameter in named_parameters:
        if parameter.requires_grad and ("bias" not in name):
            layers.append(name)
            ave_grads.append(parameter.abs().mean().detach().numpy())
            max_grads.append(parameter.abs().max().detach().numpy())
    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("%s%s%i%s" % (settings.RESULTS_DIR, "gradients/", settings.GRAD_PLOTTING_ITR, "_gradients.pdf"))
    plt.close()


class STNkd(nn.Module):  # Spatial Transformer Network
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

        # initialise weights
        # nn.init.constant_(self.conv1.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv2.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv3.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.fc1.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.fc2.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.fc3.weight, settings.WEIGHT_INIT_VAL)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.STN2 = STNkd(k=2)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        # initialise weights
        # nn.init.constant_(self.conv1.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv2.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv3.weight, settings.WEIGHT_INIT_VAL)

    def forward(self, x):
        B, D, N = x.size()

        # Collect landmarks (x, y) from set 1 and set 2
        x1 = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 2, :].unsqueeze(1)), dim=1)
        x2 = torch.cat((x[:, 1, :].unsqueeze(1), x[:, 3, :].unsqueeze(1)), dim=1)
        trans1 = self.STN2(x1)
        trans2 = self.STN2(x2)
        x1 = torch.bmm(x1.transpose(2, 1), trans1).transpose(2, 1)
        x2 = torch.bmm(x2.transpose(2, 1), trans2).transpose(2, 1)
        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans_feat


def loss_function(self, estimate, target):
    b, _, _ = estimate.shape
    estimated_thetas = estimate[:, :, 0]
    mask = torch.zeros_like(estimated_thetas)
    quantiles = torch.tensor([0.25, 0.75]).to(self.device)
    theta_quantiles = torch.quantile(estimated_thetas, quantiles, dim=1).transpose(0, 1)
    mask[(estimated_thetas > theta_quantiles[:, 0].unsqueeze(1)) & (
            estimated_thetas < theta_quantiles[:, 1].unsqueeze(1))] = 1

    if settings.DO_PLOTS_IN_LOSS:
        plt.figure(figsize=(10, 7))
        plt.grid()
        plt.ylim(-0.2, 0.2)
        plt.plot(np.sort(estimated_thetas.detach().numpy()), 'b+', markersize=2, mew=0.3, label="thetas")
        plt.plot(np.sort((estimated_thetas * mask).detach().numpy()), 'r+', markersize=2, mew=0.3, label="thetas")
        plt.title("Thetas from each match as batches")
        plt.ylabel("Theta (rad)")
        plt.xlabel("Index")
        plt.savefig(
            "%s%s" % (settings.RESULTS_DIR, "thetas_in_batches.pdf"))
        plt.close()

        # Get only those thetas that pass through the mask and plot them
        theta_subsection = np.sort(estimated_thetas[0, :].detach() * mask[0, :])
        theta_subsection = theta_subsection[theta_subsection != 0]
        plt.figure(figsize=(10, 7))
        plt.grid()
        plt.ylim(-0.2, 0.2)
        plt.plot(np.sort(estimated_thetas.detach().numpy()[0, :]), 'b+', markersize=2, mew=0.3, label="all thetas")
        plt.plot(np.linspace(quantiles[0].item() * settings.K_MAX_MATCHES, quantiles[1].item() * settings.K_MAX_MATCHES,
                             len(theta_subsection)), theta_subsection, 'r+', markersize=2, mew=0.3,
                 label="inner thetas")
        plt.title("Sorted thetas with quantiles")
        plt.ylabel("Theta (rad)")
        plt.xlabel("Index")
        plt.legend()
        plt.savefig(
            "%s%s" % (settings.RESULTS_DIR, "theta_quantiles.pdf"))
        plt.close()
        pdb.set_trace()
    target = target.float()
    _, n = estimated_thetas.shape
    target = torch.tile(target.unsqueeze(1), (1, n, 1))

    loss = F.mse_loss(estimated_thetas, target[:, :, 0], reduction="none")
    loss = loss * mask
    loss = loss.mean()
    return loss


class weightedTanh(nn.Module):
    def __init__(self, weights=1 / settings.MAX_LANDMARK_RANGE_METRES):
        super().__init__()
        self.weights = weights

    def forward(self, input):
        ex = torch.exp(2 * self.weights * input)
        return (ex - 1) / (ex + 1)


class PointNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PointNet, self).__init__()
        self.hparams = hparams
        self.k = 2  # correction to x and y, will add a mask as 3rd dim later
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=4)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.tanh = weightedTanh()
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2, nn.ReLU(), self.conv3, self.bn3,
                                 nn.ReLU(), self.conv4, self.tanh)

        self.cme = CircularMotionEstimationBase()

        # initialise weights
        # nn.init.constant_(self.conv1.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv2.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv3.weight, settings.WEIGHT_INIT_VAL)
        # nn.init.constant_(self.conv4.weight, settings.WEIGHT_INIT_VAL)

    def _forward(self, x):
        b, n, c = x.shape
        # pdb.set_trace()
        x = x.transpose(1, 2).float()
        landmark_positions = x.float()
        x, _ = self.feat(x)
        x = self.net(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(b, n, self.k)

        prediction_set = torch.zeros(b, n, c).to(self.device)
        # Perhaps set minimum prediction gate here
        # min_correction = 0.0004
        # too_tiny_0 = torch.where(x[:, :, 0].abs() < min_correction)
        # too_tiny_1 = torch.where(x[:, :, 1].abs() < min_correction)
        # x[:, too_tiny_0[1], 0] = 0
        # x[:, too_tiny_1[1], 1] = 0
        prediction_set[:, :, 1] = x[:, :, 0]
        prediction_set[:, :, 3] = x[:, :, 1]

        if settings.DO_CORRECTION_MAGNITUDE_PLOTS:
            import numpy as np
            plt.figure(figsize=(10, 10))
            plt.grid()
            plt.title("Corrections in x and y position of each landmark")
            plt.plot(np.array(x[:, :, 0].squeeze(0).detach().numpy()), 'rx', markersize=2, mew=0.3, label="0s")
            plt.plot(np.array(x[:, :, 1].squeeze(0).detach().numpy()), 'b+', markersize=2, mew=0.3, label="1s")
            plt.savefig("%s%s%i%s" % (
                settings.RESULTS_DIR, "corrections/", settings.CORRECTION_PLOTTING_ITR, "_corrections.pdf"))
            plt.close()
            settings.CORRECTION_PLOTTING_ITR += 1

        landmark_positions = landmark_positions.transpose(1, 2)
        corrected_landmark_positions = landmark_positions.add(prediction_set)

        # Scale landmark positions back up to metres (after being between [-1, 1] for predictions)
        corrected_landmark_positions = torch.mul(corrected_landmark_positions, settings.MAX_LANDMARK_RANGE_METRES)

        # Quick check
        if settings.DO_PLOTS_IN_FORWARD_PASS:
            import numpy as np
            landmark_positions = landmark_positions * settings.MAX_LANDMARK_RANGE_METRES
            plt.figure(figsize=(10, 10))
            plt.grid()
            plt.xlim(-settings.MAX_LANDMARK_RANGE_METRES, settings.MAX_LANDMARK_RANGE_METRES)
            plt.ylim(-settings.MAX_LANDMARK_RANGE_METRES, settings.MAX_LANDMARK_RANGE_METRES)
            plt.plot(np.array(landmark_positions[0, :, 1].detach().numpy()),
                     np.array(landmark_positions[0, :, 3].detach().numpy()), 'b,', label="original_landmarks")
            plt.plot(np.array(corrected_landmark_positions[0, :, 1].detach().numpy()),
                     np.array(corrected_landmark_positions[0, :, 3].detach().numpy()), 'r,',
                     label="corrected_landmarks")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("%s%s%i%s" % (
                settings.RESULTS_DIR, "landmarks/", settings.PLOTTING_ITR, "_landmarks-and-corrections.pdf"))
            plt.close()

            original_thetas = self.cme(landmark_positions).squeeze(0)[:, 0]
            corrected_thetas = self.cme(corrected_landmark_positions).squeeze(0)[:, 0]
            plt.figure(figsize=(10, 7))
            plt.grid()
            plt.ylim(-0.2, 0.2)
            plt.plot(np.sort(original_thetas.detach().numpy()), 'b+', markersize=2, mew=0.3, label="original_thetas")
            plt.plot(np.sort(corrected_thetas.detach().numpy()), 'rx', markersize=2, mew=0.3, label="corrected_thetas")
            # plt.plot(original_thetas.detach().numpy(), 'b.', markersize=2, label="original_thetas")
            # plt.plot(corrected_thetas.detach().numpy(), 'rx', markersize=2, label="corrected_thetas")
            plt.title("Sorted thetas from each match")
            plt.ylabel("Theta (rad)")
            plt.xlabel("Index")
            plt.legend()
            plt.savefig(
                "%s%s%i%s" % (settings.RESULTS_DIR, "thetas/", settings.PLOTTING_ITR, "_thetas_for_a_sample.pdf"))
            settings.PLOTTING_ITR += 1
            plt.close()
            # pdb.set_trace()

        return corrected_landmark_positions

    def forward(self, x):
        corrected_landmark_positions = self._forward(x)
        return self.cme(corrected_landmark_positions), corrected_landmark_positions

    def training_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        prediction = self.forward(x)[0].to(self.device)
        loss = loss_function(self, prediction, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if settings.DO_GRADIENT_PLOTS:
            if settings.GRAD_PLOTTING_ITR > 0:
                plot_grad_flow(self.named_parameters())
            settings.GRAD_PLOTTING_ITR += 1
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        prediction = self.forward(x)[0].to(self.device)
        loss = loss_function(self, prediction, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
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
    x = torch.rand((32, 600, 4))
    hparams = {}
    net = PointNet(hparams)
    y = net._forward(x)[0]
    plot_grad_flow(net.named_parameters())
    pdb.set_trace()
