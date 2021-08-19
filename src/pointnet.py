# Source of original template compiled from here: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
import numpy as np
import math
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
from loss_functions import LossFunctionOnTheta, LossFunctionCMParameters, LossFunctionFinalPose
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

    def forward(self, x):
        B, D, N = x.size()

        trans = self.STN2(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)  # I wonder if this should be split in 2 too
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
        self.k = 4  # 2  # correction to x and y, will add a mask as 3rd dim later
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True,
                                    channel=2)  # why not try with global_feat? - June 2021
        self.conv1 = torch.nn.Conv1d(1088 * 2, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        # self.tanh = weightedTanh()
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2, nn.ReLU(), self.conv3, self.bn3,
                                 nn.ReLU(), self.conv4)

        self.cme = CircularMotionEstimationBase()
        self.loss = LossFunctionFinalPose(self.device)

        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, 0.5 / math.sqrt(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Initialise last conv weights to 0 to ensure the initial correction is zero/small (?)
        self.net[-1].weight.data.zero_()

    def _forward(self, x):
        b, n, c = x.shape
        x = x.transpose(1, 2).float()
        landmark_positions = x.float()
        # Split landmarks and pass them through encoder individually
        x1 = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 2, :].unsqueeze(1)), dim=1)
        x2 = torch.cat((x[:, 1, :].unsqueeze(1), x[:, 3, :].unsqueeze(1)), dim=1)
        x1, _ = self.feat(x1)
        x2, _ = self.feat(x2)
        x = torch.cat((x1, x2), dim=1)

        x = self.net(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(b, n, self.k)

        # prediction_set = torch.zeros(b, n, c).to(self.device)
        # prediction_set[:, :, 1] = x[:, :, 0]
        # prediction_set[:, :, 3] = x[:, :, 1]
        # # Trying to correct both here
        # prediction_set[:, :, 0] = x[:, :, 2]
        # prediction_set[:, :, 2] = x[:, :, 3]

        if settings.DO_CORRECTION_MAGNITUDE_PLOTS:
            import numpy as np
            plt.figure(figsize=(10, 10))
            plt.grid()
            plt.title("Corrections in x and y position of each landmark")
            plt.plot(np.array(x[:, :, 0].squeeze(0).detach().numpy()), 'rx', markersize=2, mew=0.3,
                     label="y-correction")
            plt.plot(np.array(x[:, :, 1].squeeze(0).detach().numpy()), 'b+', markersize=2, mew=0.3,
                     label="x-correction")
            plt.savefig("%s%s%i%s" % (
                settings.RESULTS_DIR, "corrections/", settings.CORRECTION_PLOTTING_ITR, "_corrections.pdf"))
            plt.close()
            settings.CORRECTION_PLOTTING_ITR += 1

        landmark_positions = landmark_positions.transpose(1, 2)
        # corrected_landmark_positions = landmark_positions.add(prediction_set)
        corrected_landmark_positions = landmark_positions.add(x)

        # Scale landmark positions back up to metres (after being between [-1, 1] for predictions)
        corrected_landmark_positions = torch.mul(corrected_landmark_positions, settings.MAX_LANDMARK_RANGE_METRES)

        # Quick check
        if settings.DO_PLOTS_IN_FORWARD_PASS:
            import numpy as np
            landmark_positions = landmark_positions * settings.MAX_LANDMARK_RANGE_METRES
            alpha_val = 0.5
            plt.figure(figsize=(5, 5))
            plt.grid()
            dim_metres = 75  # settings.MAX_LANDMARK_RANGE_METRES
            plt.xlim(-dim_metres, dim_metres)
            plt.ylim(-dim_metres, dim_metres)
            plt.plot(np.array(landmark_positions[0, :, 1].detach().numpy()),
                     np.array(landmark_positions[0, :, 3].detach().numpy()), 'b,', alpha=alpha_val,
                     label="original_landmarks")
            plt.plot(np.array(corrected_landmark_positions[0, :, 1].detach().numpy()),
                     np.array(corrected_landmark_positions[0, :, 3].detach().numpy()), 'r,', alpha=alpha_val,
                     label="corrected_landmarks")
            plt.plot(np.array(landmark_positions[0, :, 0].detach().numpy()),
                     np.array(landmark_positions[0, :, 2].detach().numpy()), 'g,', alpha=alpha_val,
                     label="original_landmarks")
            plt.plot(np.array(corrected_landmark_positions[0, :, 0].detach().numpy()),
                     np.array(corrected_landmark_positions[0, :, 2].detach().numpy()), 'k,', alpha=alpha_val,
                     label="corrected_landmarks")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
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
        cme_parameters = self.cme(x)
        # Need to get landmarks within the inner set
        # Use thetas as a proxy for "best" matches (based on how well they are supported)
        estimated_thetas = cme_parameters[:, :, 0].to(self.device).type(torch.FloatTensor)
        b, n, _ = cme_parameters.shape
        values, indices = estimated_thetas.sort()
        expected_number_of_inliers = 300
        lower_index = int((settings.K_MAX_MATCHES - expected_number_of_inliers) / 2)
        upper_index = int((settings.K_MAX_MATCHES + expected_number_of_inliers) / 2)
        x = torch.gather(x, 1, indices[:, lower_index:upper_index].to(self.device).unsqueeze(2).expand(-1, -1, 4))

        # make_plots = False
        # if make_plots:
        #     cropped_thetas = torch.gather(estimated_thetas, 1, indices[:, lower_index:upper_index])
        #     first_n_thetas = torch.index_select(estimated_thetas, 1, torch.arange(0, expected_number_of_inliers))
        #     pdb.set_trace()
        #     import numpy as np
        #     plt.figure(figsize=(5, 5))
        #     plt.grid()
        #     plt.plot(np.array(np.sort(estimated_thetas[0, :].detach().numpy())), 'b,', label="original_thetas")
        #     plt.plot(np.array(np.sort(cropped_thetas[0, :].detach().numpy())), 'r,', label="cropped_thetas")
        #     plt.plot(np.array(np.sort(first_n_thetas[0, :].detach().numpy())), 'g,', label="n_thetas")
        #     plt.legend()
        #     plt.savefig("%s%s%i%s" % (
        #         settings.RESULTS_DIR, "landmarks/", settings.PLOTTING_ITR, "_debugging-plot.pdf"))
        #     plt.close()
        #     pdb.set_trace()
        corrected_landmark_positions = self._forward(x)
        return self.cme(corrected_landmark_positions), corrected_landmark_positions

    def training_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        prediction = self.forward(x)[0].to(self.device)
        loss = self.loss(prediction, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if settings.DO_GRADIENT_PLOTS:
            if settings.GRAD_PLOTTING_ITR > 0:
                plot_grad_flow(self.named_parameters())
            settings.GRAD_PLOTTING_ITR += 1
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['landmarks'], batch['cm_parameters']
        prediction = self.forward(x)[0].to(self.device)
        loss = self.loss(prediction, y)
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
