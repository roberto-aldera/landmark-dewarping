# Source of original template compiled from here: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import pytorch_lightning as pl
import settings
from circular_motion_functions import CircularMotionEstimationBase
from loss_functions import LossFunctionFinalPoseVsCmeGt
from argparse import ArgumentParser
from utilities import plot_scores_and_thetas, plot_thetas_in_batch
import pdb


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


class PointNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PointNet, self).__init__()
        self.hparams = hparams
        self.k = 1  # just predict a score for each match
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=2)
        self.conv1 = torch.nn.Conv1d(1088 * 2, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2, nn.ReLU(), self.conv3, self.bn3,
                                 nn.ReLU(), self.conv4, nn.Sigmoid())

        self.cme = CircularMotionEstimationBase()
        self.loss = LossFunctionFinalPoseVsCmeGt(self.device)

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
        # self.net[-1].weight.data.zero_()

    def _forward(self, x):
        b, n, c = x.shape
        x = x.transpose(1, 2).float()
        # Split landmarks and pass them through encoder individually
        x1 = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 2, :].unsqueeze(1)), dim=1)
        x2 = torch.cat((x[:, 1, :].unsqueeze(1), x[:, 3, :].unsqueeze(1)), dim=1)
        x1, _ = self.feat(x1)
        x2, _ = self.feat(x2)
        x = torch.cat((x1, x2), dim=1)

        x = self.net(x)
        x = x.transpose(2, 1).contiguous()
        scores = x.view(b, n)  # , self.k)

        return scores

    def forward(self, x):
        cme_parameters = self.cme(x)
        # estimated_thetas = cme_parameters[:, :, 0].to(self.device).type(torch.FloatTensor)
        scores = self._forward(x)

        threshold = min(torch.max(scores), 0.5)
        b, n, _ = x.shape
        mask = torch.zeros(b, n)
        mask[scores > threshold] = 1

        # Find dx, dy, dth for these indices
        raw_thetas = cme_parameters[:, :, 0].to(self.device).type(torch.FloatTensor)
        raw_curvatures = cme_parameters[:, :, 1].to(self.device).type(torch.FloatTensor)
        raw_radii = 1 / raw_curvatures.type(torch.FloatTensor)

        phi = raw_thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * raw_radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[raw_radii == float('inf')] = 0
        d_y[raw_radii == float('inf')] = 0

        denominator = mask.sum(dim=1) + 1e-9
        d_x_mean = (d_x * mask).sum(dim=1) / denominator
        d_y_mean = (d_y * mask).sum(dim=1) / denominator
        d_th_mean = (raw_thetas * mask).sum(dim=1) / denominator
        # pdb.set_trace()
        # mean must exclude zero values
        final_pose = torch.stack((d_x_mean, d_y_mean, d_th_mean), dim=1)

        # Do some plotting
        # if not settings.IS_RUNNING_ON_SERVER:
        # pdb.set_trace()
        # plot_scores_and_thetas(scores, estimated_thetas)
        # plot_thetas_in_batch(estimated_thetas)

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
        """
        Specify the hyperparams for this LightningModule
        """
        parent_parser.default_root_dir = settings.MODEL_DIR

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=settings.LEARNING_RATE, type=float)
        parser.add_argument('--batch_size', default=settings.BATCH_SIZE, type=int)

        # training specific (for this model)
        parser.add_argument('--max_num_epochs', default=settings.MAX_EPOCHS, type=int)

        return parser


if __name__ == "__main__":
    x = torch.rand((128, 600, 4))
    hparams = {}
    net = PointNet(hparams)
    y = net.forward(x)[0]
    # plot_grad_flow(net.named_parameters())
    pdb.set_trace()
