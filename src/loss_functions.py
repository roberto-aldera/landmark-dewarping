import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
import settings


class LossFunctionFinalPose(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, estimate, target):
        b, _, _ = estimate.shape
        # Use thetas as a proxy for "best" matches (based on how well they are supported)
        estimated_thetas = estimate[:, :, 0].to(self.device)
        b, n, _ = estimate.shape
        # mask = torch.zeros(b, n, 3)
        # quantiles = torch.tensor([0, 1]).to(self.device)
        # theta_quantiles = torch.quantile(estimated_thetas, quantiles, dim=1).transpose(0, 1)
        # mask[(estimated_thetas > theta_quantiles[:, 0].unsqueeze(1)) & (
        #         estimated_thetas < theta_quantiles[:, 1].unsqueeze(1))] = 1

        # Convert target to x, y, theta
        gt_theta = target[:, 0]
        target[target[:, 1] == 0] = 1e-9
        gt_radius = 1 / target[:, 1]  # TODO: handle case where curvature could be zero (not yet observed)

        phi = gt_theta / 2  # this is because we're enforcing circular motion
        rho = 2 * gt_radius * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        d_theta = target[:, 0]

        pose_target = torch.stack((d_x, d_y, d_theta), dim=1)
        pose_target = pose_target.float().to(self.device)
        _, n = estimated_thetas.shape
        pose_target = torch.tile(pose_target.unsqueeze(1), (1, n, 1))

        # Convert estimates to x, y, theta
        estimated_curvatures = estimate[:, :, 1]
        estimated_curvatures[estimated_curvatures == 0] = 1e-9
        estimated_radii = torch.reciprocal(estimated_curvatures).to(self.device)
        phi = estimated_thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * estimated_radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        pose_estimate = torch.stack((d_x, d_y, estimated_thetas), dim=2)

        # loss = F.mse_loss(pose_estimate, pose_target, reduction="none")
        # pdb.set_trace()
        x_y_th_weights = torch.tensor([1, 1, 10])  # out of thin air for now

        weighted_pose_error = ((pose_estimate - pose_target) ** 2) * x_y_th_weights
        masked_error = weighted_pose_error #* mask
        nonzero_masked_error = masked_error[torch.abs(masked_error).sum(dim=2) != 0]
        # pel = torch.mean(nonzero_masked_error, dim=1)
        # pel = torch.mean(masked_error[mask != 0], axis=0)  # fix this mean so only nonzero elements are counted
        # loss = torch.mean(pel)
        loss = torch.mean(nonzero_masked_error)
        return loss


class LossFunctionCMParameters(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, estimate, target):
        b, _, _ = estimate.shape
        estimated_thetas = estimate[:, :, 0].to(self.device)
        mask = torch.zeros_like(estimate)
        quantiles = torch.tensor([0.1, 0.9]).to(self.device)
        theta_quantiles = torch.quantile(estimated_thetas, quantiles, dim=1).transpose(0, 1)
        mask[(estimated_thetas > theta_quantiles[:, 0].unsqueeze(1)) & (
                estimated_thetas < theta_quantiles[:, 1].unsqueeze(1))] = 1

        target = target.float()
        _, n = estimated_thetas.shape
        target = torch.tile(target.unsqueeze(1), (1, n, 1))  # TODO: check this, might be different from theta-only case
        loss = F.mse_loss(estimate, target, reduction="none")
        loss = loss * mask
        loss = loss.mean()
        return loss


class LossFunctionOnTheta(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, estimate, target):
        b, _, _ = estimate.shape
        estimated_thetas = estimate[:, :, 0].to(self.device)
        mask = torch.zeros_like(estimated_thetas)
        quantiles = torch.tensor([0.1, 0.9])
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
            plt.plot(
                np.linspace(quantiles[0].item() * settings.K_MAX_MATCHES, quantiles[1].item() * settings.K_MAX_MATCHES,
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
        target = target.float().to(self.device)
        _, n = estimated_thetas.shape
        target = torch.tile(target.unsqueeze(1), (1, n, 1))

        loss = F.mse_loss(estimated_thetas, target[:, :, 0], reduction="none")
        loss = loss * mask
        loss = loss.mean()
        return loss
