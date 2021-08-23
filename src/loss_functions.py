import torch
import torch.nn as nn
import pdb


class LossFunctionClassification(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, scores_and_thetas):
        scores = scores_and_thetas[0]
        thetas = scores_and_thetas[1]

        # Grab indices where theta is in a certain acceptable range
        quantile_width = 0.3
        quantiles = torch.tensor([0.5 - (quantile_width / 2), 0.5 + (quantile_width / 2)], dtype=torch.float32)
        theta_quantiles = torch.quantile(thetas, quantiles)
        y_target = torch.where(((thetas >= theta_quantiles[0]) & (thetas <= theta_quantiles[1])), 1., 0.)

        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(scores, y_target)
        return loss


class LossFunctionFinalPoseVsCmeGt(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, estimate, target):
        # Convert target to x, y, theta
        target = target.to(self.device)
        gt_theta = target[:, 0]
        target[target[:, 1] == 0] = 1e-9
        gt_radius = 1 / target[:, 1]

        phi = gt_theta / 2  # this is because we're enforcing circular motion
        rho = 2 * gt_radius * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        pose_target = torch.cat((d_x.unsqueeze(1), d_y.unsqueeze(1), gt_theta.unsqueeze(1)), dim=1).to(self.device)

        # ------------------------- Process network outputs -------------------------#
        x_y_th_weights = torch.tensor([1, 1, 1]).to(self.device)  # out of thin air for now
        weighted_pose_error = ((estimate.to(self.device) - pose_target) ** 2) * x_y_th_weights
        loss = torch.mean(weighted_pose_error)
        return loss


class LossFunctionFinalPose(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, estimate, target):
        # Use thetas as a proxy for "best" matches (based on how well they are supported)
        estimated_thetas = estimate[:, :, 0].to(self.device)

        # Convert target to x, y, theta
        gt_theta = target[:, 0]
        target[target[:, 1] == 0] = 1e-9
        gt_radius = 1 / target[:, 1]

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

        x_y_th_weights = torch.tensor([1, 1, 10])  # out of thin air for now
        weighted_pose_error = ((pose_estimate - pose_target) ** 2) * x_y_th_weights
        loss = torch.mean(weighted_pose_error)
        return loss
