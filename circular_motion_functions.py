import numpy as np
import torch
import pdb
import sys
import settings
from custom_dataloader import LandmarksDataModule

sys.path.insert(-1, "/workspace/code/landmark-distortion")
from R_and_theta_utilities import get_transform_by_r_and_theta
from get_cme_parameters_from_gt import save_timestamps_and_cme_to_csv, MotionEstimate


class CircularMotionEstimationBase(torch.nn.Module):

    def forward(self, x):
        # get range, bearing for each match
        ranges, bearings, validity_mask = self.get_ranges_and_bearings_from_cartesian(x)
        # get theta, curvature for each range, bearing pair
        thetas, curvatures = self.get_thetas_and_curvatures_from_ranges_and_bearings(ranges, bearings)
        # pick theta (and its curvature) based on some heuristic (median)
        theta_estimate, curvature_estimate = self.select_theta_and_curvature(thetas, curvatures, validity_mask)
        return theta_estimate, curvature_estimate

    def get_ranges_and_bearings_from_cartesian(self, x):
        validity_mask = torch.zeros(x.shape[0:2])
        for batch_idx in range(x.shape[0]):
            valid_cols = [col_idx for col_idx, col in enumerate(torch.split(x[batch_idx], 1, dim=0)) if
                          not torch.all(col == 0)]
            validity_mask[batch_idx, valid_cols] = 1

        y2, y1, x2, x1 = torch.split(x, 1, dim=2)
        r1 = torch.norm(torch.cat([x1, y1], dim=2), dim=2, keepdim=True)
        r2 = torch.norm(torch.cat([x2, y2], dim=2), dim=2, keepdim=True)
        a1 = torch.atan2(y1, x1)
        a2 = torch.atan2(y2, x2)
        return torch.cat([r1, r2], dim=2), torch.cat([a1, a2], dim=2), validity_mask  # we want 2, 10, 2

    def get_thetas_and_curvatures_from_ranges_and_bearings(self, ranges, bearings):
        r1, r2 = torch.split(ranges, 1, dim=2)
        a1, a2 = torch.split(bearings, 1, dim=2)
        # Keep track of when landmarks are in the exact same position as their match (stationary vehicle)
        stationary_landmark_mask = (r1 == r2) & (a1 == a2)

        thetas = 2 * torch.atan(
            (-torch.sin(a2) + (r1 / r2) * torch.sin(a1)) / ((r1 / r2) * torch.cos(a1) + torch.cos(a2)))
        radii = (r2 * torch.sin(a1 - a2 - thetas)) / (2 * torch.sin(thetas / 2) * torch.sin(-a1 + (thetas / 2)))
        radii.masked_fill_(stationary_landmark_mask, float('inf'))
        curvatures = 1 / radii  # division by zero becomes inf, that's what we expect
        return thetas, curvatures

    def select_theta_and_curvature(self, thetas, curvatures, validity_mask):
        theta_estimates, curvature_estimates = [], []
        for batch_idx in range(thetas.shape[0]):
            # Mask out the padding
            valid_thetas = torch.tensor(
                [item for idx, item in enumerate(thetas[batch_idx, :]) if validity_mask[batch_idx, idx] == 1])
            valid_curvatures = torch.tensor(
                [item for idx, item in enumerate(curvatures[batch_idx, :]) if validity_mask[batch_idx, idx] == 1])
            # Might replace this going forward and use the mask (or some sort of weighted score) to pick theta
            theta_estimate, median_index = torch.median(valid_thetas, dim=0)
            # Pick the curvature that corresponds to the median theta in each batch
            curvature_estimate = valid_curvatures[median_index]
            theta_estimates.append(theta_estimate)
            curvature_estimates.append(curvature_estimate)

        # # Might replace this going forward and use the mask (or some sort of weighted score) to pick theta
        # theta_estimate, median_indices = torch.median(thetas, dim=1)
        # # Pick the curvature that corresponds to the median theta in each batch
        # curvature_estimate = torch.gather(curvatures, 1, median_indices.unsqueeze(2))
        return torch.tensor(theta_estimates), torch.tensor(curvature_estimates)
        # return theta_estimate, curvature_estimate


if __name__ == "__main__":
    print("Running :)")
    # Just a place to test the circular motion estimates are working as expected
    dm = LandmarksDataModule()
    dm.setup()
    the_thing = CircularMotionEstimationBase()
    dl = dm.train_dataloader()

    cm_estimates = []

    for data in dl:
        landmarks, cm_parameters = data['landmarks'], data['cm_parameters']
        cm_estimate = the_thing(landmarks)
        cm_estimates.append(cm_estimate)

    # Assuming batch size = 1 for this checking section...
    motion_estimates = []
    for idx in range(len(cm_estimates)):
        th_estimate = np.array(cm_estimates[idx][0])
        curvature_estimate = np.array(cm_estimates[idx][1])
        if curvature_estimate == 0:
            r_estimate = np.inf
        else:
            r_estimate = 1 / curvature_estimate

        se3_from_r_theta = get_transform_by_r_and_theta(r_estimate, th_estimate)
        x_est = se3_from_r_theta[0, 3]
        y_est = se3_from_r_theta[1, 3]
        th_est = np.arctan2(se3_from_r_theta[1, 0], se3_from_r_theta[0, 0])
        motion_estimates.append(
            MotionEstimate(theta=th_estimate, curvature=curvature_estimate, dx=x_est, dy=y_est, dth=th_est))

    timestamps = np.zeros(len(cm_estimates))
    save_timestamps_and_cme_to_csv(timestamps, motion_estimates, "cm-est", settings.RESULTS_DIR)
