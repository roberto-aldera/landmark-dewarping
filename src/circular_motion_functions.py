import numpy as np
import torch
import pdb
from dataclasses import dataclass
import csv
from tqdm import tqdm
import settings
from torch.utils.data import DataLoader
from custom_dataloader import LandmarkDataset, ToTensor, Normalise, SubsetSampling, ZeroPadding
from torchvision import transforms


@dataclass
class MotionEstimate:
    theta: float
    curvature: float
    dx: float
    dy: float
    dth: float


class CircularMotionEstimationBase(torch.nn.Module):

    def forward(self, x):
        # get range, bearing for each match
        ranges, bearings, validity_mask = self.get_ranges_and_bearings_from_cartesian(x)
        # get theta, curvature for each range, bearing pair
        thetas, curvatures = self.get_thetas_and_curvatures_from_ranges_and_bearings(ranges, bearings)
        # pick theta (and its curvature) based on some heuristic (median)
        theta_estimate, curvature_estimate = self.select_theta_and_curvature(thetas, curvatures, validity_mask)
        # pdb.set_trace()
        return torch.cat([theta_estimate, curvature_estimate], dim=1)  # dimensions here still TBD

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
        return torch.tensor(theta_estimates, requires_grad=True).view(thetas.shape[0], -1), \
               torch.tensor(curvature_estimates, requires_grad=True).view(thetas.shape[0], -1)

    def select_theta_and_curvature_with_iqr(self, thetas, curvatures, validity_mask):
        theta_estimates, curvature_estimates = [], []
        for batch_idx in range(thetas.shape[0]):
            # Mask out the padding
            valid_thetas = torch.tensor(
                [item for idx, item in enumerate(thetas[batch_idx, :]) if validity_mask[batch_idx, idx] == 1])
            valid_curvatures = torch.tensor(
                [item for idx, item in enumerate(curvatures[batch_idx, :]) if validity_mask[batch_idx, idx] == 1])
            # Might replace this going forward and use the mask (or some sort of weighted score) to pick theta
            # Find quantiles
            quantiles = torch.tensor([0.25, 0.75])
            theta_quantiles = torch.quantile(valid_thetas, quantiles.double())
            iqr_thetas_indices = ((valid_thetas >= theta_quantiles[0]) & (valid_thetas <= theta_quantiles[1])).nonzero()
            iqr_thetas = valid_thetas[(valid_thetas >= theta_quantiles[0]) & (valid_thetas <= theta_quantiles[1])]
            # Find the median in this IQR
            _, iqr_median_index = torch.median(iqr_thetas, dim=0)

            # Find the index in the bigger set corresponding to this median index
            median_index = iqr_thetas_indices[iqr_median_index]
            theta_estimate = valid_thetas[median_index]
            curvature_estimate = valid_curvatures[median_index]
            theta_estimates.append(theta_estimate)
            curvature_estimates.append(curvature_estimate)
        return torch.tensor(theta_estimates, requires_grad=True).view(thetas.shape[0], -1), \
               torch.tensor(curvature_estimates, requires_grad=True).view(thetas.shape[0], -1)


def get_transform_by_translation_and_theta(translation_x, translation_y, theta):
    T_offset = np.array([[translation_x], [translation_y]])
    pose = np.identity(4)
    pose[0, 0] = np.cos(theta)
    pose[0, 1] = -np.sin(theta)
    pose[1, 0] = np.sin(theta)
    pose[1, 1] = np.cos(theta)
    pose[0, 3] = T_offset[0]
    pose[1, 3] = T_offset[1]
    return pose


def get_transform_by_r_and_theta(rotation_radius, theta):
    phi = theta / 2  # this is because we're enforcing circular motion
    if rotation_radius == np.inf and theta == 0:
        rho = 0
    else:
        rho = 2 * rotation_radius * np.sin(phi)
    rho_x = rho * np.cos(phi)  # forward motion
    rho_y = rho * np.sin(phi)  # lateral motion
    return get_transform_by_translation_and_theta(rho_x, rho_y, theta)


def save_timestamps_and_cme_to_csv(timestamps, motion_estimates, pose_source, export_folder):
    # Save poses with format: timestamp, theta, curvature, dx, dy, dth
    with open("%s%s%s" % (export_folder, pose_source, "_poses.csv"), 'w') as poses_file:
        wr = csv.writer(poses_file, delimiter=",")
        th_values = [item.dth for item in motion_estimates]
        curvature_values = [item.curvature for item in motion_estimates]
        x_values = [item.dx for item in motion_estimates]
        y_values = [item.dy for item in motion_estimates]
        for idx in range(len(timestamps)):
            timestamp_and_motion_estimate = [timestamps[idx], th_values[idx], curvature_values[idx], x_values[idx],
                                             y_values[idx], th_values[idx]]
            wr.writerow(timestamp_and_motion_estimate)


def get_data_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        motion_estimate_data = list(reader)

    # timestamps = [int(item[0]) for item in motion_estimate_data]
    dx = [float(items[3]) for items in motion_estimate_data]
    dy = [float(items[4]) for items in motion_estimate_data]
    dth = [float(items[5]) for items in motion_estimate_data]
    return [dx, dy, dth]


def do_quick_plot_from_csv_files(gt_csv_file, est_csv_file):
    import matplotlib.pyplot as plt

    # Read in CSVs for comparison
    gt_x_y_th = get_data_from_csv(gt_csv_file)
    est_x_y_th = get_data_from_csv(est_csv_file)

    plt.figure(figsize=(15, 5))
    dim = settings.TOTAL_SAMPLES + 50
    plt.xlim(0, dim)
    plt.grid()
    plt.plot(np.array(gt_x_y_th[0]), '+-', label="dx_gt")
    plt.plot(np.array(gt_x_y_th[1]), '+-', label="dy_gt")
    plt.plot(np.array(gt_x_y_th[2]), '+-', label="dth_gt")
    plt.plot(np.array(est_x_y_th[0]), '+-', label="dx_est")
    plt.plot(np.array(est_x_y_th[1]), '+-', label="dy_est")
    plt.plot(np.array(est_x_y_th[2]), '+-', label="dth_est")
    plt.title("Pose estimates")
    plt.xlabel("Sample index")
    plt.ylabel("units/sample")
    plt.legend()
    plt.savefig("%s%s" % (settings.RESULTS_DIR, "/pose_comparison.pdf"))
    plt.close()


def check_cm_pipeline_and_optionally_export_csv(do_csv_export=False):
    cm_estimator = CircularMotionEstimationBase()
    transform = transforms.Compose([ToTensor(), SubsetSampling(), ZeroPadding()])
    dataset = LandmarkDataset(root_dir=settings.DATA_DIR, is_training_data=True,
                              transform=transform)
    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=4)

    cm_estimates = []

    for data in tqdm(data_loader):
        landmarks, cm_parameters = data['landmarks'], data['cm_parameters']
        cm_estimates.append(cm_estimator(landmarks))

    # Assuming batch size = 1 for this checking section...
    motion_estimates = []
    for idx in range(len(cm_estimates)):
        th_estimate = np.array(cm_estimates[idx].detach().numpy().squeeze(0)[0])
        curvature_estimate = np.array(cm_estimates[idx].detach().numpy().squeeze(0)[1])
        if curvature_estimate == 0:
            r_estimate = np.inf
        else:
            r_estimate = 1 / curvature_estimate

        se3_from_r_theta = get_transform_by_r_and_theta(r_estimate, th_estimate)
        x_est = se3_from_r_theta[0, 3]
        y_est = se3_from_r_theta[1, 3]
        th_est = np.arctan2(se3_from_r_theta[1, 0], se3_from_r_theta[0, 0])
        motion_estimates.append(
            MotionEstimate(theta=th_estimate.item(), curvature=curvature_estimate.item(), dx=x_est, dy=y_est,
                           dth=th_est))
    if do_csv_export:
        save_timestamps_and_cme_to_csv(timestamps=np.zeros(len(cm_estimates)), motion_estimates=motion_estimates,
                                       pose_source="cm-est", export_folder=settings.RESULTS_DIR)


if __name__ == "__main__":
    print("Running circular motion function script...")
    check_cm_pipeline_and_optionally_export_csv(do_csv_export=True)

    do_quick_plot_from_csv_files(gt_csv_file="/workspace/data/landmark-dewarping/landmark-data/training/gt_poses.csv",
                                 est_csv_file="/workspace/data/landmark-dewarping/evaluation/cm-est_poses.csv")
