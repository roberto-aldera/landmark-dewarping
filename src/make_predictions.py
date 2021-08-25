from argparse import ArgumentParser
from pathlib import Path
from pointnet import PointNet
from scorenet import ScoreNet
from torchvision import transforms
from torch.utils.data import DataLoader
from pointnet_dataloader import SingleDataset, ToTensor, Normalise, FixSampleSize
from circular_motion_functions import get_transform_by_r_and_theta, save_timestamps_and_cme_to_csv, MotionEstimate, \
    CircularMotionEstimationBase
from tqdm import tqdm
import numpy as np
import torch
import settings
import pdb
import time


def do_prediction_and_optionally_export_csv(model, data_loader, export_path, do_csv_export=True):
    raw_pose_results = []
    network_pose_results = []
    # num_samples = 500
    num_samples = min(len(data_loader.dataset), settings.TOTAL_SAMPLES)
    quantile_width = 0.3
    quantiles = torch.tensor([0.5 - (quantile_width / 2), 0.5 + (quantile_width / 2)], dtype=torch.float32)
    print("Running for", num_samples, "samples...")

    for i in tqdm(range(num_samples)):
        landmarks = data_loader.dataset[i]['landmarks'].unsqueeze(0)
        # Get Circular Motion Estimates from raw landmarks as baseline
        cme_function = CircularMotionEstimationBase()
        raw_CMEs = cme_function(landmarks * settings.MAX_LANDMARK_RANGE_METRES).squeeze(0)
        raw_thetas = raw_CMEs[:, 0].type(torch.FloatTensor)
        raw_curvatures = raw_CMEs[:, 1].type(torch.FloatTensor)

        # Grab indices where theta is in a certain acceptable range
        theta_quantiles = torch.quantile(raw_thetas, quantiles)
        inlier_indices = torch.where((raw_thetas >= theta_quantiles[0]) & (raw_thetas <= theta_quantiles[1]))

        # Find dx, dy, dth for these indices
        selected_thetas = raw_thetas[inlier_indices]
        selected_radii = 1 / raw_curvatures[inlier_indices].type(torch.FloatTensor)

        phi = selected_thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * selected_radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[selected_radii == float('inf')] = 0
        d_y[selected_radii == float('inf')] = 0

        final_pose = [d_x.mean().detach().numpy(), d_y.mean().detach().numpy(), selected_thetas.mean().detach().numpy()]
        raw_pose_results.append(final_pose)

        # ------------------------- Fancier method here -------------------------#
        from sklearn.cluster import KMeans
        # from statistics import mode
        from collections import Counter
        kmeans = KMeans(n_clusters=8, random_state=0, tol=1e-6).fit(raw_thetas.reshape(-1, 1))
        # pdb.set_trace()
        counter = Counter(kmeans.labels_)
        biggest_cluster_idx = counter.most_common(1)[0][0]  # just take first biggest for now
        # biggest_cluster_idx = mode(kmeans.labels_)  # TODO: if two clusters are both same size, use both
        # -> maybe if clusters are similar size, use both. So if second biggest is like 95% the same size or something

        thetas_in_largest_cluster = raw_thetas[kmeans.labels_ == biggest_cluster_idx]
        radii_in_largest_cluster = 1 / raw_curvatures[kmeans.labels_ == biggest_cluster_idx].type(torch.FloatTensor)

        phi = thetas_in_largest_cluster / 2  # this is because we're enforcing circular motion
        rho = 2 * radii_in_largest_cluster * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[radii_in_largest_cluster == float('inf')] = 0
        d_y[radii_in_largest_cluster == float('inf')] = 0

        final_pose = [d_x.mean().detach().numpy(), d_y.mean().detach().numpy(),
                      thetas_in_largest_cluster.mean().detach().numpy()]
        network_pose_results.append(final_pose)

        # ------------------------- Network predictions -------------------------#
        # Get Circular Motion Estimates from from landmarks that have been corrected by network
        # predicted_scores = model(landmarks)

        # final_pose = predicted_poses.detach().numpy()[0]
        # network_pose_results.append(final_pose)

    # Get poses from raw CMEs
    raw_motion_estimates = []
    for idx in range(len(raw_pose_results)):
        x_est = raw_pose_results[idx][0]
        y_est = raw_pose_results[idx][1]
        th_est = raw_pose_results[idx][2]
        raw_motion_estimates.append(
            MotionEstimate(theta=0, curvature=0, dx=x_est, dy=y_est, dth=th_est))
    if do_csv_export:
        save_timestamps_and_cme_to_csv(timestamps=np.zeros(len(raw_pose_results)),
                                       motion_estimates=raw_motion_estimates,
                                       pose_source="raw_cm", export_folder=export_path)

    # Get poses from predicted correction CMEs
    motion_estimates = []

    for idx in range(len(network_pose_results)):
        x_est = network_pose_results[idx][0]
        y_est = network_pose_results[idx][1]
        th_est = network_pose_results[idx][2]
        motion_estimates.append(
            MotionEstimate(theta=0, curvature=0, dx=x_est, dy=y_est, dth=th_est))
    if do_csv_export:
        save_timestamps_and_cme_to_csv(timestamps=np.zeros(len(network_pose_results)),
                                       motion_estimates=motion_estimates,
                                       pose_source="network_cm", export_folder=export_path)


if __name__ == "__main__":
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = settings.RESULTS_DIR + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    # results_path = None  # just to stop folders being created while debugging
    print("Results will be saved to:", results_path)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_path', type=str,
                        default="%s%s%s" % (settings.MODEL_DIR, settings.ARCHITECTURE_TYPE, ".ckpt"),
                        help='Path to model that will be used to make predictions')

    temp_args, _ = parser.parse_known_args()
    parser = ScoreNet.add_model_specific_args(parser)
    params = parser.parse_args()

    # Prepare model for evaluation
    print("Loading model from:", params.model_path)
    model = ScoreNet(params)
    model = model.load_from_checkpoint(params.model_path)
    model.eval()

    # Load data to evaluate over (just training data for now)
    print("Evaluation running on data stored at:", settings.EVALUATION_DATA_DIR)
    transform = transforms.Compose([ToTensor(), Normalise(), FixSampleSize()])
    dataset = SingleDataset(root_dir=settings.EVALUATION_DATA_DIR, transform=transform)

    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=1)

    do_prediction_and_optionally_export_csv(model, data_loader, results_path)
