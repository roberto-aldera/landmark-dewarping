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
    num_samples = 1000
    # num_samples = min(len(data_loader.dataset), settings.TOTAL_SAMPLES)
    quantile_width = 0.5
    quantiles = torch.tensor([0.5 - (quantile_width / 2), 0.5 + (quantile_width / 2)], dtype=torch.float32)
    print("Running for", num_samples, "samples...")

    for i in tqdm(range(num_samples)):
        landmarks = data_loader.dataset[i]['landmarks'].unsqueeze(0)
        # Get Circular Motion Estimates from raw landmarks as baseline
        cme_function = CircularMotionEstimationBase()
        raw_CMEs = cme_function(landmarks * settings.MAX_LANDMARK_RANGE_METRES).squeeze(0)
        raw_thetas = raw_CMEs[:, 0].type(torch.FloatTensor)
        raw_curvatures = raw_CMEs[:, 1].type(torch.FloatTensor)

        sorted_thetas, sorted_indices = torch.sort(raw_thetas)
        sorted_curvatures = raw_curvatures[sorted_indices]

        # Grab indices where theta is in a certain acceptable range
        theta_quantiles = torch.quantile(sorted_thetas, quantiles)
        inlier_indices = torch.where((sorted_thetas >= theta_quantiles[0]) & (sorted_thetas <= theta_quantiles[1]))

        # Find dx, dy, dth for these indices
        selected_thetas = sorted_thetas[inlier_indices]
        selected_radii = 1 / sorted_curvatures[inlier_indices].type(torch.FloatTensor)

        phi = selected_thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * selected_radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[selected_radii == float('inf')] = 0
        d_y[selected_radii == float('inf')] = 0

        final_pose = [d_x.mean().detach().numpy(), d_y.mean().detach().numpy(), selected_thetas.mean().detach().numpy()]
        raw_pose_results.append(final_pose)

        # ------------------------- Network predictions -------------------------#
        # Get Circular Motion Estimates from from landmarks that have been corrected by network
        predicted_scores, _ = model(landmarks)
        if torch.sum(predicted_scores != 0):
            predicted_scores = predicted_scores / torch.sum(predicted_scores)
        # pdb.set_trace()
        # Get poses, and then weight each match by the score
        unsorted_thetas = raw_CMEs[:, 0].type(torch.FloatTensor)
        sorted_thetas, sorted_indices = torch.sort(unsorted_thetas)
        curvatures = raw_CMEs[:, 1].type(torch.FloatTensor)
        curvatures = curvatures[sorted_indices]
        radii = 1 / curvatures.type(torch.FloatTensor)

        phi = sorted_thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[radii == float('inf')] = 0
        d_y[radii == float('inf')] = 0

        # Weight all dx, dy, dth by the (normalised) scores, and sum to get final pose
        d_x = d_x * predicted_scores
        d_y = d_y * predicted_scores
        d_th = sorted_thetas * predicted_scores

        final_pose = torch.cat((torch.sum(d_x, dim=1).unsqueeze(1), torch.sum(d_y, dim=1).unsqueeze(1),
                                torch.sum(d_th, dim=1).unsqueeze(1)), dim=1)

        final_pose = final_pose.detach().numpy()[0]
        network_pose_results.append(final_pose)

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
