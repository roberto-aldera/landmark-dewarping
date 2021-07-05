from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
# from cmnet import CMNet
from pointnet import PointNet
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
# from custom_dataloader import LandmarksDataModule, LandmarkDataset, ToTensor, Normalise, SubsetSampling, ZeroPadding
from pointnet_dataloader import LandmarksDataModule, LandmarkDataset, ToTensor, Normalise, FixSampleSize
from circular_motion_functions import get_transform_by_r_and_theta, save_timestamps_and_cme_to_csv, MotionEstimate, \
    CircularMotionEstimationBase
from get_rigid_body_motion import get_motion_estimate_from_svd
from tqdm import tqdm
import numpy as np
import torch
import settings
import pdb
import csv
import time


def debugging_with_plots(model, data_loader):
    gt_cm = []
    cm_predictions = []
    num_samples = len(data_loader.dataset)
    for i in tqdm(range(num_samples)):
        landmarks = data_loader.dataset[i]['landmarks'].unsqueeze(0)
        cm_predictions_from_landmarks = model(landmarks)[0].detach().squeeze(0)
        best_theta = np.median(cm_predictions_from_landmarks[:, 0])
        best_curvature = 0  # don't need this yet
        cm_predictions.append([best_theta, best_curvature])
        gt_cm.append(data_loader.dataset[i]['cm_parameters'])

    # Plot the landmarks from the match (the second set will have had the corrections applied at this point)
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(np.array(landmarks[0, :, 0]), np.array(landmarks[0, :, 2]), ',')
    plt.plot(np.array(landmarks[0, :, 1]), np.array(landmarks[0, :, 3]), ',')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("%s%s" % (settings.RESULTS_DIR, "landmarks.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "landmarks.pdf"))

    # Plot thetas and the predicted thetas
    gt_thetas = [item[0] for item in gt_cm]
    pred_thetas = [item[0] for item in cm_predictions]

    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(gt_thetas, '+', label="gt_theta")
    plt.plot(pred_thetas, '.', label="est_theta")
    plt.legend()
    plt.savefig("%s%s" % (settings.RESULTS_DIR, "thetas.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "thetas.pdf"))


def do_prediction_and_optionally_export_csv(model, data_loader, export_path, do_csv_export=True):
    raw_pose_results = []
    dewarped_pose_results = []
    num_samples = min(len(data_loader.dataset), settings.TOTAL_SAMPLES)
    quantile_width = 0.5
    quantiles = torch.tensor([0.5 - (quantile_width / 2), 0.5 + (quantile_width / 2)], dtype=torch.float32)

    for i in tqdm(range(num_samples)):
        # i += 150  # quick hack to process instances where there is observable ego-motion
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

        # Get Circular Motion Estimates from from landmarks that have been corrected by network
        prediction, corrected_landmarks = model(landmarks)
        prediction = prediction.detach().squeeze(0)
        dewarped_thetas = prediction[:, 0]
        dewarped_curvatures = prediction[:, 1]

        # Grab indices where theta is in a certain acceptable range
        theta_quantiles = torch.quantile(dewarped_thetas, quantiles)
        inlier_indices = torch.where((dewarped_thetas > theta_quantiles[0]) & (dewarped_thetas < theta_quantiles[1]))
        # Find dx, dy, dth for these indices
        selected_thetas = dewarped_thetas[inlier_indices]
        selected_radii = 1 / dewarped_curvatures[inlier_indices].type(torch.FloatTensor)

        phi = selected_thetas / 2  # this is because we're enforcing circular motion
        rho = 2 * selected_radii * torch.sin(phi)
        d_x = rho * torch.cos(phi)  # forward motion
        d_y = rho * torch.sin(phi)  # lateral motion
        # Special cases
        d_x[selected_radii == float('inf')] = 0
        d_y[selected_radii == float('inf')] = 0

        final_pose = [d_x.mean().detach().numpy(), d_y.mean().detach().numpy(), selected_thetas.mean().detach().numpy()]
        dewarped_pose_results.append(final_pose)

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
    for idx in range(len(dewarped_pose_results)):
        x_est = dewarped_pose_results[idx][0]
        y_est = dewarped_pose_results[idx][1]
        th_est = dewarped_pose_results[idx][2]
        motion_estimates.append(
            MotionEstimate(theta=0, curvature=0, dx=x_est, dy=y_est, dth=th_est))
    if do_csv_export:
        save_timestamps_and_cme_to_csv(timestamps=np.zeros(len(dewarped_pose_results)),
                                       motion_estimates=motion_estimates,
                                       pose_source="corrections_cm", export_folder=export_path)


def get_data_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        motion_estimate_data = list(reader)

    # timestamps = [int(item[0]) for item in motion_estimate_data]
    dx = [float(items[3]) for items in motion_estimate_data]
    dy = [float(items[4]) for items in motion_estimate_data]
    dth = [float(items[5]) for items in motion_estimate_data]
    return [dx, dy, dth]


def do_quick_plot_from_csv_files(gt_csv_file, raw_csv_file, pred_csv_file, export_path):
    # Read in CSVs for comparison
    gt_x_y_th = get_data_from_csv(gt_csv_file)
    raw_x_y_th = get_data_from_csv(raw_csv_file)
    pred_x_y_th = get_data_from_csv(pred_csv_file)

    plt.figure(figsize=(15, 5))
    dim = settings.TOTAL_SAMPLES + 50
    plt.xlim(0, dim)
    plt.grid()
    m_size = 5
    line_width = 0.5
    plt.plot(np.array(gt_x_y_th[0]), 'b+-', linewidth=line_width, markersize=m_size, mew=0.3, label="dx_gt")
    plt.plot(np.array(gt_x_y_th[1]), 'bx-', linewidth=line_width, markersize=m_size, mew=0.3, label="dy_gt")
    plt.plot(np.array(gt_x_y_th[2]), 'bo-', linewidth=line_width, markersize=m_size, mew=0.3, fillstyle="none",
             label="dth_gt")
    plt.plot(np.array(raw_x_y_th[0]), 'r+-', linewidth=line_width, markersize=m_size, mew=0.3, label="dx_raw")
    plt.plot(np.array(raw_x_y_th[1]), 'rx-', linewidth=line_width, markersize=m_size, mew=0.3, label="dy_raw")
    plt.plot(np.array(raw_x_y_th[2]), 'ro-', linewidth=line_width, markersize=m_size, mew=0.3, fillstyle="none",
             label="dth_raw")
    plt.plot(np.array(pred_x_y_th[0]), 'g+-', linewidth=line_width, markersize=m_size, mew=0.3, label="dx_pred")
    plt.plot(np.array(pred_x_y_th[1]), 'gx-', linewidth=line_width, markersize=m_size, mew=0.3, label="dy_pred")
    plt.plot(np.array(pred_x_y_th[2]), 'go-', linewidth=line_width, markersize=m_size, mew=0.3, fillstyle="none",
             label="dth_pred")
    plt.title("Pose estimates")
    plt.xlabel("Sample index")
    plt.ylabel("units/sample")
    plt.legend()
    plt.savefig("%s%s" % (export_path, "pose_predictions_comparison.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (export_path, "pose_predictions_comparison.pdf"))


if __name__ == "__main__":
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = settings.RESULTS_DIR + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=settings.ARCHITECTURE_TYPE, help='cmnet or...')

    temp_args, _ = parser.parse_known_args()
    parser = PointNet.add_model_specific_args(parser)
    params = parser.parse_args()

    # Prepare model for evaluation
    # path_to_model = "/Volumes/scratchdata/roberto/landmark-dewarping/models/lightning_logs/version_179/checkpoints/epoch=1-step=97.ckpt"
    path_to_model = "%s%s%s" % (settings.MODEL_DIR, params.model_name, ".ckpt")
    # path_to_model = "/workspace/data/scratchdata/landmark-dewarping/models/pointnet.ckpt"
    # path_to_model = "/Volumes/scratchdata/roberto/landmark-dewarping/models/pointnet.ckpt"

    model = PointNet(params)
    model = model.load_from_checkpoint(path_to_model)
    model.eval()
    print("Loaded model from:", path_to_model)

    # Load data to evaluate over (just training data for now)
    transform = transforms.Compose([ToTensor(), Normalise(), FixSampleSize()])
    dataset = LandmarkDataset(root_dir=settings.DATA_DIR, is_training_data=True,
                              transform=transform)
    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=1)

    # debugging_with_plots(model, data_loader)
    do_prediction_and_optionally_export_csv(model, data_loader, results_path)
    # do_quick_plot_from_csv_files(gt_csv_file=settings.DATA_DIR + "training/gt_poses.csv",
    #                              raw_csv_file=results_path + "raw_cm_poses.csv",
    #                              pred_csv_file=results_path + "corrections_cm_poses.csv",
    #                              export_path=results_path)
