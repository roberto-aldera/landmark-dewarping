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
from tqdm import tqdm
import numpy as np
import torch
import settings
import pdb
import csv


def debugging_with_plots(model, data_loader):
    gt_cm = []
    cm_predictions = []
    num_samples = len(data_loader.dataset)
    for i in tqdm(range(num_samples)):
        landmarks = data_loader.dataset[i]['landmarks'].unsqueeze(0)
        cm_predictions_from_landmarks = model(landmarks).detach().numpy().squeeze(0)
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


def do_prediction_and_optionally_export_csv(model, data_loader, do_csv_export=True):
    raw_cm = []
    cm_predictions = []
    num_samples = len(data_loader.dataset)  # settings.TOTAL_SAMPLES
    for i in tqdm(range(num_samples)):
        landmarks = data_loader.dataset[i]['landmarks'].unsqueeze(0)
        # Get Circular Motion Estimates from raw landmarks as baseline
        cme_function = CircularMotionEstimationBase()
        raw_CMEs = cme_function(landmarks* settings.MAX_LANDMARK_RANGE_METRES).squeeze(0)
        raw_thetas = raw_CMEs[:, 0]
        raw_theta_estimate, raw_median_index = torch.median(raw_thetas, dim=0)
        raw_curvature_estimate = raw_CMEs[raw_median_index, 1]
        best_raw_cm_prediction = [raw_theta_estimate, raw_curvature_estimate]
        raw_cm.append(best_raw_cm_prediction)

        # Get Circular Motion Estimates from from landmarks that have been corrected by network
        prediction = model(landmarks).detach().squeeze(0)
        valid_predictions = prediction[~torch.any(prediction.isnan(), dim=1)]
        valid_thetas = valid_predictions[:, 0]
        theta_estimate, median_index = torch.median(valid_thetas, dim=0)
        curvature_estimate = valid_predictions[median_index, 1]
        best_cm_prediction = [theta_estimate, curvature_estimate]
        cm_predictions.append(best_cm_prediction)
        # cm_predictions.append(model(landmarks).detach().numpy().squeeze(0))

    # Get poses from raw CMEs
    raw_motion_estimates = []
    for idx in range(len(raw_cm)):
        th_estimate = np.array(raw_cm[idx][0])
        curvature_estimate = np.array(raw_cm[idx][1])
        if curvature_estimate == 0:
            r_estimate = np.inf
        else:
            r_estimate = 1 / curvature_estimate

        se3_from_r_theta = get_transform_by_r_and_theta(r_estimate, th_estimate)
        x_est = se3_from_r_theta[0, 3]
        y_est = se3_from_r_theta[1, 3]
        th_est = np.arctan2(se3_from_r_theta[1, 0], se3_from_r_theta[0, 0])
        raw_motion_estimates.append(
            MotionEstimate(theta=th_estimate, curvature=curvature_estimate, dx=x_est, dy=y_est, dth=th_est))
    if do_csv_export:
        save_timestamps_and_cme_to_csv(timestamps=np.zeros(len(cm_predictions)), motion_estimates=raw_motion_estimates,
                                       pose_source="raw_cm-pred", export_folder=settings.RESULTS_DIR)

    # Get poses from predicted correction CMEs
    motion_estimates = []
    for idx in range(len(cm_predictions)):
        th_estimate = np.array(cm_predictions[idx][0])
        curvature_estimate = np.array(cm_predictions[idx][1])
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
    if do_csv_export:
        save_timestamps_and_cme_to_csv(timestamps=np.zeros(len(cm_predictions)), motion_estimates=motion_estimates,
                                       pose_source="cm-pred", export_folder=settings.RESULTS_DIR)


def get_data_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        motion_estimate_data = list(reader)

    # timestamps = [int(item[0]) for item in motion_estimate_data]
    dx = [float(items[3]) for items in motion_estimate_data]
    dy = [float(items[4]) for items in motion_estimate_data]
    dth = [float(items[5]) for items in motion_estimate_data]
    return [dx, dy, dth]


def do_quick_plot_from_csv_files(gt_csv_file, raw_csv_file, pred_csv_file):
    # Read in CSVs for comparison
    gt_x_y_th = get_data_from_csv(gt_csv_file)
    raw_x_y_th = get_data_from_csv(raw_csv_file)
    pred_x_y_th = get_data_from_csv(pred_csv_file)

    plt.figure(figsize=(15, 5))
    dim = settings.TOTAL_SAMPLES + 50
    plt.xlim(0, dim)
    plt.grid()
    plt.plot(np.array(gt_x_y_th[0]), '+-', label="dx_gt")
    # plt.plot(np.array(gt_x_y_th[1]), '+-', label="dy_gt")
    # plt.plot(np.array(gt_x_y_th[2]), '+-', label="dth_gt")
    plt.plot(np.array(raw_x_y_th[0]), '+-', label="dx_raw")
    # plt.plot(np.array(raw_x_y_th[1]), '+-', label="dy_raw")
    # plt.plot(np.array(raw_x_y_th[2]), '+-', label="dth_raw")
    plt.plot(np.array(pred_x_y_th[0]), '+-', label="dx_pred")
    # plt.plot(np.array(pred_x_y_th[1]), '+-', label="dy_pred")
    # plt.plot(np.array(pred_x_y_th[2]), '+-', label="dth_pred")
    plt.title("Pose estimates")
    plt.xlabel("Sample index")
    plt.ylabel("units/sample")
    plt.legend()
    plt.savefig("%s%s" % (settings.RESULTS_DIR, "pose_predictions_comparison.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "pose_predictions_comparison.pdf"))


if __name__ == "__main__":
    Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=settings.ARCHITECTURE_TYPE, help='cmnet or...')

    temp_args, _ = parser.parse_known_args()
    parser = PointNet.add_model_specific_args(parser)
    params = parser.parse_args()

    # Prepare model for evaluation
    path_to_model = "%s%s%s" % (settings.MODEL_DIR, params.model_name, ".ckpt")
    # path_to_model = "/workspace/data/scratchdata/landmark-dewarping/models/pointnet.ckpt"

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
    do_prediction_and_optionally_export_csv(model, data_loader)
    do_quick_plot_from_csv_files(gt_csv_file="/workspace/data/landmark-dewarping/landmark-data/training/gt_poses.csv",
                                 raw_csv_file="/workspace/data/landmark-dewarping/evaluation/raw_cm-pred_poses.csv",
                                 pred_csv_file="/workspace/data/landmark-dewarping/evaluation/cm-pred_poses.csv")
