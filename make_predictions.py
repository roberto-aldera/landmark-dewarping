from argparse import ArgumentParser
from pathlib import Path
from cmnet import CMNet
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from custom_dataloader import LandmarksDataModule, LandmarkDataset, ToTensor, Normalise, SubsetSampling, ZeroPadding
from tqdm import tqdm
import numpy as np
import sys
import settings
import pdb
import csv

sys.path.insert(-1, "/workspace/code/landmark-distortion")
from R_and_theta_utilities import get_transform_by_r_and_theta
from get_cme_parameters_from_gt import save_timestamps_and_cme_to_csv, MotionEstimate


def do_prediction_and_optionally_export_csv(model, data_loader, do_csv_export=True):
    cm_predictions = []
    num_samples = len(data_loader.dataset)  # settings.TOTAL_SAMPLES
    for i in tqdm(range(num_samples)):
        landmarks = data_loader.dataset[i]['landmarks'].unsqueeze(0)
        cm_predictions.append(model(landmarks).detach().numpy().squeeze(0))
        # cm_predictions.append(data_loader.dataset[i]['cm_parameters'])  # these are the gt readings

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


def do_quick_plot_from_csv_files(gt_csv_file, pred_csv_file):
    import matplotlib.pyplot as plt

    # Read in CSVs for comparison
    gt_x_y_th = get_data_from_csv(gt_csv_file)
    pred_x_y_th = get_data_from_csv(pred_csv_file)

    plt.figure(figsize=(15, 5))
    dim = settings.TOTAL_SAMPLES + 50
    plt.xlim(0, dim)
    plt.grid()
    plt.plot(np.array(gt_x_y_th[0]), '+-', label="dx_gt")
    plt.plot(np.array(gt_x_y_th[1]), '+-', label="dy_gt")
    plt.plot(np.array(gt_x_y_th[2]), '+-', label="dth_gt")
    plt.plot(np.array(pred_x_y_th[0]), '+-', label="dx_pred")
    plt.plot(np.array(pred_x_y_th[1]), '+-', label="dy_pred")
    plt.plot(np.array(pred_x_y_th[2]), '+-', label="dth_pred")
    plt.title("Pose estimates")
    plt.xlabel("Sample index")
    plt.ylabel("units/sample")
    plt.legend()
    plt.savefig("%s%s" % (settings.RESULTS_DIR, "/pose_predictions_comparison.pdf"))
    plt.close()


if __name__ == "__main__":
    Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=settings.ARCHITECTURE_TYPE, help='cmnet or...')

    temp_args, _ = parser.parse_known_args()
    parser = CMNet.add_model_specific_args(parser)
    params = parser.parse_args()

    # Prepare model for evaluation
    path_to_model = "%s%s%s" % (settings.MODEL_DIR, params.model_name, ".ckpt")
    model = CMNet.load_from_checkpoint(path_to_model)
    model.eval()
    print("Loaded model from:", path_to_model)

    # Load data to evaluate over (just training data for now)
    transform = transforms.Compose([ToTensor(), Normalise(), SubsetSampling(), ZeroPadding()])
    dataset = LandmarkDataset(root_dir=settings.DATA_DIR, is_training_data=True,
                              transform=transform)
    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=4)

    do_prediction_and_optionally_export_csv(model, data_loader)
    do_quick_plot_from_csv_files(gt_csv_file="/workspace/data/landmark-dewarping/landmark-data/training/gt_poses.csv",
                                 pred_csv_file="/workspace/data/landmark-dewarping/evaluation/cm-pred_poses.csv")
