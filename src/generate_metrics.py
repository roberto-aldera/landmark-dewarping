import numpy as np
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb
from pyslam.metrics import TrajectoryMetrics
import pandas as pd
import csv
from liegroups import SE3


def get_metrics(params):
    # Some code to run KITTI metrics over poses, based on pyslam TrajectoryMetrics
    gt_se3s, gt_timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    gt_se3s = gt_se3s[settings.K_RADAR_INDEX_OFFSET:]

    # Pose estimates from full matches
    full_matches_timestamps, full_matches_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.path + "6900_full_matches_poses.csv")

    # Aux 0 - TODO: bring timestamps back.
    _, aux0_x_y_th = get_timestamps_and_x_y_th_from_circular_motion_estimate_csv(
        params.path + "raw_cm-pred_poses.csv")

    # Aux 1 - TODO: bring timestamps back.
    _, aux1_x_y_th = get_timestamps_and_x_y_th_from_circular_motion_estimate_csv(
        params.path + "cm-pred_poses.csv")

    # Cropping if necessary
    full_matches_timestamps, full_matches_x_y_th = full_matches_timestamps[:2000], full_matches_x_y_th[:2000]
    # aux0_timestamps, aux0_x_y_th = aux0_timestamps[:2000], aux0_x_y_th[:2000]

    full_matches_se3s = get_raw_se3s_from_x_y_th(full_matches_x_y_th)
    aux0_se3s = get_raw_se3s_from_x_y_th(aux0_x_y_th)
    aux1_se3s = get_raw_se3s_from_x_y_th(aux1_x_y_th)

    relative_pose_index = settings.K_RADAR_INDEX_OFFSET + 1
    relative_pose_timestamp = gt_timestamps[relative_pose_index]

    # ensure timestamps are within a reasonable limit of each other (microseconds)
    assert (full_matches_timestamps[0] - relative_pose_timestamp) < 500
    # assert (aux0_timestamps[0] - relative_pose_timestamp) < 500
    # assert (aux1_timestamps[0] - relative_pose_timestamp) < 500

    # making global poses from the relative poses
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_global_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    fm_global_se3s = [np.identity(4)]
    for i in range(1, len(full_matches_se3s)):
        fm_global_se3s.append(fm_global_se3s[i - 1] @ full_matches_se3s[i])
    full_matches_global_SE3s = get_se3s_from_raw_se3s(fm_global_se3s)

    aux0_global_se3s = [np.identity(4)]
    for i in range(1, len(aux0_se3s)):
        aux0_global_se3s.append(aux0_global_se3s[i - 1] @ aux0_se3s[i])
    aux0_global_SE3s = get_se3s_from_raw_se3s(aux0_global_se3s)

    aux1_global_se3s = [np.identity(4)]
    for i in range(1, len(aux1_se3s)):
        aux1_global_se3s.append(aux1_global_se3s[i - 1] @ aux1_se3s[i])
    aux1_global_SE3s = get_se3s_from_raw_se3s(aux1_global_se3s)

    # segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    segment_lengths = [50, 100, 150]

    tm_gt_fullmatches = TrajectoryMetrics(gt_global_SE3s, full_matches_global_SE3s)
    print_trajectory_metrics(tm_gt_fullmatches, segment_lengths, data_name="full match")

    tm_gt_aux0 = TrajectoryMetrics(gt_global_SE3s, aux0_global_SE3s)
    print_trajectory_metrics(tm_gt_aux0, segment_lengths, data_name=settings.AUX0_NAME)

    tm_gt_aux1 = TrajectoryMetrics(gt_global_SE3s, aux1_global_SE3s)
    print_trajectory_metrics(tm_gt_aux1, segment_lengths, data_name=settings.AUX1_NAME)

    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    output_path_for_metrics = Path(params.path + "visualised_metrics")
    if output_path_for_metrics.exists() and output_path_for_metrics.is_dir():
        shutil.rmtree(output_path_for_metrics)
    output_path_for_metrics.mkdir(parents=True)

    visualiser = TrajectoryVisualizer(
        {"full_matches": tm_gt_fullmatches, settings.AUX0_NAME: tm_gt_aux0, settings.AUX1_NAME: tm_gt_aux1})
    visualiser.plot_cum_norm_err(outfile="%s%s" % (output_path_for_metrics, "/cumulative_norm_errors.pdf"))
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="%s%s" % (output_path_for_metrics, "/segment_errors.pdf"))
    visualiser.plot_topdown(which_plane='yx',  # this is a custom flip to conform to MRG convention, instead of xy
                            outfile="%s%s" % (output_path_for_metrics, "/topdown.pdf"), figsize=(10, 10))


def print_trajectory_metrics(tm_gt_est, segment_lengths, data_name="this"):
    print("\nTrajectory Metrics for", data_name, "set:")
    # print("endpoint_error:", tm_gt_est.endpoint_error(segment_lengths))
    # print("segment_errors:", tm_gt_est.segment_errors(segment_lengths))
    # print("traj_errors:", tm_gt_est.traj_errors())
    # print("rel_errors:", tm_gt_est.rel_errors())
    # print("error_norms:", tm_gt_est.error_norms())
    print("mean_err:", tm_gt_est.mean_err())
    # print("cum_err:", tm_gt_est.cum_err())
    print("rms_err:", tm_gt_est.rms_err())


def get_ground_truth_poses_from_csv(path_to_gt_csv):
    """
    Load poses from csv for the Oxford radar robotcar 10k dataset.
    """
    df = pd.read_csv(path_to_gt_csv)
    # print(df.head())
    x_vals = df['x']
    y_vals = df['y']
    th_vals = df['yaw']
    timestamps = df['source_radar_timestamp']

    se3s = []
    for i in range(len(df.index)):
        th = th_vals[i]
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = x_vals[i]
        pose[1, 3] = y_vals[i]
        se3s.append(pose)
    return se3s, timestamps


def get_timestamps_and_x_y_th_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        pose_data = list(reader)

    timestamps = [int(item[0]) for item in pose_data]
    x_y_th = [items[1:] for items in pose_data]
    return timestamps, x_y_th


def get_timestamps_and_x_y_th_from_circular_motion_estimate_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        pose_data = list(reader)

    # timestamps = [int(item[0]) for item in pose_data]
    timestamps = [int(0) for item in pose_data]
    x_y_th = [items[3:] for items in pose_data]
    return timestamps, x_y_th


def get_se3s_from_raw_se3s(raw_se3s):
    """
    Transform from raw se3 matrices into fancier SE3 type
    """
    se3s = []
    for pose in raw_se3s:
        se3s.append(SE3.from_matrix(np.asarray(pose)))
    return se3s


def get_raw_se3s_from_x_y_th(x_y_th):
    """
    Transform from x_y_th list into raw SE3 type
    """
    se3s = []
    for sample in x_y_th:
        th = float(sample[2])
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = float(sample[0])
        pose[1, 3] = float(sample[1])
        se3s.append(pose)
    return se3s


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--path', type=str, default="",
                        help='Path to folder where inputs are and where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")
    # python generate_metrics.py --path "/workspace/data/landmark-dewarping/metrics/" --num_samples 2000

    get_metrics(params)


if __name__ == "__main__":
    main()
