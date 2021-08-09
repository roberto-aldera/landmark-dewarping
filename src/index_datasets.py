# Create an index file of every training csv, and the corresponding ground truth pose (label)

import os
import pandas as pd
import numpy as np
import settings
import pdb


def index_datasets(data_root):
    index_file = data_root + "/dataset_index.txt"
    try:
        os.remove(index_file)
    except OSError:
        pass
    with open(index_file, 'a') as the_file:
        datasets_folder = data_root + "datasets/"
        folders = [f for f in os.listdir(datasets_folder) if not f.startswith('.')]  # ignores hidden files/folders
        for folder in folders:
            landmark_folder = datasets_folder + folder + "/exported_matched_landmarks/"
            num_instances = len(os.listdir(landmark_folder))
            for idx in range(num_instances):
                landmark_file = landmark_folder + "training_" + str(idx) + ".csv"
                the_file.write(landmark_file + '\n')


# Take cme-compliant gt_poses and check they line up with the RO timestamps, keep these
# Then concatenate them for all datasets
def process_gt_poses(data_root):
    gt_poses_file = data_root + "/all_gt_poses.csv"
    try:
        os.remove(gt_poses_file)
    except OSError:
        pass
    datasets_folder = data_root + "datasets/"
    folders = [f for f in os.listdir(datasets_folder) if not f.startswith('.')]  # ignores hidden files/folders
    for folder in folders:
        data_folder = datasets_folder + folder
        ro_timestamps = pd.read_csv(data_folder + "/ro_timestamps.txt", header=None)
        all_cm_parameters = pd.read_csv(data_folder + "/gt_poses.csv", header=None)
        # We expect the ground truth poses to be aligned with RO, but they are always out by 1 frame (~250000)
        assert ((all_cm_parameters.iloc[1, 0] - ro_timestamps.iloc[0, 0]) < 500)

        # Crop ground truth poses to only take those that align with RO
        cm_parameters_cropped = all_cm_parameters[1:len(ro_timestamps)+1]
        # Then check timestamps are all within small tolerance
        assert ((cm_parameters_cropped.iloc[0, 0] - ro_timestamps.iloc[0, 0]) < 500)
        assert ((cm_parameters_cropped.iloc[-1, 0] - ro_timestamps.iloc[-1, 0]) < 500)

        # Write gt_poses to a file, appending as we go
        cm_parameters_cropped.to_csv(gt_poses_file, mode='a', header=None, index=False)


def main():
    index_datasets(data_root=settings.DATA_DIR)
    process_gt_poses(data_root=settings.DATA_DIR)


if __name__ == "__main__":
    main()
