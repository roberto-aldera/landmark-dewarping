# Create an index file of every training csv, and the corresponding ground truth pose (label)

import os
import pandas as pd
import numpy as np
import settings
import pdb


def index_datasets(data_root, ro_end_offsets):
    index_file = data_root + "/dataset_index.txt"
    try:
        os.remove(index_file)
    except OSError:
        pass
    with open(index_file, 'a') as the_file:
        datasets_folder = data_root + "datasets/"
        folders = [f for f in os.listdir(datasets_folder) if not f.startswith('.')]  # ignores hidden files/folders
        for index, folder in enumerate(folders):
            landmark_folder = datasets_folder + folder + "/exported_matched_landmarks/"
            num_instances = len(os.listdir(landmark_folder)) - ro_end_offsets[index]
            print("Number of landmark set instances:", num_instances)
            for idx in range(num_instances):
                landmark_file = landmark_folder + "training_" + str(idx) + ".csv"
                the_file.write(landmark_file + '\n')


# Take cme-compliant gt_poses and check they line up with the RO timestamps, keep these
# Then concatenate them for all datasets
# Also handle ro_offsets, for cases where ro is longer than gt
# Seems like ro timestamps are always 1 frame out of sync (probably a source/destination timestamp thing)
# so handling of that happens here

def process_gt_poses_and_align_data(data_root):
    gt_poses_file = data_root + "/all_gt_poses.csv"
    try:
        os.remove(gt_poses_file)
    except OSError:
        pass

    ro_end_offsets = []

    datasets_folder = data_root + "datasets/"
    folders = [f for f in os.listdir(datasets_folder) if not f.startswith('.')]  # ignores hidden files/folders
    for folder in folders:
        data_folder = datasets_folder + folder
        ro_timestamps = pd.read_csv(data_folder + "/ro_timestamps.txt", header=None)
        all_cm_parameters = pd.read_csv(data_folder + "/gt_poses.csv", header=None)

        print("all_cm_parameters size:", len(all_cm_parameters))
        print("ro_timestamps size:", len(ro_timestamps))

        # Crop ground truth poses to only take those that align with RO
        cm_parameters_cropped = all_cm_parameters[:len(ro_timestamps)]
        # Then check timestamps are all within small tolerance
        ro_start, ro_end = 0, 0
        gt_start, gt_end = 0, 0
        timestamp_start_delta = cm_parameters_cropped.iloc[0, 0] - ro_timestamps.iloc[0, 0]
        timestamp_end_delta = cm_parameters_cropped.iloc[-1, 0] - ro_timestamps.iloc[-1, 0]

        if np.abs(timestamp_start_delta) > 500:  # then there's an offset needing to be corrected for
            if timestamp_start_delta < 0:
                gt_start = int(np.abs(np.round(timestamp_start_delta / 250000)))
            else:
                ro_start = int(np.abs(np.round(timestamp_start_delta / 250000)))
        if np.abs(timestamp_end_delta) > 500:  # then there's an offset needing to be corrected for
            if timestamp_end_delta > 0:
                gt_end = int(np.abs(np.round(timestamp_end_delta / 250000)))
            else:
                ro_end = int(np.abs(np.round(timestamp_end_delta / 250000)))
        ro_end = ro_end - 1  # there seems to be a single frame offset, so accounting for it here
        print(gt_start, gt_end, ro_start, ro_end)

        # print(np.abs((cm_parameters_cropped.iloc[gt_start, 0] - ro_timestamps.iloc[ro_start, 0])))
        # print(np.abs((cm_parameters_cropped.iloc[-(gt_end + 1), 0] - ro_timestamps.iloc[-(ro_end + 1), 0])))

        gt_start = gt_start - 1  # there seems to be a single frame offset, so accounting for it here
        if gt_end > 0:
            cm_parameters_cropped = cm_parameters_cropped[gt_start:-(gt_end + 1)]

        # ro_timestamps_cropped = ro_timestamps
        # if ro_end > 0:
        #     ro_timestamps_cropped = ro_timestamps[ro_start:-(ro_end + 1)]

        # These asserts must account for the single frame offset that seems to be present
        # assert (np.abs((cm_parameters_cropped.iloc[0, 0] - ro_timestamps_cropped.iloc[0, 0])) < 250000)
        # assert (np.abs((cm_parameters_cropped.iloc[-1, 0] - ro_timestamps_cropped.iloc[-1, 0])) < 500)

        # Write gt_poses to a file, appending as we go
        cm_parameters_cropped.to_csv(gt_poses_file, mode='a', header=None, index=False)

        print("cm_parameters_cropped size:", len(cm_parameters_cropped))
        print("ro_end size:", ro_end)
        pdb.set_trace()

        ro_end_offsets.append(ro_end)
    return ro_end_offsets


def main():
    ro_end_offsets = process_gt_poses_and_align_data(data_root=settings.DATA_DIR)
    index_datasets(data_root=settings.DATA_DIR, ro_end_offsets=ro_end_offsets)


if __name__ == "__main__":
    main()
