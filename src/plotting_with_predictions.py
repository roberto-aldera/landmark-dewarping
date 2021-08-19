# A place to store functions that plot while predictions are being generated
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv

import settings


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


def get_data_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        motion_estimate_data = list(reader)

    # timestamps = [int(item[0]) for item in motion_estimate_data]
    dx = [float(items[3]) for items in motion_estimate_data]
    dy = [float(items[4]) for items in motion_estimate_data]
    dth = [float(items[5]) for items in motion_estimate_data]
    return [dx, dy, dth]
