import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import logging

sys.path.insert(-1, "/workspace/code/landmark-distortion")
from get_rigid_body_motion import get_motion_estimate_from_svd

# create logger
logger = logging.getLogger('__name__')
np.random.seed(0)


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


def get_robot_to_world_transform():
    # Need to go from x+ forward, y+ right, theta+ cw to x+ right, y+ forward, theta+ ccw
    rot_z = np.identity(4)
    th_90 = -np.pi / 2
    rot_z[0, 0] = np.cos(th_90)
    rot_z[0, 1] = -np.sin(th_90)
    rot_z[1, 0] = np.sin(th_90)
    rot_z[1, 1] = np.cos(th_90)

    rot_y = np.identity(4)
    th_180 = np.pi
    rot_y[0, 0] = np.cos(th_180)
    rot_y[0, 2] = np.sin(th_180)
    rot_y[2, 0] = -np.sin(th_180)
    rot_y[2, 2] = np.cos(th_180)

    # first rotate around y-axis, then final 90 deg around z-axis (apply rot_z onto rot_y)
    return rot_z @ rot_y


def plot_points_and_poses(P1, P2, relative_pose):
    start_position = np.array([0, 0])
    end_position = [0, 0]
    end_position = np.r_[end_position, 0, 1]  # add z = 0, and final 1 for homogenous coordinates for se3 multiplication
    end_position = relative_pose @ end_position

    print("start_position:", start_position)
    print("end_position:", end_position[0:2])
    T_robot_to_world = get_robot_to_world_transform()
    end_position = T_robot_to_world @ end_position
    P1 = T_robot_to_world @ P1
    P2 = T_robot_to_world @ P2

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(start_position[0], start_position[1], 'ro')
    plt.plot(end_position[0], end_position[1], 'g*')
    plt.plot(P1[0], P1[1], 'rx')

    for idx in range(P1.shape[1]):
        x1 = start_position[0]
        y1 = start_position[1]
        x2 = P1[0][idx]
        y2 = P1[1][idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)

    plt.title("Frame 1")
    plt.grid()
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(1, 2, 2)
    plt.plot(P2[0], P2[1], 'gx')
    relative_end_position = start_position  # np.linalg.inv(pose) @ end_position
    plt.plot(relative_end_position[0], relative_end_position[1], 'g*')

    for idx in range(P1.shape[1]):
        x1 = relative_end_position[0]
        y1 = relative_end_position[1]
        x2 = P2[0][idx]
        y2 = P2[1][idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)

    plt.title("Frame 2")
    plt.grid()
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/data/landmark-dewarping/toy_kinematics.png")
    plt.close()


def get_and_compare_estimates(live_points, previous_points):
    v, theta_R = get_motion_estimate_from_svd(live_points, previous_points, weights=np.ones(live_points.shape[1]))
    x_y_theta_estimate = [v[0], v[1], theta_R]
    return x_y_theta_estimate


def get_points_with_added_noise(points, noise_standard_deviation=0.1):
    noisy_points = np.array(points)
    noisy_points[0] += np.random.normal(0, noise_standard_deviation, noisy_points.shape[1])
    noisy_points[1] += np.random.normal(0, noise_standard_deviation, noisy_points.shape[1])
    return noisy_points


def get_warped_points(points_to_warp, warping_factor_x=1.2, warping_factor_y=1.2):
    warped_points = np.array(points_to_warp)
    warped_points[0] *= warping_factor_x
    warped_points[1] *= warping_factor_y
    return warped_points


def generate_simple_error_metrics(x_y_theta_gt, x_y_theta_estimate):
    errors = np.array(x_y_theta_gt) - np.array(x_y_theta_estimate)
    translational_error = np.linalg.norm(np.array([errors[0], errors[1]]))
    logger.info(f'True/estimate translation: {x_y_theta_gt} <-> {x_y_theta_estimate}')
    logger.info(f'Translational error (m): {translational_error}')
    logger.info(f'Angular error (deg): {errors[2] * 180 / np.pi}')


def main():
    logging_level = logging.INFO
    logger.setLevel(logging_level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.info("Running kinematics on toy data...")

    T_offset = np.array([2, 0.5])
    theta_offset = 0.2
    x_y_theta_gt = [T_offset[0], T_offset[1], theta_offset]
    pose = get_transform_by_translation_and_theta(translation_x=x_y_theta_gt[0], translation_y=x_y_theta_gt[1],
                                                  theta=x_y_theta_gt[2])

    x_coords = np.array([1, 1.5, 1, -1, -1])
    y_coords = np.array([-1, 0, 1, 1, -1])
    num_points = len(x_coords)
    P1 = np.array([x_coords, y_coords, np.zeros(num_points), np.ones(num_points)])
    P2 = np.linalg.inv(pose) @ P1

    xy_points_previous = np.array([P1[0], P1[1]])
    xy_points_live = np.array([P2[0], P2[1]])

    noisy_xy_points_live = get_points_with_added_noise(xy_points_live, noise_standard_deviation=0.05)
    warped_xy_points_live = get_warped_points(noisy_xy_points_live, warping_factor_x=1.2, warping_factor_y=0.8)

    x_y_theta_estimate = get_and_compare_estimates(noisy_xy_points_live, xy_points_previous)
    x_y_theta_estimate_warped = get_and_compare_estimates(warped_xy_points_live,
                                                          xy_points_previous)

    generate_simple_error_metrics(x_y_theta_gt, x_y_theta_estimate)
    generate_simple_error_metrics(x_y_theta_gt, x_y_theta_estimate_warped)

    estimated_pose = get_transform_by_translation_and_theta(translation_x=x_y_theta_estimate_warped[0],
                                                            translation_y=x_y_theta_estimate_warped[1],
                                                            theta=x_y_theta_estimate_warped[2])
    P2[0] = warped_xy_points_live[0]
    P2[1] = warped_xy_points_live[1]
    plot_points_and_poses(P1, P2, estimated_pose)


if __name__ == "__main__":
    main()
