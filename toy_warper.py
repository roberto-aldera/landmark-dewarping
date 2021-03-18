import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys

sys.path.insert(-1, "/workspace/code/landmark-distortion")
from get_rigid_body_motion import get_motion_estimate_from_svd

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


def add_noise_to_points(points, noise_standard_deviation=0.1):
    noisy_points = np.array(points)
    noisy_points[0] += np.random.normal(0, noise_standard_deviation, points.shape[1])
    noisy_points[1] += np.random.normal(0, noise_standard_deviation, points.shape[1])
    return noisy_points


def main():
    print("Running kinematics on toy data...")
    T_offset = np.array([2, 0.5])
    theta_offset = 0.2
    pose = get_transform_by_translation_and_theta(translation_x=T_offset[0], translation_y=T_offset[1],
                                                  theta=theta_offset)

    x_coords = np.array([1, 1.5, 1, -1, -1])
    y_coords = np.array([-1, 0, 1, 1, -1])
    num_points = len(x_coords)
    P1 = np.array([x_coords, y_coords, np.zeros(num_points), np.ones(num_points)])
    P2 = np.linalg.inv(pose) @ P1

    xy_points_previous = np.array([P1[0], P1[1]])
    xy_points_live = np.array([P2[0], P2[1]])

    noisy_xy_points_live = add_noise_to_points(xy_points_live, noise_standard_deviation=0.05)
    P2[0] = noisy_xy_points_live[0]
    P2[1] = noisy_xy_points_live[1]

    v, theta_R = get_motion_estimate_from_svd(noisy_xy_points_live, xy_points_previous, np.ones(5))
    print("True translation:\n", T_offset, "\n estimate:", v)
    print("True rotation:", theta_offset, "\n estimate:", theta_R)

    estimated_pose = get_transform_by_translation_and_theta(translation_x=v[0], translation_y=v[1],
                                                            theta=theta_R)
    plot_points_and_poses(P1, P2, estimated_pose)


if __name__ == "__main__":
    main()
