#!/usr/bin/env python
# coding: utf-8

import numpy as np
from vedo import *
import time
import os

class RobotArm():
    def __init__(self, partLengths, arm_location):
        self.arm_location = arm_location
        self.L1, self.L2, self.L3, self.L4 = partLengths
        self.build_all_parts()

    def build_all_parts(self):
        self.Part1 = self.create_arm_part_mesh(self.L1)
        self.Part2 = self.create_arm_part_mesh(self.L2)
        self.Part3 = self.create_arm_part_mesh(self.L3)
        self.Part4 = self.create_coordinate_frame_mesh()

    def set_pose(self, Phi):
        T_01, T_02, T_03, T_04, e = self.forward_kinematics(Phi)
        self.build_all_parts()
        self.Part1.apply_transform(T_01)
        self.Part2.apply_transform(T_02)
        self.Part3.apply_transform(T_03)
        self.Part4.apply_transform(T_04)

    def rotation_matrix(self, theta, axis_name):
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        if axis_name == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis_name == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis_name == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def create_coordinate_frame_mesh(self):
        x_axis = Arrow(start_pt=(0, 0, 0), end_pt=(1, 0, 0), c='red', alpha=1).linewidth(5)
        y_axis = Arrow(start_pt=(0, 0, 0), end_pt=(0, 1, 0), c='green', alpha=1).linewidth(5)
        z_axis = Arrow(start_pt=(0, 0, 0), end_pt=(0, 0, 1), c='blue', alpha=1).linewidth(5)
        origin = Sphere(pos=[0, 0, 0], c="black", r=0.1)
        return x_axis + y_axis + z_axis + origin

    def create_arm_part_mesh(self, L):
        sphere = Sphere(r=0.4).pos(0, 0, 0).color("lightblue").alpha(.8)
        FrameArrows = self.create_coordinate_frame_mesh()
        cylinder = Cylinder(r=0.4, height=L, pos=(L / 2 + 0.4, 0, 0), c="lightblue", alpha=.8, axis=(1, 0, 0))
        return FrameArrows + cylinder + sphere

    def get_local_frame_matrix(self, R_ij, t_ij):
        return np.block([[R_ij, t_ij], [np.zeros((1, 3)), 1]])

    def forward_kinematics(self, Phi):
        radius = 0.4
        T_01 = self.get_local_frame_matrix(self.rotation_matrix(Phi[0], 'z'), self.arm_location)
        T_12 = self.get_local_frame_matrix(self.rotation_matrix(Phi[1], 'z'), np.array([[self.L1 + 2 * radius], [0], [0]]))
        T_02 = T_01 @ T_12  # Calculate T_02
        T_23 = self.get_local_frame_matrix(self.rotation_matrix(Phi[2], 'z'), np.array([[self.L2 + 2 * radius], [0], [0]]))
        T_03 = T_02 @ T_23  # Calculate T_03
        T_34 = self.get_local_frame_matrix(self.rotation_matrix(Phi[3], 'z'), np.array([[self.L3 + radius], [0], [0]]))
        T_04 = T_03 @ T_34  # Calculate T_04
        return T_01, T_02, T_03, T_04, T_04[0:3, 3]

    def jacobian_approximation(self, phi, delta=0.01):
        J = np.zeros((3, len(phi)))
        e_initial = self.forward_kinematics(phi)[-1]
        for i in range(len(phi)):
            phi_perturbed = np.copy(phi)
            phi_perturbed[i] += delta
            e_perturbed = self.forward_kinematics(phi_perturbed)[-1]
            J[:, i] = (e_perturbed - e_initial) / delta
        return J

    def update_joints(self, phi, target, learning_rate=0.1):
        e = self.forward_kinematics(phi)[-1]  # Current end effector position
        J = self.jacobian_approximation(phi)  # Jacobian matrix
        error = target - e  # Error vector
        phi += np.dot(np.linalg.pinv(J), error) * learning_rate  # Update joint angles
        return phi

    def create_goal_representation(self, target):
        return Sphere(pos=target, c='gold', r=0.5)

def place_robots_on_circumference(robot_arm_class, radius, n_robots=3, height=0, x_offset=0):
    robots = []
    for i in range(n_robots):
        angle = i * (360 / n_robots)
        radian = np.deg2rad(angle)
        x = radius * np.cos(radian) + x_offset
        y = radius * np.sin(radian)
        z = height
        robots.append(robot_arm_class(partLengths=[5, 8, 3, 0], arm_location=np.array([[x], [y], [z]])))
    return robots

def animate_robots(robots, target, steps=100, tolerance=0.1):
    video = Video("tmp.mp4", fps=4)
    axes = Axes(xrange=(0, 35), yrange=(-8, 15), zrange=(0, 10))
    floor = Box(pos=(15, 0, -0.1), length=40, width=20, height=0.1, c='gray')
    goal_representation = robots[0].create_goal_representation(target)
    plt = Plotter(bg='beige', bg2='lb', axes=10, offscreen=False, interactive=False)
    plt.show(axes, floor, goal_representation, viewup="z")

    trajectories = [[] for _ in robots]
    converged = False

    for step in range(steps):
        if converged:
            break  # Stop the animation if the algorithm has converged

        for idx, robot in enumerate(robots):
            phi = np.random.rand(4) * 360
            phi = robot.update_joints(phi, target)
            robot.set_pose(phi)
            end_effector_pos = robot.forward_kinematics(phi)[-1]
            trajectories[idx].append(end_effector_pos)
            if np.linalg.norm(end_effector_pos - target) < tolerance:
                converged = True  # Convergence condition met

        plt.clear()
        plt += axes + floor + goal_representation
        for robot, trajectory in zip(robots, trajectories):
            plt += [robot.Part1, robot.Part2, robot.Part3, robot.Part4]
            for point in trajectory:
                plt += Sphere(pos=point, r=0.1, c='red')
        plt.render()
        video.add_frame()
        time.sleep(0.1)

    video.close()
    os.system("ffmpeg -i tmp.mp4 -pix_fmt yuv420p animation.mp4")
    os.system("rm tmp.mp4")

    if converged:
        print("The algorithm has converged to the target.")
    else:
        print("The algorithm did not converge within the given steps.")

def main():
    circle_radius = 8
    robot_height = 1
    x_offset = 7
    target = np.array([5, 5, 5])
    robots = place_robots_on_circumference(RobotArm, radius=circle_radius, height=robot_height, x_offset=x_offset)
    animate_robots(robots, target, steps=100)

if __name__ == '__main__':
    main()