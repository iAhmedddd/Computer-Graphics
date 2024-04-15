#!/usr/bin/env python
# coding: utf-8

from vedo import *
import numpy as np


def RotationMatrix(theta, axis_name):
    """Calculate single rotation of theta matrix around x, y, or z."""
    c = np.cos(np.radians(theta))
    s = np.sin(np.radians(theta))
    if axis_name == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]])
    elif axis_name == 'y':
        rotation_matrix = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]])
    elif axis_name == 'z':
        rotation_matrix = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])
    return rotation_matrix


def getLocalFrameMatrix(R_ij, t_ij):
    """Returns the matrix representing the local frame."""
    T_ij = np.block([[R_ij, t_ij],
                     [np.zeros((1, 3)), 1]])
    return T_ij


def createCoordinateFrameMesh():
    """Returns the mesh representing a coordinate frame."""
    _shaft_radius = 0.05
    _head_radius = 0.10
    _alpha = 1
    x_axisArrow = Arrow(start_pt=(0, 0, 0), end_pt=(1, 0, 0), shaft_radius=_shaft_radius, head_radius=_head_radius,
                        c='red', alpha=_alpha)
    y_axisArrow = Arrow(start_pt=(0, 0, 0), end_pt=(0, 1, 0), shaft_radius=_shaft_radius, head_radius=_head_radius,
                        c='green', alpha=_alpha)
    z_axisArrow = Arrow(start_pt=(0, 0, 0), end_pt=(0, 0, 1), shaft_radius=_shaft_radius, head_radius=_head_radius,
                        c='blue', alpha=_alpha)
    originDot = Sphere(pos=[0, 0, 0], c="black", r=0.10)
    F = x_axisArrow + y_axisArrow + z_axisArrow + originDot
    return F

def main():
    # Initialize the plotter
    plotter = Plotter(interactive=False)

    # Robot arm specifications
    L1, L2, L3 = 5, 8, 3  # Lengths of the robot arm segments
    joint_angles = [[30, -10, 0], [30, -10, 0], [30, -10, 0]]

    axes_dict = dict(xrange=(-5, 20), yrange=(-10, 10), zrange=(-5, 15),
                     xtitle="X-axis", ytitle="Y-axis", ztitle="Z-axis")

    all_objects = []  # List to hold all visualiza M Vtion objects
    for angles in joint_angles:
        T_01, T_12, T_23 = calculate_transformations(angles, L1, L2)
        Frame1 = draw_segment(T_01, L1, "yellow")
        Frame2 = draw_segment(T_01 @ T_12, L2, "red")
        Frame3 = draw_segment(T_01 @ T_12 @ T_23, L3, "green")
        all_objects.extend([Frame1, Frame2, Frame3])

    plotter.show(all_objects, axes=axes_dict, viewup="z").interactive().close()

def calculate_transformations(angles, L1, L2):
    """Calculate transformation matrices for each segment based on joint angles."""
    phi1, phi2, phi3 = angles
    # Base frame to first segment
    R_01 = RotationMatrix(phi1, 'z')
    t_01 = np.array([[3], [2], [0]])
    T_01 = getLocalFrameMatrix(R_01, t_01)

    # First to second segment
    R_12 = RotationMatrix(phi2, 'z')
    t_12 = np.array([[L1], [0], [0]])
    T_12 = getLocalFrameMatrix(R_12, t_12)

    # Second to third segment (end-effector)
    R_23 = RotationMatrix(phi3, 'z')
    t_23 = np.array([[L2], [0], [0]])
    T_23 = getLocalFrameMatrix(R_23, t_23)

    return T_01, T_12, T_23

def draw_segment(T, length, color):
    """Draw a segment of the robot arm."""
    pos = T[:3, 3]  # Extract translation component
    rot = T[:3, :3]  # Extract rotation component
    orientation = np.arctan2(rot[1, 0], rot[0, 0])  # Calculate orientation from rotation matrix
    segment = Cylinder(pos=pos, height=length, r=0.4, c=color, axis=(np.cos(orientation), np.sin(orientation), 0))
    segment.apply_transform(T)
    return segment

if __name__ == '__main__':
    main()