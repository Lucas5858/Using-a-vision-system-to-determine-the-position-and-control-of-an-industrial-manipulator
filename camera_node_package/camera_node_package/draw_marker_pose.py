#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from math import degrees, sqrt, atan2
from typing import Optional, Union, Tuple
from .camera_calibration import load_coefficients
from .marker_detection import detect_on_image

def is_rotation_matrix(rot_mtx: np.ndarray) -> bool:
    """Checks if a matrix is a valid rotation matrix.

    Args:
        rot_mtx (array-like): Matrix to be checked.

    Returns:
        bool: Test result

    """
    should_be_identity = np.dot(np.transpose(rot_mtx), rot_mtx)
    should_be_one = np.linalg.det(rot_mtx)
    is_identity = np.allclose(should_be_identity, np.identity(rot_mtx.shape[0], dtype=rot_mtx.dtype))
    is_one = np.allclose(should_be_one, 1.0)
    return is_identity and is_one


def rotation_matrix_to_euler_angles(rot_mtx: np.ndarray) -> np.array:
    """Calculates euler angles from rotation matrix.

    Args:
        rot_mtx (array-like): Valid 3x3 rotation matrix.

    Returns:
        np.array: Rotation around x, y and z axes in radians.

    """
    if not is_rotation_matrix(rot_mtx):
        raise ValueError("Object is not a rotation matrix")

    sy = sqrt(rot_mtx[0, 0] * rot_mtx[0, 0] + rot_mtx[1, 0] * rot_mtx[1, 0])
    is_singular = sy < 1e-6

    if not is_singular:
        x_rot = atan2(rot_mtx[2, 1], rot_mtx[2, 2])
        y_rot = atan2(-rot_mtx[2, 0], sy)
        z_rot = atan2(rot_mtx[1, 0], rot_mtx[0, 0])
    else:
        x_rot = atan2(-rot_mtx[1, 2], rot_mtx[1, 1])
        y_rot = atan2(-rot_mtx[2, 0], sy)
        z_rot = 0

    return np.array([x_rot, y_rot, z_rot])

def draw_markers_pose(image, corners_list, ids, rvec_list, tvec_list, camera_matrix, dist_coeffs, marker_length):
    # Check if there are any detected markers
    if len(corners_list):
        # Loop over every detected marker's data
        for corners, id, rvec, tvec in zip(corners_list, ids, rvec_list, tvec_list):
            rvec = rvec[0]
            tvec = tvec[0]

            # Draw a square around detected marker
            cv2.aruco.drawDetectedMarkers(image, [corners], id)

            # Draw axis of the marker
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_length/2)

            # Obtain the rotation matrix to get euler angles
            rot_mtx_t = cv2.Rodrigues(rvec)[0].T
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rot_mtx_t)

            # Text to display
            tra_text = "({:.0f}, {:.0f}, {:.0f})".format(tvec[0]*10, tvec[1]*10, tvec[2]*10)
            rot_text = "({:.0f}, {:.0f}, {:.0f})".format(degrees(roll), degrees(pitch), degrees(yaw))

            # Parameters for correct text display
            x_txt, y_txt = [int(min(i)) for i in zip(*corners[0])]
            size_marker = [int(max(i) - min(i)) for i in zip(*corners[0])]
            font_scale = sum(size_marker)/450
            offset = int(size_marker[1]/10)
            color = (19, 111, 216)

            # Draw rotation and translation values
            cv2.putText(image, tra_text, (x_txt, y_txt+4*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            cv2.putText(image, rot_text, (x_txt, y_txt+7*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)