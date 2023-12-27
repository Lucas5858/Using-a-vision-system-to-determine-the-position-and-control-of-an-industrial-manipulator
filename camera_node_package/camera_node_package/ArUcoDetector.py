#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
import rclpy
from rclpy.node import Node, ParameterDescriptor
from rclpy.parameter import ParameterType
from .constants import ARUCO_DICT, CAMERA_MATRIX, DIST_COEFFS
from .draw_marker_pose import draw_markers_pose, rotation_matrix_to_euler_angles
import cv2
import cv2.aruco as aruco
import time
import os
import numpy as np
from enum import Enum
from camera_interfaces.msg import Marker, MarkerList

# Define the ArUcoDetector class, a ROS2 node for detecting ArUco markers
class ArUcoDetector(Node):
    def __init__(self, camera_id=6, dict_name="DICT_5X5_50", detection_time_threshold=0.1, removal_time_threshold=0.3, camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS, min_valid_distance=0.0, max_valid_distance=0.0, display_preview = True):
        super().__init__("aruco_detector_node")
        
        # Declare and get ROS2 parameters
        self.declare_parameter('camera_id', camera_id)
        self.declare_parameter('dict_name', dict_name)
        self.declare_parameter('min_valid_distance', min_valid_distance, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('max_valid_distance', max_valid_distance, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('display_preview', display_preview)
        
        camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        dict_name = self.get_parameter('dict_name').get_parameter_value().string_value
        min_valid_distance = self.get_parameter('min_valid_distance').get_parameter_value().double_value
        max_valid_distance = self.get_parameter('max_valid_distance').get_parameter_value().double_value
        display_preview = self.get_parameter('display_preview').get_parameter_value().bool_value
        
        # Initialize camera and ArUco detection settings
        self.camera = cv2.VideoCapture(camera_id)  # Adjust the camera index as needed
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.selected_dict = ARUCO_DICT.get(dict_name, None)
        self.parameters = aruco.DetectorParameters()
        self.marker_history = {}
        self.detection_time_threshold = detection_time_threshold
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = 1.0
        self.min_valid_distance = min_valid_distance if min_valid_distance != 0.0 else None
        self.max_valid_distance = max_valid_distance if max_valid_distance != 0.0 else None
        self.display = display_preview
        self.removal_time_threshold = removal_time_threshold
        self.valid_detections = {}
        self.last_seen = {}
        self.publisher = self.create_publisher(MarkerList, 'valid_markers', 10)
        self.timer = self.create_timer(0.034, self.timer_callback)

    # Estimate the pose of detected markers
    def estimate_pose(self, corners, ids, marker_length):
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera calibration parameters are not set.")

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, self.camera_matrix, self.dist_coeffs)
        marker_data = []
        if ids is None:
            return []
        
        for i in range(len(ids)):
            distance = np.linalg.norm(tvecs[i][0])
            if (self.min_valid_distance is None or distance >= self.min_valid_distance) and (self.max_valid_distance is None or distance <= self.max_valid_distance):
                marker_data.append((ids[i][0], corners[i], rvecs[i], tvecs[i]))
        return marker_data
    
    # Process each video frame to detect and handle markers
    def process_video_frame(self):
        ret, frame = self.camera.read()
        if ret:
            markers, ids = self.detect_markers(frame)
            marker_data = self.estimate_pose(markers, ids, self.marker_length)
            self.verify_and_capture(marker_data)
            
            if self.display:
                marker_array = list(self.valid_detections.items())
                ids = []
                corners = []
                rvecs = []
                tvecs = []
                for id, (corner, rvec, tvec) in marker_array:
                    ids.append(id)
                    corners.append(corner)
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                self.draw_markers(frame, corners, ids, rvecs, tvecs, self.camera_matrix, self.dist_coeffs, self.marker_length)
                self.display_frame(frame)

    # Display the processed frame
    def display_frame(self, frame):
        cv2.imshow('Camera Preview', frame)
        cv2.waitKey(1)

    # Detect ArUco markers in the given frame
    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = None, None

        if self.selected_dict:
            corners, ids, _ = aruco.detectMarkers(gray, aruco.getPredefinedDictionary(self.selected_dict), parameters=self.parameters)
        else:
            for key in ARUCO_DICT:
                corners, ids, _ = aruco.detectMarkers(gray, aruco.getPredefinedDictionary(ARUCO_DICT[key]), parameters=self.parameters)
                if ids is not None:
                    break  # Stop if any markers are found

        return corners, ids

    # Verify and capture valid marker data
    def verify_and_capture(self, marker_data):
        current_time = time.time()

        # Update valid detections and last seen time
        detected_ids = [md[0] for md in marker_data]
        for id, corner, rvec, tvec in marker_data:
            self.last_seen[id] = current_time  # Update last seen time
            if id not in self.marker_history:
                self.marker_history[id] = current_time
            else:
                if current_time - self.marker_history[id] >= self.detection_time_threshold:
                    self.valid_detections[id] = (corner, rvec, tvec)  # Update or add new detection

        # Remove markers from valid detections based on last seen time
        for id in list(self.marker_history.keys()):
            if id not in detected_ids and current_time - self.last_seen[id] >= self.removal_time_threshold:
                self.valid_detections.pop(id, None)
                del self.marker_history[id]
                del self.last_seen[id]

    # Save an image of the current frame
    def save_image(self, frame):
        timestamp = int(time.time())
        filename = os.path.join(self.image_save_path, f"marker_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

    # Draw detected markers and their poses on the frame
    def draw_markers(self, frame, corners, ids, rvec, tvec, camera_matrix, dist_coeffs, marker_length):
        ids = np.array(ids).reshape((len(ids), 1))
        draw_markers_pose(image=frame, corners_list=corners, ids=ids, rvec_list=rvec, tvec_list=tvec, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=marker_length)

    # Callback function for the ROS2 timer
    def timer_callback(self):
        self.process_video_frame()
        valid_markers = self.valid_detections
        markers = []
        for id, (corners, rvec, tvec) in valid_markers.items():
            marker = Marker()
            marker.id = int(id)
            
            rot_mtx_t = cv2.Rodrigues(rvec[0])[0].T
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rot_mtx_t)
            
            marker.rotation.x = float(roll)
            marker.rotation.y = float(pitch)
            marker.rotation.z = float(yaw)
            
            marker.translation.x = float(100*tvec[0][0])
            marker.translation.y = float(100*tvec[0][1])
            marker.translation.z = float(100*tvec[0][2])
            markers.append(marker)
        
        message = MarkerList()
        message.header.stamp = self.get_clock().now().to_msg()
        message.marker_list = markers
        self.publisher.publish(message)

    # Clean up resources when the node is destroyed
    def destroy_node(self):
        cv2.destroyAllWindows()
        self.camera.release()

# Main function to initialize and run the ROS2 node
def main():
    rclpy.init()
    node = ArUcoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# Entry point of the script
if __name__ == '__main__':
    main()
