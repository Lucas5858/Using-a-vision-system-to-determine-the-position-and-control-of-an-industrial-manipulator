#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from camera_interfaces.msg import MarkerList, Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import numpy as np
from ur_msgs.srv import SetIO

# Define the MyController class, a ROS2 node for robotic arm control
class MyController(Node):
    def __init__(self):
        super().__init__('my_controller')
        # Declare and initialize parameters for joint positions and control functions
        self.declare_parameters(
            namespace='',
            parameters=[
                ('shoulder_pan_joint', 0.0),
                ('shoulder_lift_joint', -1.5708),
                ('elbow_joint', 0.0),
                ('wrist_1_joint', -1.5708),
                ('wrist_2_joint', 0.0),
                ('wrist_3_joint', 0.0),
                ('function_choice', 1),
                ('log_marker_data', False),
                ('marker_to_follow', 2)
            ]
        )

        # Create a publisher for joint trajectories
        self.publisher_ = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Read the function_choice parameter
        self.function_choice = self.get_parameter('function_choice').value

        # Subscribe to marker data and set up callback
        self.subscription = self.create_subscription(
            MarkerList,
            'valid_markers',
            self.listener_callback,
            10)

        # Initialize data structures for storing and processing marker data
        # Dictionary to store marker data
        self.marker_data = {}
        # Rotation Matrix and Translation Vector
        self.rotation_matrix = None
        self.translation_vector = None
        
        # Set up parameter for marker 
        self.marker_to_follow = self.get_parameter('marker_to_follow').value

        # Set up different functionalities based on the function_choice parameter
        if self.function_choice == 0:
            # Set up a timer for alternating trajectories
            self.alternating_trajectory_timer = self.create_timer(5, self.alternate_trajectory)
        elif self.function_choice == 1:
            # Set up a timer for looping trajectories
            self.looping_trajectory_timer = self.create_timer(5, self.looping_trajectory)
        elif self.function_choice == 2:
            # Set up a service client for a gripper
            self.client = self.create_client(SetIO, '/io_and_status_controller/set_io')
            while not self.client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service for a gripper not available, waiting...')
            self.pick_and_place_running = False      

        # Callback group for the timer
        self.callback_group = ReentrantCallbackGroup()

        # Create a timer for logging marker data
        self.timer = self.create_timer(5, self.log_marker_data, callback_group=self.callback_group)
        self.get_logger().info('Node created, waiting to detect initial markers: 1, 3, 5')

    # Callback function for processing incoming marker data
    def listener_callback(self, msg):
        time_stamp = Time.from_msg(msg.header.stamp).nanoseconds
        for marker in msg.marker_list:
            self.store_marker_data(marker, time_stamp)
            
        # Try to update rotation matrix and translation vector
        self.update_camera_transformation()
        
        if self.function_choice == 2 and all(mid in self.marker_data for mid in [6, 8]) and not self.rotation_matrix is None and not self.translation_vector is None:
            # Marker 6
            if 6 in self.marker_data and time_stamp - self.marker_data[6]['last_seen'] > 5e9 and time_stamp - self.marker_data[8]['last_seen'] <1e8:
                self.pick_and_place_init(6)
            # Marker 8
            if 8 in self.marker_data and time_stamp - self.marker_data[8]['last_seen'] > 5e9 and time_stamp - self.marker_data[6]['last_seen'] <1e8:
                self.pick_and_place_init(8)
        elif self.function_choice == 3:
            if not self.rotation_matrix is None and not self.translation_vector is None:
                if self.marker_to_follow in self.marker_data:
                    position = self.transform_point(self.marker_data[self.marker_to_follow]['translation'])
                    self.get_logger().info(f'Marker\'s {self.marker_to_follow} last position: {position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f} [mm]')
                else:
                    self.get_logger().info(f'Marker with id {self.marker_to_follow} has not been detected yet')
            else:
                self.get_logger().info('Waiting to detect initial markers...')

    # Function to alternate between two predefined trajectories
    def alternate_trajectory(self):
        if not hasattr(self, 'toggle'):
            self.toggle = False
        self.toggle = not self.toggle

        if self.toggle:
            joint_values = [0.0000, -1.5708, 0.0000, -1.5708, 0.0000, 0.0000]
        else:
            joint_values = [-1.1894, -1.5952, -2.0535, -1.0609, 1.5757, 3.5284]
            
        self.publish_trajectory(joint_values)
        
    # Function to loop through a series of predefined trajectories
    def looping_trajectory(self):
        if not hasattr(self, 'curr_point'):
            self.curr_point = 0
        
        if not self.rotation_matrix is None and not self.translation_vector is None:
            joint_values = [
                            [0.0000, -1.5708, 0.0000, -1.5708, 0.0000, 0.0000],     # Home
                            [-1.1894, -1.5952, -2.0535, -1.0609, 1.5757, 3.5284],   # Marker 3 above
                            [-1.1900, -1.945, -2.2623, -0.5030, 1.5738, 3.5308],    # Marker 3 directly
                            [-1.1894, -1.5952, -2.0535, -1.0609, 1.5757, 3.5284],   # Marker 3 above
                            [0.3815, -1.5956, -2.0525, -1.0638, 1.5750, 5.0100],    # Marker 1 above
                            [0.3808, -1.9441, -2.2602, -0.5091, 1.5718, 5.1047],    # Marker 1 directly
                            [0.3815, -1.5956, -2.0525, -1.0638, 1.5750, 5.0100],    # Marker 1 above
                            [1.1669, -1.5951, -2.0525, -1.0652, 1.5758, 5.8846],    # Marker 5 above
                            [1.1666, -1.9436, -2.2618, -0.5072, 1.5739, 5.8872],    # Marker 5 directly
                            [1.1669, -1.5951, -2.0525, -1.0652, 1.5758, 5.8846],    # Marker 5 above
                        ]
            
            # Send given point and update counter
            self.publish_trajectory(joint_values[self.curr_point])
            self.curr_point = self.curr_point + 1 if self.curr_point < len(joint_values) - 1 else 0
            
    # Initialize pick and place operation for a specific marker
    def pick_and_place_init(self, marker_id):
        if not self.pick_and_place_running:
            self.pick_and_place_running = True
            self.pick_and_place_state = 0
            self.marker_id = marker_id
            self.pick_and_place_timer = self.create_timer(1.0, self.pick_and_place_step)
       
    # Step through the pick and place operation    
    def pick_and_place_step(self):
        if self.marker_id == 6:
            if self.pick_and_place_state == 0:
                self.get_logger().info('Pick and place for marker 6 started')
                self.publish_trajectory([0.6243, -2.0801, -1.2128, -1.4190, 1.5706, 5.3397], time_to_reach=5)
                self.update_timer(6.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 1:
                self.publish_trajectory([0.6238, -2.2330, -1.4765, -1.0023, 1.5710, 5.3407], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 2:
                self.switch_gripper(True)
                self.update_timer(1.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 3:
                self.publish_trajectory([0.6243, -2.0801, -1.2128, -1.4190, 1.5706, 5.3397], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 4:
                self.publish_trajectory([1.5134, -1.8237, -1.5809, -1.3083, 1.5739, 6.2298])
                self.update_timer(5.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 5:
                self.publish_trajectory([1.5130, -2.0221, -1.8382, -0.8526, 1.5738, 6.2315], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 6:
                self.switch_gripper(False)
                self.update_timer(1.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 7:
                self.publish_trajectory([1.5134, -1.8237, -1.5809, -1.3083, 1.5739, 6.2298], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 8:
                self.publish_trajectory([0.0000, -1.5708, 0.0000, -1.5708, 0.0000, 0.0000], time_to_reach=5)
                self.update_timer(6.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 9:
                self.get_logger().info('Pick and place for marker 6 finished')
                current_time = self.get_clock().now().nanoseconds
                self.marker_data[6]['last_seen'] = current_time
                self.marker_data[8]['last_seen'] = current_time
                self.pick_and_place_running = False
                self.pick_and_place_timer.cancel()
                
        elif self.marker_id == 8:
            if self.pick_and_place_state == 0:
                self.get_logger().info('Pick and place for marker 8 started')
                self.publish_trajectory([1.5134, -1.8237, -1.5809, -1.3083, 1.5739, 6.2298], time_to_reach=5)
                self.update_timer(6.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 1:
                self.publish_trajectory([1.5130, -2.0221, -1.8382, -0.8526, 1.5738, 6.2315], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 2:
                self.switch_gripper(True)
                self.update_timer(1.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 3:
                self.publish_trajectory([1.5134, -1.8237, -1.5809, -1.3083, 1.5739, 6.2298], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 4:
                self.publish_trajectory([0.6243, -2.0801, -1.2128, -1.4190, 1.5706, 5.3397])
                self.update_timer(5.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 5:
                self.publish_trajectory([0.6238, -2.2330, -1.4765, -1.0023, 1.5710, 5.3407], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 6:
                self.switch_gripper(False)
                self.update_timer(1.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 7:
                self.publish_trajectory([0.6243, -2.0801, -1.2128, -1.4190, 1.5706, 5.3397], time_to_reach=2)
                self.update_timer(3.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 8:
                self.publish_trajectory([0.0000, -1.5708, 0.0000, -1.5708, 0.0000, 0.0000], time_to_reach=5)
                self.update_timer(6.0)
                self.pick_and_place_state += 1

            elif self.pick_and_place_state == 9:
                self.get_logger().info('Pick and place for marker 8 finished')
                current_time = self.get_clock().now().nanoseconds
                self.marker_data[6]['last_seen'] = current_time
                self.marker_data[8]['last_seen'] = current_time
                self.pick_and_place_running = False
                self.pick_and_place_timer.cancel()
 
    # Update the timer for the pick and place operation
    def update_timer(self, duration):
        self.pick_and_place_timer.cancel()
        self.pick_and_place_timer = self.create_timer(duration, self.pick_and_place_step)

    # Store incoming marker data in a dictionary
    def store_marker_data(self, marker, time_stamp):
        self.marker_data[marker.id] = {
            'rotation': [marker.rotation.x, marker.rotation.y, marker.rotation.z],
            'translation': [marker.translation.x, marker.translation.y, marker.translation.z],
            'last_seen' : time_stamp
        }

    # Log current marker data if logging is enabled
    def log_marker_data(self):
        # Check if logging is enabled
        if self.get_parameter('log_marker_data').value:
            # Log the current state of the marker_data dictionary every 5 seconds
            self.get_logger().info(f'Current Marker Data: {self.marker_data}')

    # Publish a trajectory to the robotic arm
    def publish_trajectory(self, joint_values=None, time_to_reach=4):
        trajectory = JointTrajectory()
        trajectory.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        point = JointTrajectoryPoint()
        if joint_values is None:
            joint_values = [
                self.get_parameter('shoulder_pan_joint').value,
                self.get_parameter('shoulder_lift_joint').value,
                self.get_parameter('elbow_joint').value,
                self.get_parameter('wrist_1_joint').value,
                self.get_parameter('wrist_2_joint').value,
                self.get_parameter('wrist_3_joint').value
            ]
        point.positions = joint_values
        point.time_from_start.sec = time_to_reach
        trajectory.points.append(point)
        self.publisher_.publish(trajectory)
        self.get_logger().info('Publishing trajectory')
    
    # Send a request to the IO service
    def send_request(self, fun, pin, state):
        req = SetIO.Request()
        req.fun = fun
        req.pin = pin
        req.state = state
        self.client.call_async(req)
    
    # Control the gripper's state (on/off)
    def switch_gripper(self, is_on):
        self.get_logger().info('Sending request')
        if is_on:
            self.send_request(1, 16, 0.0)
            self.send_request(1, 17, 1.0)
        else:
            self.send_request(1, 17, 0.0)
            self.send_request(1, 16, 1.0)
    
    # Update the transformation matrix between camera and robot coordinates
    def update_camera_transformation(self): 
        # Bool value to check if transformation matrix has been created
        create_transformation = True if (self.rotation_matrix is None or self.translation_vector is None) else False
                
        # Check if camera points exist
        if create_transformation and all(mid in self.marker_data for mid in [1, 3, 5]):
            camera_points = [self.marker_data[mid]['translation'] for mid in [1, 3, 5]]
            robot_points = [
                [300.0, 0.0, 100.0],
                [0.0, -300.0, 100.0],
                [212.0, 212.0, 100.0]
               ]
            # Calculate rotation matrix and translation vector
            self.rotation_matrix, self.translation_vector = self.calculate_transformation(camera_points, robot_points)
            
            self.get_logger().info('Initial markers have been detected, rotation matrix and translation vector have been created')

    # Calculate the transformation matrix from camera to robot coordinates
    def calculate_transformation(self, camera_points, robot_points):
        # Ensure inputs are numpy arrays
        camera_points = np.array(camera_points)
        robot_points = np.array(robot_points)

        # Calculate centroids
        centroid_camera = np.mean(camera_points, axis=0)
        centroid_robot = np.mean(robot_points, axis=0)

        # Align points relative to centroids
        aligned_camera_points = camera_points - centroid_camera
        aligned_robot_points = robot_points - centroid_robot

        # Compute covariance matrix
        covariance_matrix = np.dot(aligned_camera_points.T, aligned_robot_points)

        # Singular Value Decomposition (SVD) to find rotation matrix
        U, _, Vt = np.linalg.svd(covariance_matrix)
        self.rotation_matrix = np.dot(Vt.T, U.T)

        # Ensure a right-handed coordinate system (determinant should be 1)
        if np.linalg.det(self.rotation_matrix) < 0:
            Vt[2, :] *= -1
            self.rotation_matrix = np.dot(Vt.T, U.T)

        # Compute translation vector
        self.translation_vector = centroid_robot - np.dot(self.rotation_matrix, centroid_camera)

        return self.rotation_matrix, self.translation_vector

    # Transform a point from camera to robot coordinates
    def transform_point(self, point):
        point = np.array(point)
        transformed_point = np.dot(self.rotation_matrix, point) + self.translation_vector
        return transformed_point

# Main function to initialize and run the ROS2 node
def main(args=None):
    rclpy.init(args=args)
    my_controller = MyController()
    executor = MultiThreadedExecutor()
    rclpy.spin(my_controller, executor=executor)
    my_controller.destroy_node()
    rclpy.shutdown()

# Entry point of the script
if __name__ == '__main__':
    main()
