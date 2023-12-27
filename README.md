# Using a vision system to determine the position and control of an industrial manipulator

### This repository contains a source code for an engineering thesis of Łukasz Szyszka.

Some parts of the code, especially from camera_node_package folder originate from the repository created by Michał Święciło:
https://github.com/SwieciloM/Rover_Vision_System

Folder Final inverse kinematics - unused contains the code that is responsible for solving an inverse kinematics task for UR3/CB3 robot, based on Denavit-Hartenberg (DH) parametrs. It is not used in the final version of the project, since the code sometimes creates 'risky' poses for the robot.
This part of the code is based on the repository:
https://github.com/MichaelRyanGreer/Instructional/blob/main/inverse_kinematics/inverse_kinematics_DH_parameters.ipynb
