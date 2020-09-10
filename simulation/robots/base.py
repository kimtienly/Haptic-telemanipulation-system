"""
RobotArmBase class for robot arm control in general
RobotLogger: base class for logging robot data.
"""


import math
import numpy as np
import pybullet as p
import os
from simulation.robots import RobotLogger


class RobotArmBase:
    def __init__(self):

        self.sim_client = None
        self.ee_idx = None
        self.num_dofs = None
        self.joint_ranges = None
        self.rest_poses = None
        self.model_file = None
        self.robot_id = None
        self.rest_poses = None
        self.all_indices = None
        self.reset_positions = None
        self.sensor_indices = None
        self.tau_max = None

        self.curr_q, self.curr_dq = None, None  # Joint angle, joint velocity
        # Finger position, finger velocity
        self.curr_fing_pos, self.curr_fing_vel = None, None
        # Current joint forces [Fx, Fy, Fz, Mx, My, Mz]
        self.curr_joint_forces, self.curr_fing_forces = None, None
        self.joint_torque, self.prev_cmd_fing = None, None  # Previous command torque

        # Link position, velocity (World)
        self.curr_x, self.curr_dx = None, None
        # Link orientation (Quat), angular velocity (World)
        self.curr_orn, self.curr_dorn = None, None
        self.curr_ee_x, self.curr_ee_dx = None, None  # Finger position, finger velocity
        # Finger orientation (Quat), angular velocity (World)
        self.curr_ee_orn, self.curr_ee_dorn = None, None
        self.curr_ee_rot = None
        self.des_ee_x, self.des_ee_rot = np.zeros((3,)), np.zeros((3,))
        self.ee_force = np.zeros((3,))

        self.logger = RobotLogger()

    def startLogging(self):
        self.logger.isLogging = True
        self.logger.startLogging()

    def stopLogging(self):
        self.logger.isLogging = False
        self.logger.stopLogging()

    def load_robot(self, position=(0, 0, 0), orientation=(0, 0, math.pi / 2),
                   model_file=None, use_fixed_base=True, flags=None, global_scaling=1.):
        """
        Loads the robot in the given physics client based on certain parameters.
        """
        self.model_file = model_file
        # Convert orientation to quaternion if euler. Ensure right dimensions for position and orientation
        if len(orientation) == 3:
            orientation = self.sim_client.getQuaternionFromEuler(orientation)
        print(position, orientation)
        assert len(position) == 3 and len(orientation) == 4

        # Check if flags are provided else use default flags
        if flags is None:
            flags = self.sim_client.URDF_USE_SELF_COLLISION | self.sim_client.URDF_USE_INERTIA_FROM_FILE

        # Load the URDF model
        self.robot_id = self.sim_client.loadURDF(fileName=self.model_file,
                                                 basePosition=position,
                                                 baseOrientation=orientation,
                                                 useFixedBase=use_fixed_base,
                                                 flags=flags,
                                                 globalScaling=global_scaling)
        # Enable joint sensors if any
        for joint_index in self.sensor_indices:
            self.sim_client.enableJointForceTorqueSensor(bodyUniqueId=self.robot_id,
                                                         jointIndex=joint_index)

        self.initialise_robot()

        return self.robot_id

    def initialise_robot(self):
        """
        This directly resets the robot to the defined initial position and velocity.
        Note: The robot will not move to the reset position, it will directly take that position
        """
        for j in range(len(self.all_indices)):
            self.sim_client.resetJointState(bodyUniqueId=self.robot_id,
                                            jointIndex=self.all_indices[j],
                                            targetValue=self.reset_positions[j])

        self.step()

    def joint_control(self, joint_positions):
        """
        Sends a control signal to the robot arm based on given joint angles.
        param joint_positions: the desired joint angle positions
        return: None
        """
        assert len(self.all_indices) >= len(self.tau_max)
        missing_max_tau = len(self.all_indices) - len(self.tau_max)

        self.sim_client.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                                  jointIndices=self.all_indices,
                                                  targetPositions=joint_positions,
                                                  controlMode=self.sim_client.POSITION_CONTROL,
                                                  forces=self.tau_max + (5 * 240.,) * missing_max_tau)

    def joint_torque_control(self, torques):
        """
        Sends the given torque command to the robot.
        param torques: the desired joint torques
        return: None
        """
        self.sim_client.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                                  jointIndices=self.all_indices,
                                                  controlMode=self.sim_client.TORQUE_CONTROL,
                                                  forces=torques)

    def step(self):
        """
        This method controls what the robot does at each time step progression. Generally this would include
        collection and update of new state data. As each robot may need to handle this differently due to different
        kinds of states, this method needs to be implemented by each robot class.
        """
        raise NotImplementedError
