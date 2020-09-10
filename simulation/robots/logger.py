"""
RobotLogger class for logging robot data.

"""

import math
import numpy as np
import os
import csv


class RobotLogger:

    def __init__(self):

        self.isLogging = False
        self.force_feedback_list = []
        self.cart_pos_list = []
        self.cart_rot_list = []
        self.joint_torque_list = []
        self.joint_vel_list = []
        self.joint_pos_list = []
        self.gripper_list = []

        self.force_feedback = []
        self.cart_pos = []
        self.cart_rot = []

        self.des_c_pos = []
        self.des_c_rot = []

        self.joint_torque = []
        self.joint_vel = []
        self.joint_pos = []
        self.gripper = []
        self.maxTimeSteps = 5000000000000000

        self.isRecording = False
        self.recordedFile = None
        self.recordedData = []
        self.timestamp = 0

    def startLogging(self):
        self.cart_pos_list = []
        self.force_feedback_list = []
        self.force_feedback = []
        self.cart_pos = []
        self.des_c_pos = []
        self.cart_rot = []
        self.des_c_rot = []
        self.joint_torque = []
        self.joint_vel = []
        self.joint_pos = []
        self.gripper = []

    def stopLogging(self):
        self.isLogging = False
        if len(self.cart_pos_list) > 0:
            self.cart_pos = np.array(self.cart_pos_list)
            self.cart_rot = np.array(self.cart_rot_list)
            self.force_feedback = np.array(self.force_feedback_list)
            self.des_c_pos = np.array(self.des_c_pos)
            self.des_c_rot = np.array(self.des_c_rot)
            self.joint_torque = np.array(self.joint_torque_list)
            self.joint_vel = np.array(self.joint_vel_list)
            self.joint_pos = np.array(self.joint_pos_list)
            self.gripper = np.array(self.gripper_list)

    def logData(self, robot):
        """
        Record robot states with the most recent [self.maxTimeSteps] values.
        """
        self.cart_pos_list.append(robot.curr_ee_x)
        self.cart_rot_list.append(robot.curr_ee_orn)
        self.force_feedback_list.append(robot.ee_force)
        self.des_c_pos.append(robot.des_ee_x)
        self.des_c_rot.append(
            robot.sim_client.getQuaternionFromEuler(robot.des_ee_rot))
        self.joint_torque_list.append(robot.joint_torque)
        self.joint_vel_list.append(robot.curr_dq)
        self.joint_pos_list.append(robot.curr_q)
        self.gripper_list.append(robot.grasping)
        if (len(self.cart_pos_list) > self.maxTimeSteps):
            self.cart_pos_list.pop(0)
            self.cart_rot_list.pop(0)
            self.force_feedback_list.pop(0)
            self.des_c_pos.pop(0)
            self.des_c_rot.pop(0)
            self.joint_torque_list.pop(0)
            self.joint_vel_list.pop(0)
            self.joint_pos_list.pop(0)
            self.gripper_list.pop(0)

    def getEnergy(self):
        """
        return: the energy taken throughout the process
        """
        energy = 0
        for i in range(len(self.joint_torque)):
            energy += self.joint_torque[i]*self.joint_vel[i]
        return energy

    def plot(self):
        """
        Plot timestamped Cartesian positions, rotations and force feedback
        """
        import matplotlib.pyplot as plt
        if self.isLogging:
            self.stopLogging()

        cart_pos_fig = plt.figure()
        cart_rot_fig = plt.figure()
        force_feedback_fig = plt.figure()

        for j in range(3):
            plt.figure(cart_pos_fig.number)
            plt.subplot(3, 1, j + 1)
            plt.plot(self.cart_pos[:, j])
            plt.plot(self.des_c_pos[:, j], 'r')

            plt.figure(force_feedback_fig.number)
            plt.subplot(3, 1, j + 1)
            plt.plot(self.force_feedback[:, j])

        for j in range(4):
            plt.figure(cart_rot_fig.number)
            plt.subplot(4, 1, j + 1)
            plt.plot(self.cart_rot[:, j])
            plt.plot(self.des_c_rot[:, j], 'r')

        # give title to plots
        plt.figure(cart_pos_fig.number)
        plt.title(' Cartesian positions ')
        plt.figure(cart_rot_fig.number)
        plt.title(' Cartesian rotations ')
        plt.figure(force_feedback_fig.number)
        plt.title(' Force feedback ')

        plt.show()

    def save(self, path):
        """
        Save timestamped joint position, joint velocity, joint torque and gripper values
        param path: destination to save
        """
        data = []
        header = ["Joint position", "Joint velocity",
                  "Joint torque", "Gripper"]
        data.append(header)
        for i in range(0, len(self.joint_pos), 1):
            data.append([np.round(self.joint_pos[i], 4), np.round(self.joint_vel[i], 4),
                         np.round(self.joint_torque[i], 4), self.gripper[i]])
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print("Robot logger saved")
