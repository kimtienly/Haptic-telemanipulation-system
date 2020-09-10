"""
Franka-Emika Panda 7 DoF robot arm with 2-finger gripper class.
Define specific simulation and behavior.

"""

from simulation.robots import RobotArmBase
import numpy as np
import math


class PandaArm(RobotArmBase):
    def __init__(self, sim_client,
                 joint_rest_angles=(0.0, 0.2758, 0.0, -
                                    2.0917, 0.0, 2.3341, 0.7854),
                 finger_rest_position=(0.04, 0.04)):

        super().__init__()

        self.num_dofs = 7
        self.num_fingers = 2

        # Joint angle limits (rad)
        self.q_min = (-2.8973, -1.7628, -2.8973, -
                      3.0718, -2.8973, -0.0175, -2.8973)
        self.q_max = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
        self.joint_ranges = tuple(np.subtract(self.q_max, self.q_min))
        # Joint velocity limits (rad/sec)
        self.dq_max = (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100)
        # Joint acceleration limits (rad/sec^2)
        self.ddq_max = (15., 7.5, 10., 12.5, 15., 20., 20.)
        # Joint jerk limits (rad/sec^3)
        self.dddq_max = (7500., 3750., 5000., 6250., 7500., 10000., 10000)

        # Joint torque limits (Nm)
        self.tau_max = (87., 87., 87., 87., 12., 12., 12)
        # Angular impact limits (Nm/sec)
        self.dtau_max = (1000.,) * 7

        # Indices
        self.joint_indices = (0, 1, 2, 3, 4, 5, 6)
        self.finger_indices = (9, 10)
        self.all_indices = self.joint_indices + self.finger_indices
        self.sensor_indices = self.all_indices
        self.ee_idx = 11  # panda_grasptarget

        self.joint_rest_angles = joint_rest_angles
        self.finger_rest_position = finger_rest_position
        self.reset_positions = self.joint_rest_angles + self.finger_rest_position

        self.sim_client = sim_client
        self.grasping = False

    def load_robot(self, position=(0, 0, 0), orientation=(0, 0, math.pi / 2),
                   model_file="franka_panda/panda.urdf", use_fixed_base=True, flags=None, global_scaling=1.):
        """
        Loads the panda robot arm into the given simulation client.
        param position: Position of robot base joint
        param orientation: Orientation of the robot base joint
        param model_file: Default pybullet urdf "franka_panda/panda.urdf"
        param use_fixed_base: Base is fixed for stationary robots eg. arms fixed to a table
        param flags: pybullet simulation flags (check pybullet documentation)
        param global_scaling: float scaling factor of robot. 1.0 is regular size
        """
        super().load_robot(position, orientation, model_file,
                           use_fixed_base, flags, global_scaling)

    def step(self):
        """
        Updates the robot state information.
        """
        self.curr_q, self.curr_dq, self.curr_joint_forces, self.joint_torque = tuple(zip(*self.sim_client.getJointStates(bodyUniqueId=self.robot_id,
                                                                                                                         jointIndices=self.joint_indices)))

        self.curr_fing_pos, self.curr_fing_vel, self.curr_fing_forces, self.prev_cmd_fing = tuple(zip(*self.sim_client.getJointStates(bodyUniqueId=self.robot_id,
                                                                                                                                      jointIndices=self.finger_indices)))

        _, _, _, _, self.curr_x, self.curr_orn, self.curr_dx, self.curr_dorn = tuple(zip(*self.sim_client.getLinkStates(bodyUniqueId=self.robot_id, linkIndices=self.joint_indices,
                                                                                                                        computeLinkVelocity=1)))
        _, _, _, _, self.curr_ee_x, self.curr_ee_orn, self.curr_ee_dx, self.curr_ee_dorn = self.sim_client.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.ee_idx,
                                                                                                                        computeLinkVelocity=1)

    def move_via_ee_controller(self, position, orientation, gripper):
        """
        Move the robot using end effector position (3D) and orientation (3D) commands.
        Also includes gripper control
        param position: Desired end effector position
        param orientation: Desired End effector orientation
        param gripper: Desired gripper action (+: open, -: close)
        return: None
        """
        # Low-pass filter
        fc = 1000.0
        T = 1/60
        K = 0.0001592
        for i in range(3):
            self.des_ee_x[i] = self.curr_ee_x[i] * \
                math.exp(-2*math.pi*fc*T) + position[i]*2*math.pi*K*fc
            tmp = self.sim_client.getEulerFromQuaternion(self.curr_ee_orn)
            self.des_ee_rot[i] = tmp[i] * \
                math.exp(-2*math.pi*fc*T) + orientation[i]*2*math.pi*K*fc

        joint_positions = self.calculate_ik(
            self.des_ee_x, self.sim_client.getQuaternionFromEuler(self.des_ee_rot))
        self.joint_control(
            joint_positions + np.array([0.] * 7 + [gripper] * 2))
        self.grasping = True if gripper < 0 else False
        self.step()
        if self.logger.isLogging:
            self.logger.logData(self)

    def calculate_ik(self, position, orientation, num_iter=5):
        """
        Calculate joint positions using Inverse Kinematics
        param position: desired position
        param orientation: desired orientation
        param num_iter: Number of iterations of IK solver
        return: tuple of joint positions
        """
        return self.sim_client.calculateInverseKinematics(bodyUniqueId=self.robot_id,
                                                          endEffectorLinkIndex=self.ee_idx,
                                                          targetPosition=position,
                                                          targetOrientation=orientation,
                                                          lowerLimits=self.q_min,
                                                          upperLimits=self.q_max,
                                                          restPoses=self.joint_rest_angles,
                                                          jointRanges=self.joint_ranges,
                                                          maxNumIterations=num_iter)

    def objectTouch(self, objectId):
        """
        Check whether the robot is touching a object.
        param objectID: the ID value of the object to check 
        return: information of the contact
        """
        return self.sim_client.getContactPoints(
            self.robot_id, objectId)

    def get_Jacobian(self):
        """
        return: full Jacobian matrix that includes both linear and angular Jacobian.
        """
        state_matrix = np.array([np.array(
            self.sim_client.getJointState(bodyUniqueId=self.robot_id, jointIndex=jointIndex))
            for jointIndex in self.joint_indices])
        q = state_matrix[:, 0]
        jac_t, jac_r = self.sim_client.calculateJacobian(self.robot_id, self.ee_idx,
                                                         [0., 0., 0.], list(q) + [0.] * 2, [0.] * 9, [0.] * 9)
        J = np.concatenate(
            (np.array(jac_t)[:, :7], np.array(jac_r)[:, :7]), axis=0)
        return np.array(jac_t)[:, :7]

    def get_force(self, scale_force=0.003):
        """
        return: contact force value
        """
        state_matrix = np.array([np.array(
            self.sim_client.getJointState(bodyUniqueId=self.robot_id, jointIndex=jointIndex))
            for jointIndex in self.joint_indices])
        q = state_matrix[:, 0]
        torque_j = np.array(state_matrix[:, 3], dtype=np.float)
        jac_t, jac_r = self.sim_client.calculateJacobian(self.robot_id, self.ee_idx,
                                                         [0., 0., 0.], list(q) + [0.] * 2, [0.] * 9, [0.] * 9)

        J = np.concatenate(
            (np.array(jac_t)[:, :7], np.array(jac_r)[:, :7]), axis=0)

        try:
            JwJ_reg = (J.T).dot(J)
            self.ee_force = np.linalg.solve(JwJ_reg, torque_j)
            self.ee_force = J.dot(self.ee_force)
            self.ee_force = self.ee_force[:3]*scale_force
            return self.ee_force
        except:
            return self.ee_force

    def isGrasping(self):
        """
        return: grasping state of the gripper
        """
        return self.grasping
